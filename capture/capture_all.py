
import os
import sys
import inspect
curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(curr_dir)
sys.path.insert(0, parent_dir)

import logging
import time
from sys import exit
import cv2
import gphoto2 as gp
import numpy as np
from camera_controller import BaseCameraController
from matplotlib import pyplot as plt
from tqdm import tqdm
from datetime import datetime
from data_processing.charuco_calib import calibrate_stereo_camera, save_h5py_file
import yaml
from sl_decode import get_left_disp_map_from_sl, generate_rectify_data, load_h5py_to_dict, invert_map


class Eos5dX2Controller(BaseCameraController):
    def __init__(self, device_ports, name, output_root_path):
        """
        :param list of str device_ports: serial number of connected cameras
        :param list of str name: device names
        :param str output_root_path: storage directory
        """
        super(Eos5dX2Controller, self).__init__(device_ports, name, output_root_path)
        self.init()


    def __len__(self):
        """ return the number of photos taken by one shoot. """
        return 2


    def get_serialnum(self, camera):
        text = camera.get_summary()
        slash = str(text).find('Serial Number:')
        serial = str(text)[slash:slash + 100].splitlines()[0]
        serial = serial[15:]
        return serial


    def get_summary(self):
        for i, cam in enumerate(self.camera):
            text = cam.get_summary()
            print('Summary')
            print('=======')
            print(str(text))


    def init(self):
        self.file_name_prefix_buf = []
        self.camera_file_paths_list = []
        self.camera_list = Eos5dX2Controller.available_devices()
        if len(self.camera_list) < len(self):
            print('{} camera detected'.format(len(self.camera_list)))
            exit()
        self.camera_list = self.camera_list[:len(self)]
        self.camera = []
        port_info_list = gp.PortInfoList()
        port_info_list.load()
        for i, cam in enumerate(self.camera_list):
            idx = port_info_list.lookup_path(cam[1])
            new_camera = gp.Camera()
            new_camera.set_port_info(port_info_list[idx])
            new_camera.init()
            self.camera.append(new_camera)
        # ascending order by the serial number
        self.camera.sort(key=lambda x: self.get_serialnum(x))
        self.device_ports = [self.get_serialnum(x) for x in self.camera]
        self.get_summary()


    def close(self):
        for i in range(len(self)):
            self.camera[i].exit()


    @classmethod
    def available_devices(cls):
        camera_list = []
        for name, addr in gp.check_result(gp.gp_camera_autodetect()):
            camera_list.append((name, addr))
        return camera_list


    @classmethod
    def find_port_by_name(cls, name):
        pass


    def set_to_manual_mode(self):
        """
        set camera mode to manual mode and turn off all of post processing.
        disable auto setting mode.
        TODO: Make sure this function works.
        """
        config = [0] * len(self)
        for i in range(len(self)):
            config[i] = self.camera[i].get_config()
            auto_exposure_conf = config[i].get_child_by_name("autoexposuremode")
            auto_exposure_conf.set_value("Manual")
            self.camera[i].set_config(config[i])
            time.sleep(0.5)


    def set_config(self, shutter_speed, aperture, iso, imgformat):
        """Init
        Args:
            shutter_speed ([type]): [description]
            fps ([type]): [description]
            gain (float, optional): [description]. Defaults to 0.0.
        """
        config = [0] * len(self)
        for i in range(len(self)):
            config[i] = self.camera[i].get_config()
            shutter_speed_conf = config[i].get_child_by_name("shutterspeed")
            shutter_speed_conf.set_value(str(shutter_speed))
            aperture_conf = config[i].get_child_by_name("aperture")
            aperture_conf.set_value(str(aperture))
            iso_conf = config[i].get_child_by_name("iso")
            iso_conf.set_value(str(iso))
            drivemode_conf = config[i].get_child_by_name("drivemode")
            drivemode_conf.set_value("Single")
            capturetarget_conf = config[i].get_child_by_name("capturetarget")
            capturetarget_conf.set_value("Memory card")
            format_conf = config[i].get_child_by_name("imageformat")
            format_conf.set_value(imgformat)
            self.camera[i].set_config(config[i])
            time.sleep(0.5)


    def set_aperture(self, apertures):
        config = [0] * len(self)
        for i in range(len(self)):
            config[i] = self.camera[i].get_config()
            aperture_conf = config[i].get_child_by_name("aperture")
            aperture_conf.set_value(str(apertures[i]))
            self.camera[i].set_config(config[i])
            time.sleep(0.1)


    def capture_and_get_image_path(self, idx, q):
        results = self.camera[idx].capture(gp.GP_CAPTURE_IMAGE)  # この関数呼んでからシャッター切るまでの時間はランダム
        q.put([idx, results.folder, os.path.splitext(results.name)[0]])


    def shoot(self, file_name_prefix):
        """
        take photo
        :param str file_name_prefix: any message
        :return:
        """
        self.file_name_prefix_buf.append(file_name_prefix)
        camera_file_paths = [0] * len(self)
        for i in range(len(self)):
            time.sleep(0.1)
            camera_file_paths[i] = self.camera[i].capture(gp.GP_CAPTURE_IMAGE)
            # print('camera_file_paths {}:'.format(i), camera_file_paths[i].name)
        self.camera_file_paths_list.append(camera_file_paths)


    def list_files(self, camera, path='/'):
        result = []
        # get files
        for name, value in camera.folder_list_files(path):
            result.append(os.path.join(path, name))
        # read folders
        folders = []
        for name, value in camera.folder_list_folders(path):
            folders.append(name)
        # recurse over subfolders
        for name in folders:
            result.extend(self.list_files(camera, os.path.join(path, name)))
        return result


    def transfer(self, form=['RAW', 'JPG']):
        self.close()
        for camera_file_paths, file_name_prefix in tqdm(zip(self.camera_file_paths_list, self.file_name_prefix_buf)):
            for i in range(len(self)):
                camera_output_path = os.path.join(self.output_root_path, "cam{camera_id}".format(camera_id=i))
                if not os.path.exists(camera_output_path):
                    os.makedirs(camera_output_path)
                img_name, _ = os.path.splitext(camera_file_paths[i].name)
                if 'RAW' in form:
                    try:
                        camera_file_raw = self.camera[i].file_get(camera_file_paths[i].folder, img_name + ".CR2", gp.GP_FILE_TYPE_NORMAL)
                        file_name_raw = file_name_prefix + ".CR2"
                        camera_file_raw.save(os.path.join(camera_output_path, file_name_raw))
                        self.camera[i].file_delete(camera_file_paths[i].folder, img_name + ".CR2")
                    except:
                        print('No CR2 images are provided')
                if 'JPG' in form:
                    try:
                        camera_file_rgb = self.camera[i].file_get(camera_file_paths[i].folder, img_name + ".JPG", gp.GP_FILE_TYPE_NORMAL)
                        file_name_rgb = file_name_prefix + ".JPG"
                        camera_file_rgb.save(os.path.join(camera_output_path, file_name_rgb))
                        self.camera[i].file_delete(camera_file_paths[i].folder, img_name + ".JPG")
                    except:
                        print('No JPG images are provided')


##-----------------------------------------------------------------------------------------------------


def generate_patterns(proj_resolution):
    # generate patterns
    graycode = cv2.structured_light_GrayCodePattern.create(width=proj_resolution[0], height=proj_resolution[1])
    _, patterns = graycode.generate()
    black, white = graycode.getImagesForShadowMasks(np.zeros_like(patterns[0]), np.zeros_like(patterns[0]))
    patterns = patterns + [white, black]  # horizontal, vertical, black-white
    return patterns


def stereo_calib_callback(d, save_dir, calib_flags):
    data_dir_l = [os.path.join(save_dir, 'cam0/calib_{}.JPG'.format(i)) for i in range(d['n_calib_img'])]
    data_dir_r = [os.path.join(save_dir, 'cam1/calib_{}.JPG'.format(i)) for i in range(d['n_calib_img'])]
    cam_init_l, cam_init_r = np.array(d['cam_mat_init_l']), np.array(d['cam_mat_init_r'])
    dist_init_l, dist_init_r = np.array(d['dist_init_l']).astype(float).reshape(-1, 1), np.array(d['dist_init_r']).astype(float).reshape(-1, 1)
    calib_data = calibrate_stereo_camera(d['board_size'], d['marker_division'], data_dir_l, data_dir_r,
                                         cam_init_l, dist_init_l, cam_init_r, dist_init_r, calib_flags=calib_flags,
                                         vis_scale=0.5, verbose=True, img_size=(1680, 1120))
    return calib_data


def check_num_of_AF_point(cr2_dir):
    from exiftool import ExifTool
    with ExifTool() as et:
        exif = et.get_metadata(cr2_dir)
    n_AF_point = int(exif['MakerNotes:ValidAFPoints'])
    return n_AF_point



##-----------------------------------------------------------------------------------------------------



if __name__ == '__main__':
    ## setups
    with open("capture_config.yaml", 'r') as stream:
        yml = yaml.safe_load(stream)
    save_dir = os.path.join(yml['base_dir'], datetime.now().strftime('%Y%m%d%H%M%S'))

    ## camera init
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(name)s: %(message)s', level=logging.WARNING)
    callback_obj = gp.check_result(gp.use_python_logging())
    eos5dcon = Eos5dX2Controller(None, ['Eos5d_0', 'Eos5d_1'], save_dir)
    eos5dcon.set_to_manual_mode()


    ## STEP-1: capture DP images
    assert len(yml['dp_f_number_list']) == len(yml['dp_shutter_speed_list']), 'F-numbers and Shutter-speeds are of different lengths!'
    print('---------- Start STEP-1: capture {} DP images -------------'.format(len(yml['dp_f_number_list'])))
    for f_n, ss, iso in zip(yml['dp_f_number_list'], yml['dp_shutter_speed_list'], yml['dp_iso_list']):
        eos5dcon.set_config(ss, f_n, iso, 'RAW')
        eos5dcon.shoot('FN_{}'.format(f_n))
    eos5dcon.transfer(form=['RAW'])
    n_AF_point_0 = check_num_of_AF_point(os.path.join(save_dir, 'cam0/FN_22.CR2'))
    n_AF_point_1 = check_num_of_AF_point(os.path.join(save_dir, 'cam1/FN_22.CR2'))
    assert n_AF_point_0 == 1 and n_AF_point_1 == 1, 'Wrong number of AF points! Get {} and {}'.format(n_AF_point_0, n_AF_point_1)


    ## STEP-2: capture sl images
    sl_patterns = generate_patterns(proj_resolution=yml['proj_resolution'])
    input('Starting SL capture. Please\n'
          '0. Lock the lens!!!! 1. Open the projector. 2. Focus it. 3.Turn off light \n '
          'Press Enter to continue...')
    print('---------- Start STEP-2: capture {} sl patterns -------------'.format(len(sl_patterns)))
    eos5dcon.set_config(yml['sl_shutter_speed'], yml['sl_f_number'], yml['sl_iso'], 'Smaller JPEG')  # set camera config
    for i, x in tqdm(enumerate(sl_patterns)):
        capname = '{}'.format(i)
        cv2.namedWindow(capname, cv2.WND_PROP_FULLSCREEN)
        cv2.moveWindow(capname, yml['major_screen_resolution'], 0)
        cv2.setWindowProperty(capname, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(capname, x)
        cv2.waitKey(700)
        eos5dcon.shoot('sl_{}'.format(i))
        time.sleep(0.5)
    cv2.destroyAllWindows()
    eos5dcon.transfer(form=['JPG'])



    ## STEP-3: capture calib images
    print('---------- Start STEP-3: capture {} calib images. Please turn off the projector -------------'.format(yml['n_calib_img']))
    eos5dcon.set_config(yml['calib_shutter_speed'], yml['calib_f_number'], yml['calib_iso'],
                        'Smaller JPEG')  # set camera config
    while True:
        ## do one capture
        for i in range(yml['n_calib_img']):
            input('Please setup the {}/{} pose. Press Enter to continue...'.format(i + 1, yml['n_calib_img']))
            eos5dcon.shoot('calib_{}'.format(i))
        eos5dcon.transfer(form=['JPG'])
        ## check if the current calib data is usable, if not, re-capture
        try:
            calib_flags = cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_USE_INTRINSIC_GUESS
            calib_flags += cv2.CALIB_FIX_TANGENT_DIST + cv2.CALIB_FIX_K1 + cv2.CALIB_FIX_K2 + cv2.CALIB_FIX_K3
            calib_data = stereo_calib_callback(yml, save_dir, calib_flags)
        except:
            print('Calibration failed. Please re-capture the data')
            continue
        label = input('Use the current calib result? (y/n)')
        if label in ['y', 'Y']:
            save_h5py_file(os.path.join(save_dir, 'calib.h5'), calib_data)
            break
        elif label in ['n', 'N']:
            continue

    eos5dcon.close()


    ## STEP 4: calculate disparity maps
    print('computing the disparity map')
    if yml['use_full_proj_res']:
        names = np.arange(len(sl_patterns))
        proj_res = yml['proj_resolution']
    else:
        names = np.arange(20).tolist() + np.arange(22, 42).tolist() + [44, 45]
        proj_res = (960, 600)
    img_l_dir_list = [os.path.join(save_dir, 'cam0/sl_{}.JPG'.format(i)) for i in names]
    img_r_dir_list = [os.path.join(save_dir, 'cam1/sl_{}.JPG'.format(i)) for i in names]
    calib_data = load_h5py_to_dict(os.path.join(save_dir, 'calib.h5'))
    white_img = cv2.resize(cv2.imread(img_l_dir_list[-2], 0), (1680, 1120))
    map_x_l, map_y_l, map_x_r, map_y_r, _, _, _ = generate_rectify_data(calib_data, size=(1680, 1120))
    disp_l = get_left_disp_map_from_sl(img_l_dir_list, img_r_dir_list, proj_resolution=proj_res, map_x_l=map_x_l,
                                       map_y_l=map_y_l, map_x_r=map_x_r, map_y_r=map_y_r, verbose=False,
                                       white_thred=yml['white_thred'], black_thred=yml['black_thred'],
                                       img_size=(1680, 1120))
    # reversely mapped disp map
    inv_map_x, inv_map_y = invert_map(map_x_l, map_y_l)
    unwarped_sl_disp = cv2.remap(disp_l, inv_map_x, inv_map_y, cv2.INTER_LINEAR)

    # plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].imshow(white_img)
    ax[1].imshow(unwarped_sl_disp)
    ax[0].set_title('left white image')
    ax[1].set_title('left unwarped disp map')
    plt.tight_layout()
    plt.show()