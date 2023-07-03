import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from glob import glob
from tqdm import tqdm
from data_processing.sl_decode_multiscale import *
from data_processing.charuco_calib import load_h5py_file, save_h5py_file
from scipy import ndimage
from pathlib import Path
from exiftool import ExifTool
from scipy import stats




def sl_decode_mutliscale_known_calib(data_dir_0, data_dir_1, calib_data, white_thred=5, black_thred=5, proj_res=(1920, 1080), img_size=(1680, 1120),
                                     sl_scales=[1, 2, 4, 8], min_depth=None, max_depth=None, reverse_calib=False, use_radius_filter=True):

    # STEP-0: generate rectification mappings
    if not reverse_calib:
        M1, M2, d1, d2 = calib_data['M0'], calib_data['M1'], calib_data['d0'], calib_data['d1']
        R, t = calib_data['R'], calib_data['t']
    else:
        M2, M1, d2, d1 = calib_data['M0'], calib_data['M1'], calib_data['d0'], calib_data['d1']
        R, t = calib_data['R'].T, -1.0 * calib_data['t']

    R1, R2, P1, P2 = cv2.stereoRectify(cameraMatrix1=M1, cameraMatrix2=M2, distCoeffs1=d1, distCoeffs2=d2, R=R, T=t, flags=cv2.CALIB_ZERO_DISPARITY,
                                       alpha=-1, imageSize=img_size, newImageSize=img_size)[0:4]
    map_x_l, map_y_l = cv2.initUndistortRectifyMap(M1, d1, R1, P1, img_size, cv2.CV_32FC1)
    map_x_r, map_y_r = cv2.initUndistortRectifyMap(M2, d2, R2, P2, img_size, cv2.CV_32FC1)

    ## STEP-1: obtain multi-scale disp maps
    all_sl_disp = []
    for scale in sl_scales:
        curr_proj_res = [proj_res[0] // scale, proj_res[1] // scale]  # resolution of the projector
        # get disp_l from sl
        n_vertical, n_horizontal = int(2 * np.ceil(np.log2(curr_proj_res[0]))), int(2 * np.ceil(np.log2(curr_proj_res[1])))
        names = np.arange(n_vertical).tolist() + np.arange(22, 22 + n_horizontal).tolist() + [44, 45]
        img_l_dir_list = [os.path.join(data_dir_0, 'sl_{}.JPG'.format(i)) for i in names]
        img_r_dir_list = [os.path.join(data_dir_1, 'sl_{}.JPG'.format(i)) for i in names]
        sl_disp_l = get_left_disp_map_from_sl(img_l_dir_list, img_r_dir_list, proj_resolution=curr_proj_res, map_x_l=map_x_l, map_y_l=map_y_l,
                                              map_x_r=map_x_r, map_y_r=map_y_r, white_thred=white_thred, black_thred=black_thred,
                                              verbose=False, img_size=img_size)
        all_sl_disp.append(sl_disp_l)

    ## STEP-2: merge the multi-scale disp maps
    merged_sl_disp = all_sl_disp[0].copy()
    for x in all_sl_disp[1:]:
        mask = np.zeros_like(merged_sl_disp)
        mask[merged_sl_disp == 0] = 1
        merged_sl_disp += x * mask

    ## STEP-3: clean the disp map
    clean_disp = disp_morphology_filter(merged_sl_disp)
    clean_disp = pcd_outlier_filter(clean_disp, P1, P2, clean=True, use_radius_filter=use_radius_filter)
    clean_disp = ndimage.median_filter(clean_disp, size=20)

    ## STEP-4: generate depth and unwarp
    depth = disp_to_depth(clean_disp, baseline=np.linalg.norm(t), focal_length=P1[0, 0])
    inv_map_x, inv_map_y = invert_map(map_x_l, map_y_l)
    unwarped_depth = cv2.remap(depth, inv_map_x, inv_map_y, cv2.INTER_NEAREST)
    if max_depth is not None:
        unwarped_depth[unwarped_depth >= max_depth] = 0
    if min_depth is not None:
        unwarped_depth[unwarped_depth <= min_depth] = 0

    ## return data
    return unwarped_depth





def extract_AF_points_as_cv_coord(img_dir, img_size=(6720, 4480)):
    ## extract AF points from exif
    with ExifTool() as et:
        exif = et.get_metadata(img_dir)
    n_valid_af_point = int(exif['MakerNotes:ValidAFPoints'])
    x_pos = exif['MakerNotes:AFAreaXPositions']
    y_pos = exif['MakerNotes:AFAreaYPositions']
    x_pos = np.array([int(a) for a in x_pos.split(' ')])[:n_valid_af_point]
    y_pos = np.array([int(a) for a in y_pos.split(' ')])[:n_valid_af_point]
    canon_coord = np.stack([x_pos, y_pos], axis=1)
    ## convert cannon coord to opencv coord
    canon_coord[:, 1] = -1.0 * canon_coord[:, 1]
    opencv_coord = canon_coord + np.array([img_size[0] // 2, img_size[1] // 2])
    return opencv_coord, n_valid_af_point



def calc_focus_dis(depth, AF_point, win_size=(30, 20), HSE_multiplier=3.5, plot_AF_area=False):
    ## calculate AF area
    w_lower, w_upper = max(0, AF_point[0] - win_size[0]), min(AF_point[0] + win_size[0], depth.shape[1])
    h_lower, h_upper = max(0, AF_point[1] - win_size[1]), min(AF_point[1] + win_size[1], depth.shape[0])
    ## extract depth from AF area
    AF_area_rr, AF_area_cc = np.meshgrid(np.arange(h_lower, h_upper), np.arange(w_lower, w_upper))
    if plot_AF_area:
        plt.imshow(depth)
        plt.plot(AF_area_cc.flatten(), AF_area_rr.flatten())
        plt.show()
    AF_area_dep = depth[AF_area_rr, AF_area_cc].flatten()
    AF_area_dep = AF_area_dep[AF_area_dep != 0]
    ## apply a huber-skip estimator to remove outliers
    MAD = stats.median_abs_deviation(AF_area_dep, scale='normal')
    if abs(MAD) > 1e-5:
        AF_area_dep = AF_area_dep[np.abs((AF_area_dep - np.median(AF_area_dep)) / MAD) <= HSE_multiplier]
    if np.isinf(AF_area_dep).any() or np.isnan(AF_area_dep).any() or AF_area_dep.shape[0] == 0:
        return np.nan
    else:
        return np.mean(AF_area_dep)




def generate_coc_map(depth, focus_dis, focal_len, f_number, pixel_size):
    """ generate the signed CoC map in pixel unit. depth and focus_dis in mm, focal_len in pixel """
    coc = ((depth - focus_dis) / depth) * (focal_len ** 2 / (f_number * (focus_dis / pixel_size - focal_len)))
    coc[depth == 0] = 0
    return coc




##---------------------------------------------------------------------------------





if __name__ == '__main__':

    part = 'part1'
    raw_data_dir = '/media/li/52F3-84AF/DP_data/{}/raw_data'.format(part)
    save_dir = '/media/li/52F3-84AF/DP_data/{}/valid_data'.format(part)
    img_size = (1680, 1120)
    pixel_size = 36 / img_size[0]

    success_log = './{}_log/success_case.txt'.format(part)
    fail_log = './{}_log/fail_case.txt'.format(part)
    for x in [success_log, fail_log]:
        open(x, 'w').close()


    ## start loop
    # all_names = sorted([x.split('/')[-1] for x in glob(os.path.join(raw_data_dir, '*/*'))])
    all_names = np.loadtxt('./{}_log/final_fail_case.txt'.format(part), dtype=str)

    for curr_name in tqdm(all_names):
        curr_dir = os.path.join(raw_data_dir, curr_name[:8], curr_name)
        print('processing: {}'.format(curr_name))

        try:
            ## ----- STEP 1: calibration
            cam_init_l, dist_init_l = np.array([[85 / pixel_size, 0, img_size[0] / 2], [0, 85 / pixel_size, img_size[1] / 2], [0, 0, 1]]), np.zeros(5)
            cam_init_r, dist_init_r = np.array([[85 / pixel_size, 0, img_size[0] / 2], [0, 85 / pixel_size, img_size[1] / 2], [0, 0, 1]]), np.zeros(5)
            calib_flags = cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_USE_INTRINSIC_GUESS
            calib_flags += cv2.CALIB_FIX_TANGENT_DIST + cv2.CALIB_FIX_K1 + cv2.CALIB_FIX_K2 + cv2.CALIB_FIX_K3
            img_dir_list_l, img_dir_list_r = sorted(glob(os.path.join(curr_dir, 'cam0/calib_*.JPG'))), sorted(glob(os.path.join(curr_dir, 'cam1/calib_*.JPG')))
            # img_dir_list_l, img_dir_list_r = img_dir_list_l[2:3], img_dir_list_r[2:3]
            meta_data = calibrate_stereo_camera([15, 11.194], [12, 9], img_dir_list_l, img_dir_list_r, cam_init_l, dist_init_l,
                                                cam_init_r, dist_init_r, calib_flags=calib_flags, vis_scale=0.5, verbose=False, img_size=img_size)
            meta_data['M0'] = meta_data.pop('M1')
            meta_data['M1'] = meta_data.pop('M2')
            meta_data['d0'] = meta_data.pop('d1')
            meta_data['d1'] = meta_data.pop('d2')


            ## ----- STEP 2: generate depth
            white_thred, black_thred = 2, 2
            sl_img_dir_0 = os.path.join(curr_dir, 'cam0')
            sl_img_dir_1 = os.path.join(curr_dir, 'cam1')
            clean_dep_0 = sl_decode_mutliscale_known_calib(sl_img_dir_0, sl_img_dir_1, meta_data, white_thred, black_thred, proj_res=(1920, 1080), img_size=img_size,
                                                           min_depth=200, max_depth=10000, reverse_calib=False, use_radius_filter=False)
            clean_dep_1 = sl_decode_mutliscale_known_calib(sl_img_dir_1, sl_img_dir_0, meta_data, white_thred, black_thred, proj_res=(1920, 1080), img_size=img_size,
                                                           min_depth=200, max_depth=10000, reverse_calib=True, use_radius_filter=False)



            ## ----- STEP 3: extract focus distance
            ## extract AF points, omit multi AF data
            win_size = (60, 40)
            down_scale = 4  # from raw resolution to 1680
            AF_point_0, _ = extract_AF_points_as_cv_coord(os.path.join(curr_dir, 'cam0/FN_22.CR2'))
            AF_point_1, _ = extract_AF_points_as_cv_coord(os.path.join(curr_dir, 'cam1/FN_22.CR2'))
            ## calculate focus distance
            AF_point_0, AF_point_1 = AF_point_0.flatten() // 4, AF_point_1.flatten() // 4
            focus_dis_0, focus_dis_1 = calc_focus_dis(clean_dep_0, AF_point_0, win_size=win_size), calc_focus_dis(clean_dep_1, AF_point_1, win_size=win_size)
            if np.isnan(focus_dis_0) or np.isnan(focus_dis_1):
                raise Exception
            meta_data['focus_dis_0'], meta_data['focus_dis_1'] = focus_dis_0, focus_dis_1
            meta_data['AF_point_0'], meta_data['AF_point_1'] = AF_point_0, AF_point_1
            meta_data['pixel_size'] = pixel_size


            ## ----- STEP 4: save res and plot
            curr_save_dir = os.path.join(save_dir, curr_name)
            Path(os.path.join(curr_save_dir, 'cam0')).mkdir(parents=True, exist_ok=True)
            Path(os.path.join(curr_save_dir, 'cam1')).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(os.path.join(curr_save_dir, 'cam0/dep.png'), np.uint16(clean_dep_0))
            cv2.imwrite(os.path.join(curr_save_dir, 'cam1/dep.png'), np.uint16(clean_dep_1))
            save_h5py_file(os.path.join(curr_save_dir, 'meta_data.h5'), meta_data)
            f = open(success_log, 'a')
            f.write(curr_name + '\n')
            f.close()
            print('focal_len: {:.2f}mm, {:.2f}mm; focus_dis: {:.2f}mm, {:.2f}mm'.format(meta_data['M0'][0, 0] * pixel_size, meta_data['M1'][0, 0] * pixel_size,
                                                                                        focus_dis_0, focus_dis_1))

            # fig, ax = plt.subplots(1, 4, figsize=(12, 4))
            # coc0 = generate_coc_map(clean_dep_0, meta_data['focus_dis_0'], meta_data['M0'][0, 0], 2, meta_data['pixel_size'])
            # coc1 = generate_coc_map(clean_dep_1, meta_data['focus_dis_1'], meta_data['M1'][0, 0], 2, meta_data['pixel_size'])
            # for i, x in enumerate([clean_dep_0, coc0, clean_dep_1, coc1]):
            #     ax[i].imshow(x)
            # plt.tight_layout()
            # plt.show()

        except:
            f = open(fail_log, 'a')
            f.write(curr_name + '\n')
            f.close()








