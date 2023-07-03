import os
import sys
import inspect
curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(curr_dir)
sys.path.insert(0, parent_dir)

from sl_decode_multiscale import *
from glob import glob
from tqdm import tqdm
from scipy import ndimage
from charuco_calib import save_h5py_file, load_h5py_file




def step_1_sl_decode(sl_img_dir_0, sl_img_dir_1, calib_data, white_thred=5, black_thred=5, proj_res=(1920, 1080),
                     img_size=(1680, 1120), sl_scales=[1, 2, 4, 8], min_depth=None, max_depth=None, reverse_calib=False):
    # STEP-0: generate rectification mappings
    if not reverse_calib:
        M1, M2, d1, d2 = calib_data['M1'], calib_data['M2'], calib_data['d1'], calib_data['d2']
        R, t = calib_data['R'], calib_data['t']
    else:
        M2, M1, d2, d1 = calib_data['M1'], calib_data['M2'], calib_data['d1'], calib_data['d2']
        R, t = calib_data['R'].T, -1.0 * calib_data['t']

    R1, R2, P1, P2 = cv2.stereoRectify(cameraMatrix1=M1, cameraMatrix2=M2, distCoeffs1=d1, distCoeffs2=d2, R=R, T=t,
                                       flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1, imageSize=img_size,
                                       newImageSize=img_size)[0:4]
    map_x_l, map_y_l = cv2.initUndistortRectifyMap(M1, d1, R1, P1, img_size, cv2.CV_32FC1)
    map_x_r, map_y_r = cv2.initUndistortRectifyMap(M2, d2, R2, P2, img_size, cv2.CV_32FC1)

    ## STEP-1: obtain multi-scale disp maps
    all_sl_disp = []
    for scale in sl_scales:
        curr_proj_res = [proj_res[0] // scale, proj_res[1] // scale]  # resolution of the projector
        # get disp_l from sl
        n_vertical, n_horizontal = int(2 * np.ceil(np.log2(curr_proj_res[0]))), int(2 * np.ceil(np.log2(curr_proj_res[1])))
        names = np.arange(n_vertical).tolist() + np.arange(22, 22 + n_horizontal).tolist() + [44, 45]
        img_l_dir_list = [os.path.join(sl_img_dir_0, 'sl_{}.JPG'.format(i)) for i in names]
        img_r_dir_list = [os.path.join(sl_img_dir_1, 'sl_{}.JPG'.format(i)) for i in names]
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
    clean_disp = pcd_outlier_filter(clean_disp, P1, P2, clean=True, use_radius_filter=False)
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




##--------------------------------------------------------------------



if __name__ == '__main__':


    ## setup
    white_thred, black_thred = 2, 2
    file_dir = '/Users/feiran-l/Desktop/dataset/DP_data/err_esti'
    pixel_size = 36 / 1680
    img_size = (1680, 1120)


    ## STEP 1: calib
    cam_init_l, dist_init_l = np.array([[85 / pixel_size, 0, img_size[0] / 2], [0, 85 / pixel_size, img_size[1] / 2], [0, 0, 1]]), np.zeros(5)
    cam_init_r, dist_init_r = np.array([[85 / pixel_size, 0, img_size[0] / 2], [0, 85 / pixel_size, img_size[1] / 2], [0, 0, 1]]), np.zeros(5)
    calib_flags = cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_USE_INTRINSIC_GUESS
    calib_flags += cv2.CALIB_FIX_TANGENT_DIST + cv2.CALIB_FIX_K1 + cv2.CALIB_FIX_K2 + cv2.CALIB_FIX_K3

    img_dir_list_l, img_dir_list_r = sorted(glob(os.path.join(file_dir, 'cam0/calib_*.JPG'))), sorted(glob(os.path.join(file_dir, 'cam1/calib_*.JPG')))
    meta_data = calibrate_stereo_camera([15, 11.194], [12, 9], img_dir_list_l, img_dir_list_r, cam_init_l, dist_init_l,
                                        cam_init_r, dist_init_r, calib_flags=calib_flags, vis_scale=0.5, verbose=True, img_size=img_size)
    save_h5py_file(os.path.join(file_dir, 'meta_data.h5'), meta_data)


    ## STEP 2: sl decode
    sl_img_dir_0, sl_img_dir_1 = os.path.join(file_dir, 'cam0'), os.path.join(file_dir, 'cam1')
    dep_0 = step_1_sl_decode(sl_img_dir_0, sl_img_dir_1, meta_data, white_thred, black_thred, proj_res=(1920, 1080),
                             img_size=(1680, 1120), min_depth=200, max_depth=10000, reverse_calib=False)
    dep_1 = step_1_sl_decode(sl_img_dir_1, sl_img_dir_0, meta_data, white_thred, black_thred, proj_res=(1920, 1080),
                             img_size=(1680, 1120), min_depth=200, max_depth=10000, reverse_calib=True)

    dep_0, dep_1 = np.uint16(dep_0), np.uint16(dep_1)

    cv2.imwrite(os.path.join(file_dir, 'cam0/dep.png'), dep_0)
    cv2.imwrite(os.path.join(file_dir, 'cam1/dep.png'), dep_1)

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(dep_0)
    ax[1].imshow(dep_1)
    plt.show()







