import os
import sys
import inspect
curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(curr_dir)
sys.path.insert(0, parent_dir)

import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import skimage
import open3d as o3d
from open3d import geometry as o3dg
from glob import glob
from data_processing.charuco_calib import calibrate_stereo_camera
from tqdm import tqdm
from scipy import ndimage


def disp_morphology_filter(disp, morphology_ker=np.ones((4, 4), np.uint8), edge_ker=np.ones((2, 2), np.uint8)):
    # hole filling
    disp = cv2.morphologyEx(disp, cv2.MORPH_CLOSE, morphology_ker)
    # boundary point removal
    if disp.mean() < 0:
        disp = cv2.dilate(disp, edge_ker)
    else:
        disp = cv2.erode(disp, edge_ker)
    return disp



def numpy_to_o3d(pcd_np, texture=None):
    valid_ids = (~np.isnan(pcd_np).any(axis=1)) * (~np.isinf(pcd_np).any(axis=1))
    valid_pcd = pcd_np[valid_ids]
    tmp = o3dg.PointCloud()
    tmp.points = o3d.utility.Vector3dVector(valid_pcd)
    if texture is not None:
        valid_texture = texture[valid_ids] / 255.0
        tmp.colors = o3d.utility.Vector3dVector(valid_texture)
    return tmp


def generate_rectify_data(calib_data, size):
    M1, M2, d1, d2 = calib_data['M1'], calib_data['M2'], calib_data['d1'], calib_data['d2']
    R, t = calib_data['R'], calib_data['t']
    flag = cv2.CALIB_ZERO_DISPARITY
    R1, R2, P1, P2, Q = cv2.stereoRectify(cameraMatrix1=M1, cameraMatrix2=M2, distCoeffs1=d1, distCoeffs2=d2, R=R, T=t,
                                          flags=flag, alpha=-1, imageSize=size, newImageSize=size)[0:5]
    map_x_l, map_y_l = cv2.initUndistortRectifyMap(M1, d1, R1, P1, size, cv2.CV_32FC1)
    map_x_r, map_y_r = cv2.initUndistortRectifyMap(M2, d2, R2, P2, size, cv2.CV_32FC1)
    return map_x_l, map_y_l, map_x_r, map_y_r, P1, P2, Q


def invert_map(map_x, map_y, iteration=10):
    """ https://stackoverflow.com/questions/41703210/inverting-a-real-valued-index-grid
        remember to use "INTER_NEAREST" flag in cv2.remap to unwarp depth map
    """
    F = np.zeros((map_x.shape[0], map_x.shape[1], 2), dtype=np.float32)
    F[:, :, 0], F[:, :, 1] = map_x, map_y
    I = np.zeros_like(F)
    I[:, :, 1], I[:, :, 0] = np.indices(map_x.shape)
    P = np.copy(I)
    for i in range(iteration):
        P += I - cv2.remap(F, P, None, interpolation=cv2.INTER_LINEAR)
    inv_map_x, inv_map_y = P[:, :, 0], P[:, :, 1]
    return inv_map_x, inv_map_y


def pcd_outlier_filter(disp, P1, P2, clean=True, nb_neighbors=100, std_ratio=0.3, use_radius_filter=False):
    """
    P1 and P2 are the camera matrix of the left and right rectified views
    """
    # apply disp map
    tmp_disp = disp.copy()
    tmp_disp[tmp_disp == 0] = np.nan
    grid_w, grid_h = np.meshgrid(np.arange(disp.shape[1]), np.arange(disp.shape[0]))
    grid_w, grid_h = grid_w.reshape(-1, 1), grid_h.reshape(-1, 1)
    cam_pts_l = np.concatenate([grid_w, grid_h], axis=1)
    cam_pts_r = np.concatenate([grid_w + tmp_disp[grid_h, grid_w], grid_h], axis=1)
    valid_ids = np.argwhere(~np.isnan(cam_pts_r[:, 0])).flatten()
    cam_pts_l, cam_pts_r = np.array(cam_pts_l[valid_ids, :]), np.array(cam_pts_r[valid_ids, :])
    # triangulate
    pts4D = cv2.triangulatePoints(P1, P2, np.float32(cam_pts_l[:, np.newaxis, :]), np.float32(cam_pts_r[:, np.newaxis, :])).T
    pts3D = pts4D[:, :3] / pts4D[:, -1:]
    # clean or not
    if clean:
        pcd = numpy_to_o3d(pts3D)
        # statistical outlier removal
        _, inlier_ids = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        outlier_ids = np.setdiff1d(np.arange(cam_pts_l.shape[0]), inlier_ids)
        disp[cam_pts_l[outlier_ids, 1], cam_pts_l[outlier_ids, 0]] = 0
        # radius outlier removal
        if use_radius_filter:
            _, inlier_ids = pcd.remove_radius_outlier(nb_points=50, radius=15)
            outlier_ids = np.setdiff1d(np.arange(cam_pts_l.shape[0]), inlier_ids)
            disp[cam_pts_l[outlier_ids, 1], cam_pts_l[outlier_ids, 0]] = 0
    return disp



def get_left_disp_map_from_sl(img_l_dir_list, img_r_dir_list, proj_resolution, map_x_l, map_y_l, map_x_r, map_y_r,
                              white_thred=5, black_thred=40, verbose=False, img_size=None):
    """
        img_dir_list should be of order: pattern + white + black
    """
    # decoder
    graycode = cv2.structured_light_GrayCodePattern.create(width=proj_resolution[0], height=proj_resolution[1])
    graycode.setWhiteThreshold(white_thred)
    graycode.setBlackThreshold(black_thred)
    num_required_imgs = graycode.getNumberOfPatternImages() + 2
    assert num_required_imgs == len(img_l_dir_list), 'Left list length is wrong: Require {}, Given {}'.format(num_required_imgs, len(img_l_dir_list))
    assert num_required_imgs == len(img_r_dir_list), 'Right list length is wrong: Require {}, Given {}'.format(num_required_imgs, len(img_r_dir_list))
    if verbose:
        print('num of imgs is {}'.format(num_required_imgs))

    # load data and rectify:
    rect_list_l, rect_list_r = [], []
    for i in range(num_required_imgs):
        img_l, img_r = cv2.imread(img_l_dir_list[i], 0), cv2.imread(img_r_dir_list[i], 0)
        if img_size is not None:
            img_l, img_r = cv2.resize(img_l, img_size), cv2.resize(img_r, img_size)
        l_rect, r_rect = cv2.remap(img_l, map_x_l, map_y_l, cv2.INTER_LINEAR), cv2.remap(img_r, map_x_r, map_y_r, cv2.INTER_LINEAR)
        if verbose:
            cv2.imshow('rectified data', skimage.transform.rescale(np.concatenate([l_rect, r_rect], axis=1), 0.2))
            cv2.waitKey(20)
        rect_list_l.append(l_rect)
        rect_list_r.append(r_rect)
    cv2.destroyAllWindows()
    # decoding
    pattern_list = np.array([rect_list_l[:-2], rect_list_r[:-2]])
    white_list = np.array([rect_list_l[-2], rect_list_r[-2]])
    black_list = np.array([rect_list_l[-1], rect_list_r[-1]])
    res, disp_l = graycode.decode(pattern_list, np.zeros_like(pattern_list[0]), black_list, white_list)
    # correct signs
    if disp_l.mean() < 0:
        disp_l[disp_l > 0] = 0
    else:
        disp_l[disp_l < 0] = 0
    return disp_l



def stereo_calib_callback(img_list_l, img_list_r, f_init, img_size=None, verbose=False):
    board_size = [15, 11.194]
    marker_division = [12, 9]
    cam_init_l = np.array([[f_init, 0, img_size[0] / 2], [0, f_init, img_size[1] / 2], [0, 0, 1]])
    cam_init_r = np.array([[f_init, 0, img_size[0] / 2], [0, f_init, img_size[1] / 2], [0, 0, 1]])
    dist_init_l, dist_init_r = np.zeros(5).reshape(-1, 1), np.zeros(5).reshape(-1, 1)
    calib_flags = cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_USE_INTRINSIC_GUESS
    calib_flags += cv2.CALIB_FIX_TANGENT_DIST + cv2.CALIB_FIX_K1 + cv2.CALIB_FIX_K2 + cv2.CALIB_FIX_K3
    # calib_flags += cv2.CALIB_FIX_PRINCIPAL_POINT
    calib_data = calibrate_stereo_camera(board_size, marker_division, img_list_l, img_list_r,
                                         cam_init_l, dist_init_l, cam_init_r, dist_init_r, calib_flags=calib_flags,
                                         vis_scale=0.5, verbose=verbose, img_size=img_size)
    return calib_data



def disp_to_depth(disp, baseline, focal_length):
    np.seterr(divide='ignore')
    depth = np.abs(baseline * focal_length / disp)
    depth[np.isinf(depth)] = 0
    return depth


def rgbd_to_open3d_pcd(rgb, dep, M):
    """ depth in mm """
    rgb = rgb / rgb.max()
    fx, fy, cx, cy = M[0, 0], M[1, 1], M[0, 2], M[1, 2]
    grid_x, grid_y = np.meshgrid(np.arange(dep.shape[1]), np.arange(dep.shape[0]))
    x, y = (grid_x - cx) * dep / fx, (grid_y - cx) * dep / fy
    pcd = np.stack([x, y, dep], axis=-1)
    pcd = np.concatenate([pcd, rgb], axis=-1)
    pcd = pcd.reshape(-1, 6)
    pcd = pcd[np.argwhere(pcd[:, 2] != 0).flatten()]
    # to open3d
    res = o3dg.PointCloud()
    res.points = o3d.utility.Vector3dVector(pcd[:, :3])
    res.colors = o3d.utility.Vector3dVector(pcd[:, 3:])
    return res


def sl_decode_mutliscale_callback(data_dir_0, data_dir_1, f_init, white_thred=5, black_thred=5, proj_res=(1920, 1080),
                                  img_size=(1680, 1120), sl_scales=[1, 2, 4, 8], verbose=False, min_depth=None, max_depth=None):

    # STEP-0: calib with target img_size
    calib_img_list_0 = sorted(glob(os.path.join(data_dir_0, 'calib_*.JPG')))
    calib_img_list_1 = sorted(glob(os.path.join(data_dir_1, 'calib_*.JPG')))

    calib_data = stereo_calib_callback(calib_img_list_0, calib_img_list_1, f_init=f_init, img_size=img_size, verbose=verbose)
    map_x_l, map_y_l, map_x_r, map_y_r, P1, P2, _ = generate_rectify_data(calib_data, size=img_size)

    print(calib_data['M1'], calib_data['M2'])


    ## STEP-1: obtain multi-scale disp maps
    all_sl_disp = []
    for scale in sl_scales:
        curr_proj_res = [proj_res[0] // scale, proj_res[1] // scale]  # resolution of the projector
        # get disp_l from sl
        n_vertical, n_horizontal = int(2 * np.ceil(np.log2(curr_proj_res[0]))), int(2 * np.ceil(np.log2(curr_proj_res[1])))
        names = np.arange(n_vertical).tolist() + np.arange(22, 22 + n_horizontal).tolist() + [44, 45]
        img_l_dir_list = [os.path.join(data_dir_0, 'sl_{}.JPG'.format(i)) for i in names]
        img_r_dir_list = [os.path.join(data_dir_1, 'sl_{}.JPG'.format(i)) for i in names]
        sl_disp_l = get_left_disp_map_from_sl(img_l_dir_list, img_r_dir_list, proj_resolution=curr_proj_res,
                                              map_x_l=map_x_l,
                                              map_y_l=map_y_l, map_x_r=map_x_r, map_y_r=map_y_r,
                                              white_thred=white_thred,
                                              black_thred=black_thred, verbose=False, img_size=img_size)
        all_sl_disp.append(sl_disp_l)

    ## STEP-2: merge the multi-scale disp maps
    merged_sl_disp = all_sl_disp[0].copy()
    for x in all_sl_disp[1:]:
        mask = np.zeros_like(merged_sl_disp)
        mask[merged_sl_disp == 0] = 1
        merged_sl_disp += x * mask

    ## STEP-3: clean the disp map
    clean_disp = disp_morphology_filter(merged_sl_disp)
    clean_disp = pcd_outlier_filter(clean_disp, P1, P2, clean=True)
    clean_disp = ndimage.median_filter(clean_disp, size=20)

    ## STEP-4: generate depth and unwarp
    white_img = cv2.resize(cv2.cvtColor(cv2.imread(img_l_dir_list[-2]), cv2.COLOR_BGR2RGB), img_size)
    depth = disp_to_depth(clean_disp, baseline=np.linalg.norm(calib_data['t']), focal_length=P1[0, 0])
    inv_map_x, inv_map_y = invert_map(map_x_l, map_y_l)
    unwarped_depth = cv2.remap(depth, inv_map_x, inv_map_y, cv2.INTER_NEAREST)
    if max_depth is not None:
        unwarped_depth[unwarped_depth >= max_depth] = 0
    if min_depth is not None:
        unwarped_depth[unwarped_depth <= min_depth] = 0

    ## return data
    pcd = rgbd_to_open3d_pcd(rgb=white_img, dep=unwarped_depth, M=calib_data['M1'])
    return unwarped_depth, pcd, white_img





##--------------------------------------------------------------------



if __name__ == '__main__':
    ## setup
    # all_files = ['/media/li/52F3-84AF/DP_data/part2/raw_data/20220317/20220317103457']
    all_files = ['/Users/feiran-l/Desktop/dataset/DP_data/part3/raw_data/20220321/20220321094850']
    f_init = 85 / (0.00536 * 4)  # focal length in pixel
    white_thred, black_thred = 2, 2
    sl_scales = [1, 2, 4, 8]
    min_dep, max_dep = 200, 15000
    ## start loop
    for data_dir in tqdm(all_files):
        data_dir_0 = os.path.join(data_dir, 'cam1')
        data_dir_1 = os.path.join(data_dir, 'cam0')
        unwarped_depth, pcd, white_img = sl_decode_mutliscale_callback(data_dir_0, data_dir_1, f_init=f_init,
                                                                       white_thred=white_thred, black_thred=black_thred,
                                                                       sl_scales=sl_scales, verbose=True,
                                                                       min_depth=min_dep, max_depth=max_dep)

        unwarped_depth = np.uint16(unwarped_depth)
        cv2.imwrite('/Users/feiran-l/Desktop/dep.png', unwarped_depth)

        ## plot results
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        for i, (x, title) in enumerate(zip([white_img, unwarped_depth], ['white_img', 'unwarped_depth'])):
            ax[i].imshow(x)
            ax[i].set_title(title)
        plt.tight_layout()
        plt.show()

        o3d.io.write_point_cloud('merged.ply', pcd)


