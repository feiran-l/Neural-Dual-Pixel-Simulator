import glob
import os
import cv2
import skimage.transform
from cv2 import aruco
import numpy as np
from matplotlib import pyplot as plt
import h5py
import yaml


def save_h5py_file(name, my_dict):
    h = h5py.File(name, 'w')
    for k, v in my_dict.items():
        h.create_dataset(k, data=np.array([v]).squeeze())
    h.close()


def load_h5py_file(data_dir):
    res = {}
    file = h5py.File(data_dir, 'r')
    for k, v in file.items():
        res[k] = np.array(v)
    file.close()
    return res


class CharucoMarker(object):
    def __init__(self, board_size=[15, 11.194], marker_division=[12, 9], calib_flags=None):
        """
        :param board_size: squareLength and markerLength. e,g.[5, 1] means 5mm and 1mm
        :param marker_division: The number of marker in vertical and horizontal direction. e.g. [5, 7]
        """
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_1000)
        self.board = aruco.CharucoBoard_create(marker_division[0], marker_division[1], board_size[0], board_size[1],
                                               self.aruco_dict)
        # variables for single camera calibration
        self.all_ids_single = []
        self.all_corners_2d_single = []
        self.all_corners_3d_single = []
        self.valid_img_index_single = []
        # variables for stereo camera calibration
        self.all_ids_stereo = []
        self.all_corners_2d_l, self.all_corners_2d_r = [], []
        self.all_corners_3d_stereo = []
        self.valid_img_index_stereo = []
        # calibration terms
        self.criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10000, 1e-9)
        self.flags = calib_flags

    def detect_corners_one_img(self, img):
        """
        detect charuco markers from one img
        :param img: image [H, W, c] read from opencv
        :return: 2D corners and their ids
        """
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners, ids, _ = aruco.detectMarkers(img_gray, self.aruco_dict)
        if len(corners) > 0:
            for corner in corners:
                cv2.cornerSubPix(img_gray, corner, (3, 3), (-1, -1), criteria=criteria)
            _, corners2, ids2 = aruco.interpolateCornersCharuco(corners, ids, img_gray, self.board)
            if corners2 is not None and ids2 is not None and len(corners2) > 3:
                return corners2, ids2
        return None, None

    def find_2D_3D_matches(self, corners, ids):
        """
        :param corners: detected 2d corner points
        :param ids: each id for 2d point
        :return: 3D corner, 2D corner
        """
        obj_point3d, img_point2d = [], []
        for i in range(len(ids)):
            point3d = self.board.chessboardCorners[ids[i][0]]  # shape [3]
            obj_point3d.append(point3d)
            img_point2d.append(corners[i])
        return np.array(obj_point3d), np.array(img_point2d)

    def verify_single_calib(self, cam_params, img_dir_list, img_size=None, vis_scale=0.5):
        error_list = []
        tol_error = 0
        intrinsic, dist, rvecs, tvecs = cam_params['M'], cam_params['d'], cam_params['R'], cam_params['t']

        # loop through all valid frames
        for i in range(len(self.valid_img_index_single)):
            points3d, points2d = self.all_corners_3d_single[i], self.all_corners_2d_single[i]
            # calculate error
            points2d_project, _ = cv2.projectPoints(points3d, rvecs[i], tvecs[i], intrinsic, distCoeffs=dist)
            error = cv2.norm(points2d, points2d_project, cv2.NORM_L2) / len(points3d)
            error_list.append(error)
            tol_error = tol_error + error
            # plot
            img = cv2.imread(img_dir_list[self.valid_img_index_single[i]])
            if img_size is not None:
                img = cv2.resize(img, img_size)
            print('re-projection error of the {}-th image: {:.4f}'.format(i, error))
            for j in range(len(points2d_project)):
                cv2.circle(img, (int(points2d_project[j][0, 0]), int(points2d_project[j][0, 1])), 2, (255, 0, 0),
                           thickness=-1)
                cv2.circle(img, (int(points2d[j][0, 0]), int(points2d[j][0, 1])), 3, (0, 255, 0), thickness=2)
                cv2.imshow('re-projection output',
                           cv2.resize(img, (int(img.shape[1] * vis_scale), int(img.shape[0] * vis_scale))))
                cv2.waitKey(2)

        # results
        print('average re-projection error is: ', tol_error / len(self.valid_img_index_single))
        return np.array(error_list)

    def get_params_for_undistorted_img(self, camerapara, w, h):
        intrinsic, dist, rvecs, tvecs = camerapara
        new_cam_mat, roi = cv2.getOptimalNewCameraMatrix(intrinsic, dist, (w, h), 1, (w, h))
        mapx, mapy = cv2.initUndistortRectifyMap(intrinsic, dist, None, new_cam_mat, (w, h), 5)
        new_dist = np.zeros_like(dist)
        # how to use after: then new_cam_mat is the camera intrinsic for the image dst without any distortion
        # dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
        return new_cam_mat, new_dist, roi, mapx, mapy

    def calib_single_camera(self, img_dir_list, cam_mat_init=None, dist_init=None, img_size=None, verbose=False):
        """
        :param data_dir: captured data path
        :param f: focal length in mm of the camera
        :param pixel_size: pixel size in mm/pixel
        :param size: downsize the image to size to speed up
        :return: camera parameters
        """
        # downsize the resolution for speed
        if img_size is None:
            img_size = cv2.imread(img_dir_list[0], 0).shape
            img_size = (img_size[1], img_size[0])

        # loop all the images
        for i in range(len(img_dir_list)):
            img = cv2.resize(cv2.imread(img_dir_list[i]), img_size)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners, ids = self.detect_corners_one_img(img_gray)
            if corners is not None:
                obj_3d_pts, img_2d_pts = self.find_2D_3D_matches(corners, ids)
                self.all_ids_single.append(ids)
                self.all_corners_2d_single.append(img_2d_pts)
                self.all_corners_3d_single.append(obj_3d_pts)
                self.valid_img_index_single.append(i)

        ## calibrating
        res, cam_mat, dist, rvecs, tvecs = cv2.calibrateCamera(self.all_corners_3d_single, self.all_corners_2d_single,
                                                               img_size, cam_mat_init, dist_init, flags=self.flags,
                                                               criteria=self.criteria)
        if verbose:
            print('single camera calibration done, reprojection error is {:.5f}'.format(res))
        calib_data = {'residual': res, 'M': cam_mat, 'd': dist, 'R': rvecs, 't': tvecs}
        return calib_data


    def malloc_single_calib(self):
        self.all_ids_single = []
        self.all_corners_2d_single = []
        self.all_corners_3d_single = []
        self.valid_img_index_single = []


    def generate_map_for_rectification(self, calib_data, size):
        """
        Usage: rectified_l = cv2.remap(img_l, map_x_l, map_y_l, interpolation=cv2.INTER_LINEAR)
        """
        M1, M2, d1, d2 = np.array(calib_data['M1']), np.array(calib_data['M2']), np.array(calib_data['d1']), np.array(
            calib_data['d2'])
        R, t = np.array(calib_data['R']), np.array(calib_data['t'])
        R1, R2, P1, P2, Q = cv2.stereoRectify(cameraMatrix1=M1, cameraMatrix2=M2, distCoeffs1=d1, distCoeffs2=d2, R=R,
                                              T=t,
                                              flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1, imageSize=size,
                                              newImageSize=size)[0:5]
        map_x_l, map_y_l = cv2.initUndistortRectifyMap(M1, d1, R1, P1, size, cv2.CV_32FC1)
        map_x_r, map_y_r = cv2.initUndistortRectifyMap(M2, d2, R2, P2, size, cv2.CV_32FC1)
        rectify_dict = {'map_x_l': map_x_l, 'map_y_l': map_y_l, 'map_x_r': map_x_r, 'map_y_r': map_y_r, 'Q': Q}
        return rectify_dict

    def draw_matches(self, img1, img2):
        akaze = cv2.AKAZE_create()
        kp1, des1 = akaze.detectAndCompute(img1, None)
        kp2, des2 = akaze.detectAndCompute(img2, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None, flags=2)
        return img3

    def verify_stereo_calib(self, calib_data, img_dir_list_l, img_dir_list_r, vis_scale=0.5, img_size=None):
        rectify_dict = self.generate_map_for_rectification(calib_data, img_size)
        map_x_l, map_y_l, map_x_r, map_y_r = rectify_dict['map_x_l'], rectify_dict['map_y_l'], rectify_dict['map_x_r'], \
                                             rectify_dict['map_y_r']
        for i, (name_l, name_r) in enumerate(zip(img_dir_list_l, img_dir_list_r)):
            gray_l, gray_r = cv2.imread(name_l, 0), cv2.imread(name_r, 0)
            if img_size is not None:
                gray_l, gray_r = cv2.resize(gray_l, img_size), cv2.resize(gray_r, img_size)
            l_rect = cv2.remap(gray_l, map_x_l, map_y_l, interpolation=cv2.INTER_LINEAR)
            r_rect = cv2.remap(gray_r, map_x_r, map_y_r, interpolation=cv2.INTER_LINEAR)
            res = self.draw_matches(l_rect, r_rect)
            cv2.namedWindow('rectified pairs', cv2.WINDOW_NORMAL)
            cv2.moveWindow('rectified pairs', 0, 0)
            # cv2.setWindowProperty('rectified pairs', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow('rectified pairs',
                       cv2.resize(res, (int(res.shape[1] * vis_scale), int(res.shape[0] * vis_scale))))
            cv2.waitKey(200)
        cv2.destroyAllWindows()

    def calib_stereo_camera(self, img_dir_list_l, img_dir_list_r, cam_mat_init_l, dist_init_l, cam_mat_init_r,
                            dist_init_r, img_size=None, verbose=False):
        # respective intrinsic calibration to initialize the stereo calibration
        calib_data_l = self.calib_single_camera(img_dir_list_l, cam_mat_init_l, dist_init_l, img_size=img_size,
                                                verbose=verbose)
        self.malloc_single_calib()
        calib_data_r = self.calib_single_camera(img_dir_list_r, cam_mat_init_r, dist_init_r, img_size=img_size,
                                                verbose=verbose)
        self.malloc_single_calib()
        cam_l, dist_l = calib_data_l['M'], calib_data_l['d']
        cam_r, dist_r = calib_data_r['M'], calib_data_r['d']

        # detect corners
        for i, (name_l, name_r) in enumerate(zip(img_dir_list_l, img_dir_list_r)):
            gray_l, gray_r = cv2.imread(name_l, 0), cv2.imread(name_r, 0)
            if img_size is not None:
                gray_l, gray_r = cv2.resize(gray_l, img_size), cv2.resize(gray_r, img_size)
            corners_l, ids_l = self.detect_corners_one_img(gray_l)
            corners_r, ids_r = self.detect_corners_one_img(gray_r)
            if corners_l is not None and corners_r is not None:
                # only use the corners detected by both imgs
                shared_ids = np.intersect1d(ids_l.flatten(), ids_r.flatten()).reshape(-1, 1)
                corners_l = [corners_l[i] for i in np.where((ids_l == shared_ids[:, None]).all(-1))[1]]
                corners_r = [corners_r[i] for i in np.where((ids_r == shared_ids[:, None]).all(-1))[1]]
                obj_pts_l, img_pts_l = self.find_2D_3D_matches(corners_l, shared_ids)
                obj_pts_r, img_pts_r = self.find_2D_3D_matches(corners_r, shared_ids)
                assert np.linalg.norm(
                    obj_pts_l - obj_pts_r) <= 1e-5, 'Stereo calibration fault: object pts are not identical in left and right images!'
                # append results
                self.all_corners_3d_stereo.append(obj_pts_l)
                self.all_corners_2d_l.append(img_pts_l)
                self.all_corners_2d_r.append(img_pts_r)
                self.all_ids_stereo.append(shared_ids)
                self.valid_img_index_stereo.append(i)

        # stereo calibration
        res, M1, d1, M2, d2, R, t, E, F = cv2.stereoCalibrate(self.all_corners_3d_stereo, self.all_corners_2d_l,
                                                              self.all_corners_2d_r, cam_l, dist_l, cam_r, dist_r,
                                                              img_size, criteria=self.criteria, flags=self.flags)
        if verbose:
            print('stereo calibration done, reprojection error is {:.5f}'.format(res))
        calib_data = {'residual': res, 'M1': M1, 'M2': M2, 'd1': d1, 'd2': d2, 'R': R, 't': t, 'E': E, 'F': F}
        return calib_data


##-----------------------------------------------------------------------------------------


def calibrate_single_camera(board_size, marker_division, img_dir_list, cam_mat_init=None, dist_init=None,
                            calib_flags=None,
                            verbose=True, vis_scale=0.5, img_size=None):
    """
    :param board_size     : charuco_board size [square_length, marker_length]
    :param marker_division: [n_marker_x, n_marker_y]
    :param img_dir_list   : data directory
    :param cam_mat_init   : initial camera matrix
    :param dist_init      : initial distortions of shape [5, 1]
    :param calib_flags    : flags used in calibration
    :param verbose        : verbose or not during visualizing the calibration results
    :param vis_scale      : plot scale in cv2.imshow to suit the screen
    :param img_size    : whether to resize to image for calibration, None to keep the original size
    :return               : camera parameters in dict
    """
    calibrator = CharucoMarker(board_size, marker_division, calib_flags=calib_flags)
    calib_data = calibrator.calib_single_camera(img_dir_list, cam_mat_init, dist_init, img_size=img_size)
    if verbose:
        calibrator.verify_single_calib(calib_data, img_dir_list, img_size=img_size, vis_scale=vis_scale)
    return calib_data


def calibrate_stereo_camera(board_size, marker_division, img_dir_list_l, img_dir_list_r, cam_mat_init_l=None,
                            dist_init_l=None, cam_mat_init_r=None, dist_init_r=None, calib_flags=None,
                            vis_scale=0.5, verbose=False, img_size=None):
    """
    :param board_size     : charuco_board size [square_length, marker_length]
    :param marker_division: [n_marker_x, n_marker_y]
    :param img_dir_list_l : left-view data directory
    :param img_dir_list_r : right-view data directory
    :param cam_mat_init_l : left initial camera matrix
    :param dist_init_l    : left initial distortions of shape [5, 1]
    :param cam_mat_init_r : right initial camera matrix
    :param dist_init_r    : right initial distortions of shape [5, 1]
    :param calib_flags    : flags used in calibration
    :param vis_scale      : plot scale in cv2.imshow to suit the screen
    :param verbose        : plot or not
    :return               : calib parameters in dict
    """
    calibrator = CharucoMarker(board_size, marker_division, calib_flags=calib_flags)
    calib_data = calibrator.calib_stereo_camera(img_dir_list_l, img_dir_list_r, cam_mat_init_l, dist_init_l,
                                                cam_mat_init_r, dist_init_r, img_size=img_size, verbose=verbose)
    if verbose:
        calibrator.verify_stereo_calib(calib_data, img_dir_list_l, img_dir_list_r, vis_scale, img_size=img_size)
    return calib_data


##--------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    ## setups
    board_size = [15, 11.194]
    marker_division = [12, 9]

    img_size = (1680, 1120)
    f_init = 85 / (0.00536 * 4)  # focal length in pixel
    cam_init_l = np.array([[f_init, 0, img_size[0] / 2], [0, f_init, img_size[1] / 2], [0, 0, 1]])
    cam_init_r = np.array([[f_init, 0, img_size[0] / 2], [0, f_init, img_size[1] / 2], [0, 0, 1]])
    dist_init_l, dist_init_r = np.zeros(5).reshape(-1, 1), np.zeros(5).reshape(-1, 1)

    calib_flags = cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_USE_INTRINSIC_GUESS
    calib_flags += cv2.CALIB_FIX_TANGENT_DIST + cv2.CALIB_FIX_K1 + cv2.CALIB_FIX_K2 + cv2.CALIB_FIX_K3

    ## single camera calibration
    # img_dir_list = sorted(glob.glob('/Users/li/Desktop/DP_data/20220202/20220202212858/cam0/calib_*.JPG'))
    # res = calibrate_single_camera(board_size, marker_division, img_dir_list, cam_mat_init=cam_init_l,
    #                               dist_init=dist_init_l, calib_flags=calib_flags, verbose=True, img_size=img_size)

    ## stereo camera calibration
    img_dir_list_l = sorted(glob.glob('/Users/feiran-l/Desktop/dataset/DP_data/part2/raw_data/20220317/20220317122932/cam0/calib_*.JPG'))
    img_dir_list_r = sorted(glob.glob('/Users/feiran-l/Desktop/dataset/DP_data/part2/raw_data/20220317/20220317122932/cam1/calib_*.JPG'))
    res = calibrate_stereo_camera(board_size, marker_division, img_dir_list_l, img_dir_list_r, cam_init_l, dist_init_l,
                                  cam_init_r, dist_init_r, calib_flags=calib_flags, vis_scale=0.5, verbose=True,
                                  img_size=img_size)
    print(res['M1'])

