import h5py
import numpy as np
from glob import glob
from tqdm import tqdm
from matplotlib import pyplot as plt
import cv2
import os
import pandas as pd
import open3d as o3d
from open3d import geometry as o3dg
from pathlib import Path
import subprocess



def download_and_unzip(url, download_dir, decompress_dir):
    download_file = os.path.join(download_dir, os.path.basename(url))
    if not os.path.exists(download_file) and not os.path.exists(download_file[:-4]):
        ## download
        subprocess.run(['curl', url, '--output', download_file])
        ## unzip
        subprocess.run(['unzip', download_file, '-d', decompress_dir], stdout = subprocess.DEVNULL)
        ## remove zip file
        subprocess.run(['rm', download_file])
    elif os.path.exists(download_file) and not os.path.exists(download_file[:-4]):
        ## unzip
        subprocess.run(['unzip', download_file, '-d', decompress_dir], stdout=subprocess.DEVNULL)
        ## remove zip file
        subprocess.run(['rm', download_file])



def partite_all_urls(all_urls_to_download):
    np.random.seed(0)
    shuffled_all_urls = np.random.choice(all_urls_to_download, len(all_urls_to_download), replace=False)
    size = 50
    count = 0
    for i in range(0, len(shuffled_all_urls), size):
        curr_urls = shuffled_all_urls[i:i+size]
        np.savetxt('./url_partition/partition_{}.txt'.format(count), curr_urls, delimiter=" ", fmt="%s")
        count += 1


def load_h5py_file(data_dir):
    res = {}
    f = h5py.File(data_dir, 'r')
    for k, v in f.items():
        res[k] = np.array(v)
    f.close()
    return res


def save_h5py_file(name, my_dict):
    h = h5py.File(name, 'w')
    for k, v in my_dict.items():
        h.create_dataset(k, data=np.array([v]).squeeze())
    h.close()



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




def parse_a_certain_scene(scene_name, data_dir, save_dir, w=1024, h=768):
    def extract_cam_intrinsics(_df, _w=1024, _h=768):
        """ the csv file provids opengl projection matrix, this function extracts camera intrinsics from it """
        _M_proj = np.array([[_df["M_proj_00"], _df["M_proj_01"], _df["M_proj_02"], _df["M_proj_03"]],
                            [_df["M_proj_10"], _df["M_proj_11"], _df["M_proj_12"], _df["M_proj_13"]],
                            [_df["M_proj_20"], _df["M_proj_21"], _df["M_proj_22"], _df["M_proj_23"]],
                            [_df["M_proj_30"], _df["M_proj_31"], _df["M_proj_32"], _df["M_proj_33"]]])
        _fx, _fy = _M_proj[0, 0] * w / 2, _M_proj[1, 1] * h / 2
        _cx, _cy = (_w - _M_proj[0, 2] * _w) / 2, (_h - _M_proj[1, 2] * _h) / 2
        _M = np.identity(3)
        _M[0, 0], _M[1, 1], _M[0, 2], _M[1, 2] = _fx, _fy, _cx, _cy
        return _M


    Path(os.path.join(save_dir, scene_name)).mkdir(exist_ok=True, parents=True)

    ## load camera intrinsics
    df_camera_parameters = pd.read_csv('metadata_camera_parameters.csv', index_col='scene_name')
    df = df_camera_parameters.loc[scene_name]
    cam_mat = extract_cam_intrinsics(df, _w=1024, _h=768)

    ## identify valid camera ids
    cam_ids = [x.split('_')[2] for x in os.listdir(os.path.join(download_dir, scene_name, 'images'))]
    cam_ids = list(set(cam_ids))

    for cam in cam_ids:
        all_ids = sorted(list(set([x.split('.')[1] for x in os.listdir(os.path.join(data_dir, scene_name, 'images/scene_cam_{}_final_hdf5'.format(cam)))])))
        for curr_id in tqdm(all_ids):
            curr_name = '{}.cam_{}.{}'.format(scene_name, cam, curr_id)

            ## step 1: load images
            rgb = cv2.imread(os.path.join(data_dir, scene_name, 'images/scene_cam_{}_final_preview/frame.{}.tonemap.jpg'.format(cam, curr_id)), -1)
            dis = load_h5py_file(os.path.join(data_dir, scene_name, 'images/scene_cam_{}_geometry_hdf5/frame.{}.depth_meters.hdf5'.format(cam, curr_id)))['dataset']
            dis = np.float32(dis) * 1000  # m -> mm
            dis[np.isnan(dis)] = 0
            normal = load_h5py_file(os.path.join(data_dir, scene_name, 'images/scene_cam_{}_geometry_hdf5/frame.{}.normal_cam.hdf5'.format(cam, curr_id)))['dataset']
            normal = np.float32(normal)
            normal = normal / np.linalg.norm(normal, 2, 2, keepdims=True)
            normal[np.isnan(normal)] = 0

            ## step 2: convert distance to optical center to real dep
            focal_len = (cam_mat[0, 0] + cam_mat[1, 1]) / 2
            uu, vv = np.meshgrid(np.linspace((-0.5 * w) + 0.5, (0.5 * w) - 0.5, w), np.linspace((-0.5 * h) + 0.5, (0.5 * h) - 0.5, h))
            denominator = np.linalg.norm(np.stack([uu, vv, np.ones_like(dis) * focal_len], axis=-1), 2, 2)
            dep = dis / denominator * focal_len

            ## step 3: save image if depth is complete
            if not np.any(dep == 0):
                my_dict = {'rgb': rgb, 'dep': dep, 'normal': normal, 'M': cam_mat}
                save_h5py_file(os.path.join(save_dir, scene_name, '{}.h5'.format(curr_name)), my_dict)




##------------------------------------------------------------------------------------------------------




if __name__ == '__main__':

    partiton_id = 8
    download_dir = '/Users/feiran-l/Downloads/hypersim8'
    save_dir = '/Users/feiran-l/Desktop/tmp/processed_hypersim/partition_{}'.format(partiton_id)

    Path(download_dir).mkdir(exist_ok=True, parents=True)
    Path(save_dir).mkdir(exist_ok=True, parents=True)

    urls = np.loadtxt('./url_partition/partition_{}.txt'.format(partiton_id), dtype=str)[34:]

    for curr_url in urls:
        scene_name = curr_url.split('/')[-1][:-4]
        print('----------- start processing {}'.format(scene_name))

        download_and_unzip(curr_url, download_dir, download_dir)
        parse_a_certain_scene(scene_name, download_dir, save_dir)
        subprocess.run(['rm', '-r', os.path.join(download_dir, scene_name)])




