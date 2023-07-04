import random
import numpy as np
from matplotlib import pyplot as plt
import cv2
import open3d as o3d
from open3d import geometry as o3dg
import os
from tqdm import tqdm
from glob import glob
from step1_parse_hypersim import load_h5py_file, save_h5py_file
from pathlib import Path
import sys
import matplotlib.patches as patches


def generate_focus_distance(dep, win_size=(60, 40)):
    AF_pt_h, AF_pt_w = np.random.randint(100, dep.shape[0]-100, 1)[0], np.random.randint(100, dep.shape[1]-100, 1)[0]
    AF_area = dep[AF_pt_h-win_size[0]//2:AF_pt_h+win_size[0]//2, AF_pt_w-win_size[1]//2:AF_pt_w+win_size[1]//2]
    return np.mean(AF_area), np.array([AF_pt_w, AF_pt_h])



def generate_coc_map(depth, focus_dis, f_number, pixel_size, focal_len):
    """ generate the signed CoC map in pixel unit. depth and focus_dis in mm, focal_len in pixel """
    with np.errstate(divide='ignore'):
        coc = ((depth - focus_dis) / depth) * (focal_len ** 2 / (f_number * (focus_dis / pixel_size - focal_len)))
    coc[depth == 0] = 0
    return coc



def adapative_pixel_size(init_pixel_size, dep, focus_dis, f_number, focal_len):
    for s in np.arange(1, 100):
        curr_pixel_size = init_pixel_size * s
        curr_coc = generate_coc_map(dep, focus_dis, f_number, curr_pixel_size, focal_len)
        if curr_coc.min() < -20 or 20 < curr_coc.max():
            return curr_pixel_size, s
        else:
            continue
    return init_pixel_size, -1



def draw_AF_point(img, point, win_size=(60, 40)):
    point = np.array(point)
    win_size = np.array(win_size)
    start_point = point - win_size // 2
    end_point = point + win_size // 2
    color = (255, 0, 0)
    thickness = 5
    image = cv2.rectangle(img, start_point, end_point, color, thickness)
    return image





##--------------------------------------------------------------------------------------------------



if __name__ == '__main__':

    np.random.seed(0)
    random.seed(0)

    partition = 'partition_8'
    all_file = sorted(glob('/Users/feiran-l/Desktop/tmp/selected_hypersim/{}/*.h5'.format(partition)))
    print('{} data to proceed'.format(len(all_file)))


    for i, curr_dir in enumerate(tqdm(all_file)):
        data = load_h5py_file(curr_dir)
        print(data.keys())
        rgb, dep = data['rgb'] / 255.0, data['dep']
        focal_len = data['thin_lens_focal_len_in_mm']
        coc = generate_coc_map(dep, data['focus_dis'], data['F_number'], data['pixel_size'], focal_len)

        fig, ax = plt.subplots(1, 3, figsize=(12, 5))
        ax[0].imshow(data['rgb'])
        ax[1].imshow(data['dep'])
        ax[2].imshow(coc)
        for j in range(len(ax)):
            ax[j].add_patch(patches.Rectangle(xy=data['af_pt'], width=0.25, height=0.5, ec='#000000', fill=False))
        ax[0].set_title('thin-lens focal len: {}'.format(data['thin_lens_focal_len_in_mm']))
        ax[1].set_title('f-number: {}'.format(data['F_number']))
        ax[2].set_title('{:.2f},  {:.2f}'.format(coc.min(), coc.max()))
        # plt.show(block=False)
        # plt.pause(0.1)
        # plt.close()

        plt.show()



