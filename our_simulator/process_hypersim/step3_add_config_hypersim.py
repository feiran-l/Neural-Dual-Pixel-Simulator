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
from matplotlib import patches



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



def adaptive_pixel_size(init_pixel_size, dep, focus_dis, f_number, pinhole_focal_len_in_pixel):
    for s in np.arange(1, 100):
        curr_pixel_size = init_pixel_size * s
        curr_pinhole_focal_len_in_mm = curr_pixel_size * pinhole_focal_len_in_pixel
        curr_thin_lens_focal_len_in_mm = 1.0 / (1.0 / focus_dis + 1.0 / curr_pinhole_focal_len_in_mm)
        curr_coc = generate_coc_map(dep, focus_dis, f_number, curr_pixel_size, curr_thin_lens_focal_len_in_mm)
        if curr_coc.min() < -20 or 20 < curr_coc.max():
            return curr_pixel_size, curr_thin_lens_focal_len_in_mm, s
        else:
            continue
    return init_pixel_size,  1.0 / (1.0 / focus_dis + 1.0 / (init_pixel_size * pinhole_focal_len_in_pixel)), -1



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

    all_F_numbers = [1.8, 2, 2.8, 4, 5.6]
    af_win_size = (60, 40)

    for partition in ['partition_8']:
        all_file = sorted(glob('/Users/feiran-l/Desktop/tmp/selected_hypersim/{}/*.h5'.format(partition)))
        print('{} data to proceed'.format(len(all_file)))

        for i, curr_dir in enumerate(tqdm(all_file)):
            data = load_h5py_file(curr_dir)

            ## generate necessary info
            init_pixel_size = 36 / 1024
            data['F_number'] = random.choice(all_F_numbers)
            focus_dis, af_pt = generate_focus_distance(data['dep'], win_size=af_win_size)
            data['focus_dis'] = focus_dis
            data['af_pt'] = af_pt
            ## scale pixel size to match coc
            pinhole_focal_len_in_pixel = (data['M'][0, 0] + data['M'][1, 1]) / 2
            pixel_size, thin_lens_focal_len_in_mm, _ = adaptive_pixel_size(init_pixel_size, data['dep'], data['focus_dis'], 1.8, pinhole_focal_len_in_pixel)
            data['pixel_size'] = pixel_size
            data['thin_lens_focal_len_in_mm'] = thin_lens_focal_len_in_mm

            ## plot and save
            init_thin_lens_focal_len_in_mm = 1.0 / (1.0 / focus_dis + 1.0 / (init_pixel_size * pinhole_focal_len_in_pixel))
            coc_with_real_f_number = generate_coc_map(depth=data['dep'], focus_dis=data['focus_dis'], f_number=data['F_number'], pixel_size=pixel_size, focal_len=thin_lens_focal_len_in_mm)
            coc_with_18_f_number = generate_coc_map(depth=data['dep'], focus_dis=data['focus_dis'], f_number=1.8, pixel_size=pixel_size, focal_len=thin_lens_focal_len_in_mm)

            # fig, ax = plt.subplots(1, 4, figsize=(15, 5))
            # ax[0].imshow(data['rgb'])
            # ax[1].imshow(data['dep'])
            # ax[2].imshow(coc_with_18_f_number)
            # ax[3].imshow(coc_with_real_f_number)
            # ax[0].add_patch(patches.Rectangle(xy=data['af_pt'], width=0.25, height=0.5, ec='#000000', fill=False))
            # ax[1].add_patch(patches.Rectangle(xy=data['af_pt'], width=0.25, height=0.5, ec='#000000', fill=False))
            # ax[2].add_patch(patches.Rectangle(xy=data['af_pt'], width=0.25, height=0.5, ec='#000000', fill=False))
            # ax[3].add_patch(patches.Rectangle(xy=data['af_pt'], width=0.25, height=0.5, ec='#000000', fill=False))
            # ax[0].set_title('thin_lens_focal_len_in_mm: {:.2f}'.format(thin_lens_focal_len_in_mm))
            # ax[1].set_title('f_number: {:.2f}'.format(data['F_number']))
            # ax[2].set_title('coc with 1.8 f-number {:.2f},  {:.2f}'.format(coc_with_18_f_number.min(), coc_with_18_f_number.max()))
            # ax[3].set_title('coc with real f-number {:.2f},  {:.2f}'.format(coc_with_real_f_number.min(), coc_with_real_f_number.max()))
            # # plt.show(block=False)
            # # plt.pause(0.1)
            # # plt.close()
            # plt.show()


            save_h5py_file(curr_dir, data)

