import os
import shutil
import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt
from charuco_calib import load_h5py_file
from glob import glob
from tqdm import tqdm
from pathlib import Path




def generate_coc_map(depth, focus_dis, focal_len, f_number, pixel_size):
    """ generate the signed CoC map in pixel unit. depth and focus_dis in mm, focal_len in pixel """
    coc = ((depth - focus_dis) / depth) * (focal_len ** 2 / (f_number * (focus_dis / pixel_size - focal_len)))
    coc[depth == 0] = 0
    return coc




##------------------------------------------------------------------------------



if __name__ == '__main__':



    for partition in ['part1', 'part2', 'part3']:

        data_dir = '/media/li/52F3-84AF/DP_data/{}/valid_data'.format(partition)
        final_data_dir = '/media/li/52F3-84AF/DP_data/final_data/{}'.format(partition)
        name_list = np.loadtxt('./{}_log/final_valid_case.txt'.format(partition), dtype=str)
        name_list = list(set(name_list.tolist()))
        Path(final_data_dir).mkdir(parents=True, exist_ok=True)


        count = 0
        for x in tqdm(name_list):
            meta_data = load_h5py_file(os.path.join(data_dir, x, 'meta_data.h5'))
            f0, f1 = meta_data['M0'][0, 0] * 36 / 1680, meta_data['M1'][0, 0] * 36 / 1680
            if f0 >= 85 and f1 >= 85:
                shutil.copytree(os.path.join(data_dir, x), os.path.join(final_data_dir, x))


