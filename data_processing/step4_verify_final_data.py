import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from data_processing.charuco_calib import load_h5py_file
from glob import glob
from tqdm import tqdm


def draw_AF_point(img, point, win_size=(60, 40)):
    point = np.array(point)
    win_size = np.array(win_size)
    start_point = point - win_size // 2
    end_point = point + win_size // 2
    color = (255, 0, 0)
    thickness = 5
    image = cv2.rectangle(img, start_point, end_point, color, thickness)
    return image


##---------------------------------------------------------------------


if __name__ == '__main__':


    all_names = glob('/Users/feiran-l/Desktop/dataset/DP_data/final_data/train/*')

    print('totally {} data'.format(len(all_names)))


    for i, curr_dir in enumerate(all_names):
        curr_name = curr_dir.split('/')[-1]
        dep_0 = cv2.imread(os.path.join(curr_dir, 'cam0/dep.png'), -1)
        dep_1 = cv2.imread(os.path.join(curr_dir, 'cam1/dep.png'), -1)
        rgb_0 = cv2.imread(os.path.join(curr_dir, 'cam0/FN_1.4_c.png'), 1)
        rgb_1 = cv2.imread(os.path.join(curr_dir, 'cam1/FN_1.4_c.png'), 1)
        rgb_0, rgb_1 = cv2.cvtColor(rgb_0, cv2.COLOR_BGR2RGB), cv2.cvtColor(rgb_1, cv2.COLOR_BGR2RGB)
        meta_data = load_h5py_file(os.path.join(curr_dir, 'meta_data.h5'))

        ## crop to roi
        # rgb_0, dep_0 = rgb_0[50:1050, 100:1100], dep_0[50:1050, 100:1100]
        # rgb_1, dep_1 = rgb_1[50:1050, 500:1500], dep_1[50:1050, 500:1500]

        ## plot
        # af_pt_0, af_pt_1 = meta_data['AF_point_0'], meta_data['AF_point_1']
        # dep_0, dep_1 = draw_AF_point(dep_0, af_pt_0), draw_AF_point(dep_1, af_pt_1)
        # rgb_0, rgb_1 = draw_AF_point(rgb_0, af_pt_0), draw_AF_point(rgb_1, af_pt_1)
        fig, ax = plt.subplots(2, 2, figsize=(10, 7))
        ax[0, 0].imshow(rgb_0)
        ax[0, 1].imshow(dep_0, cmap='turbo')
        ax[1, 0].imshow(rgb_1)
        ax[1, 1].imshow(dep_1, cmap='turbo')
        ax[0, 0].set_title('{}: name: {}'.format(i, curr_name))
        # ax[0, 1].set_title('focus dis: {:.4f}'.format(meta_data['focus_dis_0']))
        # ax[1, 1].set_title('focus dis: {:.4f}'.format(meta_data['focus_dis_1']))
        plt.tight_layout()

        print(curr_name)

        # plt.show(block=False)
        # plt.pause(0.3)
        # plt.close()

        plt.show()


