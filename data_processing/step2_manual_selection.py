import os
import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt
from charuco_calib import load_h5py_file


def on_press(event):
    sys.stdout.flush()
    if event.key == 'y':
        f = open('{}_log/manual_select.txt'.format(part), 'a')
        f.writelines(curr_name + '\n')
        f.close()
        plt.close()
    elif event.key == 'n':
        f = open('{}_log/manual_deny.txt'.format(part), 'a')
        f.writelines(curr_name + '\n')
        f.close()
        plt.close()
    else:
        print('------------- `Y` to accept, `N` to deny --------------')





def generate_coc_map(depth, focus_dis, focal_len, f_number, pixel_size):
    """ generate the signed CoC map in pixel unit. depth and focus_dis in mm, focal_len in pixel """
    coc = ((depth - focus_dis) / depth) * (focal_len ** 2 / (f_number * (focus_dis / pixel_size - focal_len)))
    coc[depth == 0] = 0
    return coc




##------------------------------------------------------------------------------



if __name__ == '__main__':

    part = 'part3'

    all_name = np.loadtxt('./{}_log/success_case.txt'.format(part), dtype='str')

    open('{}_log/manual_select.txt'.format(part), 'w').close()
    open('{}_log/manual_deny.txt'.format(part), 'w').close()

    for i, curr_name in enumerate(all_name):
        curr_dir = os.path.join('/Users/feiran-l/Desktop/dataset/DP_data/{}/valid_data/{}'.format(part, curr_name))

        ## load data
        rgb0 = cv2.imread(os.path.join(curr_dir, 'cam0/FN_22.png'), 1)
        rgb1 = cv2.imread(os.path.join(curr_dir, 'cam1/FN_22.png'), 1)
        dep0 = cv2.imread(os.path.join(curr_dir, 'cam0/dep.png'), -1)
        dep1 = cv2.imread(os.path.join(curr_dir, 'cam1/dep.png'), -1)
        if rgb0 is None or rgb1 is None or dep0 is None or dep1 is None:
            print('{} does not work'.format(curr_name))
            f = open('{}_log/manual_deny.txt'.format(part), 'a')
            f.writelines(curr_name + '\n')
            f.close()
            continue
        rgb0, rgb1 = cv2.cvtColor(rgb0, cv2.COLOR_BGR2RGB), cv2.cvtColor(rgb1, cv2.COLOR_BGR2RGB)
        meta_data = load_h5py_file(os.path.join(curr_dir, 'meta_data.h5'))

        ## plot
        fig, ax = plt.subplots(1, 4, figsize=(12, 3))
        fig.canvas.mpl_connect('key_press_event', on_press)
        for j, x in enumerate([rgb0, dep0, rgb1, dep1]):
            ax[j].imshow(x)
        ax[0].set_title('{}, {}'.format(i, curr_name))
        ax[1].set_title('{:.2f}mm'.format(meta_data['focus_dis_0']))
        ax[3].set_title('{:.2f}mm'.format(meta_data['focus_dis_1']))
        plt.tight_layout()
        plt.show()

        # plt.show(block=False)
        # plt.pause(0.2)
        # plt.close()
