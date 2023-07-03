import os
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
from pathlib import Path
import shutil


def put_raw_data_in_one_folder(src_dir, tar_blur_dir, tar_sharp_dir):
    """ save all CR2 in base_folder into one target folder, return name lists """
    Path(tar_blur_dir).mkdir(parents=True, exist_ok=True)
    Path(tar_sharp_dir).mkdir(parents=True, exist_ok=True)

    all_files_dir = sorted(glob(os.path.join(src_dir, '*/*')))
    for dir in tqdm(all_files_dir):
        all_img_dir = glob(os.path.join(dir, 'cam0/*.CR2')) + glob(os.path.join(dir, 'cam1/*.CR2'))
        for curr_dir in all_img_dir:
            name_split = curr_dir.split('/')
            if 'FN_22.CR2' in curr_dir:
                new_path = os.path.join(tar_sharp_dir, '{}_{}_{}'.format(name_split[-3], name_split[-2], name_split[-1]))
            else:
                new_path = os.path.join(tar_blur_dir, '{}_{}_{}'.format(name_split[-3], name_split[-2], name_split[-1]))
            shutil.copyfile(curr_dir, new_path)



def combined_extracted_l_r_s(name_list, save_dir, sharp_dir, l_dir, r_dir, c_dir):
    """ after extract DP views and computed depth, merge them together.
        After running this, you need to manually merge the save_dir with the dir where depth is stored. """
    for x in tqdm(name_list):
        print('processing {}'.format(x))
        for cam in ['cam0', 'cam1']:
            Path(os.path.join(save_dir, x, cam)).mkdir(parents=True, exist_ok=True)
            # sharp image
            img_s = cv2.imread(os.path.join(sharp_dir, '{}_{}_FN_22.TIF'.format(x, cam)), -1)
            cv2.imwrite(os.path.join(save_dir, x, cam, 'FN_22.png'), img_s)

            # blurred imgs
            for fn in ['1.8', '2', '2.8', '4', '5.6']:
                # left
                img_l = cv2.imread(os.path.join(l_dir, '{}_{}_FN_{}_l.TIF'.format(x, cam, fn)), -1)
                cv2.imwrite(os.path.join(save_dir, x, cam, 'FN_{}_l.png'.format(fn)), img_l)
                # right
                img_r = cv2.imread(os.path.join(r_dir, '{}_{}_FN_{}_r.TIF'.format(x, cam, fn)), -1)
                cv2.imwrite(os.path.join(save_dir, x, cam, 'FN_{}_r.png'.format(fn)), img_r)
                # combine
                img_c = cv2.imread(os.path.join(c_dir, '{}_{}_FN_{}_c.TIF'.format(x, cam, fn)), -1)
                cv2.imwrite(os.path.join(save_dir, x, cam, 'FN_{}_c.png'.format(fn)), img_c)



#----------------------------------------------------------------------------



if __name__ == '__main__':

    """ prepare for extract DP """
    # base_dir = '/Users/feiran-l/Desktop/dataset/DP_data/multi_focus'
    # src_dir = os.path.join(base_dir, 'raw_data')
    # tar_blur_dir = os.path.join(base_dir, 'data_during_process/all_cr2/blur')
    # tar_sharp_dir = os.path.join(base_dir, 'data_during_process/all_cr2/sharp')
    # put_raw_data_in_one_folder(src_dir, tar_blur_dir, tar_sharp_dir)


    """ save tiff to png """
    base_dir = '/Users/feiran-l/Desktop/dataset/DP_data/multi_focus'
    name_list = [x.split('/')[-1] for x in glob(os.path.join(base_dir, 'raw_data/*'))]
    print(len(name_list))
    save_dir = os.path.join(base_dir, 'extracted_data')
    sharp_dir = os.path.join(base_dir, 'data_during_process/sharp_TIFF')
    l_dir = os.path.join(base_dir, 'data_during_process/dp_l_TIFF')
    r_dir = os.path.join(base_dir, 'data_during_process/dp_r_TIFF')
    c_dir = os.path.join(base_dir, 'data_during_process/dp_c_TIFF')
    combined_extracted_l_r_s(name_list, save_dir, sharp_dir, l_dir, r_dir, c_dir)
















