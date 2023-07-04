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





def on_press(event):
    sys.stdout.flush()
    if event.key == 'd':
        save_h5py_file(os.path.join(save_dir, curr_name), curr_data)
        plt.close()
    elif event.key == 'n':
        plt.close()
    else:
        print('------------- `Y` to accept, `N` to deny --------------')


##--------------------------------------------------------------------------------------------------



if __name__ == '__main__':

    partition = 'partition_8'

    all_file = glob('/Users/feiran-l/Desktop/tmp/processed_hypersim/{}/*/*'.format(partition))
    save_dir = '/Users/feiran-l/Desktop/tmp/selected_hypersim/{}'.format(partition)
    Path(save_dir).mkdir(exist_ok=True, parents=True)

    all_file = [x for x in all_file if 'cam_01' in x]

    print('{} data to proceed'.format(len(all_file)))


    for i, curr_dir in enumerate(all_file):
        curr_name = curr_dir.split('/')[-1]
        ## load data
        curr_data = load_h5py_file(curr_dir)
        rgb, dep, normal, M = curr_data['rgb'], curr_data['dep'], curr_data['normal'], curr_data['normal']
        mask = np.zeros_like(dep)
        mask[dep == 0] = 1
        assert mask.sum() == 0, RuntimeError('{} mask sums to {}'.format(curr_dir, mask.sum()))

        ## plot
        fig, ax = plt.subplots(1, 4, figsize=(15, 3))
        fig.canvas.mpl_connect('key_press_event', on_press)
        for j, curr_dir in enumerate([rgb, dep, normal, mask]):
            if j == 2:
                ax[j].imshow(curr_dir / 2 + 0.5)
            else:
                ax[j].imshow(curr_dir)
        ax[0].set_title(' {}/{}'.format(i, len(all_file)))
        ax[3].set_title('{}'.format(mask.sum()))
        plt.show()