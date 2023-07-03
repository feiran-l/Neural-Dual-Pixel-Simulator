import numpy as np
from pathlib import Path
import os
from glob import glob
from tqdm import tqdm
import shutil
import random



if __name__ == '__main__':

    np.random.seed(0)
    random.seed(0)

    train_size, valid_size, test_size = 333, 55, 55
    base_dir = '/media/li/52F3-84AF/DP_data/final_data'

    part1 = glob(os.path.join(base_dir, 'part1/*'))
    part2 = glob(os.path.join(base_dir, 'part2/*'))
    part3 = glob(os.path.join(base_dir, 'part3/*'))
    all_file = ['/'.join(x.strip().split('/')[-2:]) for x in part1 + part2 + part3]
    print('{} data in total'.format(len(all_file)))

    all_ids = np.arange(len(all_file))
    random.shuffle(all_ids)

    train = np.array([all_file[i] for i in sorted(all_ids[:train_size])])
    valid = np.array([all_file[i] for i in sorted(all_ids[train_size:train_size+valid_size])])
    test = np.array([all_file[i] for i in sorted(all_ids[train_size+valid_size:])])

    print(train.shape, valid.shape, test.shape)

    ## prepare data
    for name in tqdm(train):
        save_dir = os.path.join(base_dir, 'train', name.split('/')[-1])
        shutil.move(os.path.join(base_dir, name), save_dir)

    for name in tqdm(valid):
        save_dir = os.path.join(base_dir, 'valid', name.split('/')[-1])
        shutil.move(os.path.join(base_dir, name), save_dir)

    for name in tqdm(test):
        save_dir = os.path.join(base_dir, 'test', name.split('/')[-1])
        shutil.move(os.path.join(base_dir, name), save_dir)

