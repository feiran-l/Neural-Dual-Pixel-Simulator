import imageio
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from torch.utils.data import Dataset
import h5py
from custom_transform import image_transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import functools
import random
from glob import glob
import imageio



def load_img(img_dir, normalize=True):
    img = imageio.imread(img_dir)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
    if normalize:
        if img.dtype == np.uint8:
            bit_depth = 8
        elif img.dtype == np.uint16:
            bit_depth = 16
        else:
            raise NotImplementedError('can only normalize uint8 or uint16 images!')
        mean, norm_val = 0, (2 ** bit_depth) - 1
        img = (img - mean) / norm_val
    return np.float32(img)



def load_h5py_file(data_dir):
    res = {}
    calib_data_h5 = h5py.File(data_dir, 'r')
    for k, v in calib_data_h5.items():
        res[k] = np.array(v)
    calib_data_h5.close()
    return res



def generate_coc_map(depth, focus_dis, f_number, pixel_size, focal_len):
    """ generate the signed CoC map in pixel unit. depth and focus_dis in mm, focal_len in pixel """
    with np.errstate(divide='ignore'):
        coc = ((depth - focus_dis) / depth) * (focal_len ** 2 / (f_number * (focus_dis / pixel_size - focal_len)))
    coc[depth == 0] = 0
    return coc



def collate_fn_replace_corrupted(batch, torch_dataset):
    """
        Collate function that allows to replace corrupted examples in the dataloader.
        It expects that the __get_item__ in the customed dataset returns 'None' when that occurs.
        The 'None's in the batch are replaced with another examples sampled randomly.
        Usage:
            specify the collect_fn in dataloader with functools.partial(collate_fn_replace_corrupted, your_dataset)
        Ref:
            https://stackoverflow.com/a/69578320/13176870
    """

    original_batch_len = len(batch)
    batch = list(filter(lambda x: x is not None, batch))  # Filter out all the Nones (corrupted examples)
    filtered_batch_len = len(batch)
    # Num of corrupted examples
    diff = original_batch_len - filtered_batch_len
    if diff > 0:
        # Replace corrupted examples with another examples randomly
        batch.extend([torch_dataset[random.randint(0, len(torch_dataset) - 1)] for _ in range(diff)])
        # Recursive call to replace the replacements if they are corrupted
        return collate_fn_replace_corrupted(batch, torch_dataset)
    # Finally, when the whole batch is fine, return it
    return torch.utils.data.dataloader.default_collate(batch)



class MyDataset(Dataset):
    def __init__(self, data_dir, partition, train_size=(512, 512), required_dep_percent=0.8, thin_lens_focal_len_in_mm=85, pixel_size=36 / 1680,
                 roi_0=[38, 1062, 88, 1112], roi_1=[38, 1062, 488, 1512]):
        """
        @param data_dir: data directory
        @param partition: eiter 'train' or 'val'
        @param train_size: patch size to crop to in training
        @param required_dep_percent: minimum percent of valid depths for a patch to be used in training
        @param thin_lens_focal_len_in_mm: focal length when camera is modeled as thin-lens, the number shown on the lens
        @param pixel_size: sensor board size / resolution
        @param roi_0: roi of cam_0, in order [begin_r, end_r, begin_c, end_c]
        @param roi_1: roi of cam_1, in order [begin_r, end_r, begin_c, end_c]
        """
        super(MyDataset, self).__init__()
        # parameters
        self.partition, self.required_dep_percent = partition, required_dep_percent
        self.pixel_size = pixel_size
        self.focal_len_in_pixel = thin_lens_focal_len_in_mm / self.pixel_size
        self.transform = image_transforms(tar_size=train_size)
        self.roi_0 = roi_0
        self.roi_1 = roi_1
        # list all names
        name_list = os.listdir(os.path.join(data_dir, partition))
        self.blur_list = [os.path.join(data_dir, partition, x, 'cam{}'.format(z), 'FN_{}.png'.format(y))
                          for x in name_list for z in ['0', '1'] for y in ['1.8', '2', '2.8', '4', '5.6']]

    def __len__(self):
        return len(self.blur_list)

    def _cut_by_roi(self, img, roi):
        return img[:, roi[0]:roi[1], roi[2]:roi[3]]

    def _formulate_data(self, sharp, dep, coc, dp_l, dp_r, dp_c, cam_id_str):
        sharp, dep, coc = torch.tensor(sharp.transpose(2, 0, 1)), torch.tensor(dep.transpose(2, 0, 1)), torch.tensor(coc.transpose(2, 0, 1))
        dp_l, dp_r, dp_c = torch.tensor(dp_l.transpose(2, 0, 1)), torch.tensor(dp_r.transpose(2, 0, 1)), torch.tensor(dp_c.transpose(2, 0, 1))
        if cam_id_str == '0':
            roi = self.roi_0
        elif cam_id_str == '1':
            roi = self.roi_1
        sharp, dep, coc = self._cut_by_roi(sharp, roi), self._cut_by_roi(dep, roi), self._cut_by_roi(coc, roi)
        dp_l, dp_r, dp_c = self._cut_by_roi(dp_l, roi), self._cut_by_roi(dp_r, roi), self._cut_by_roi(dp_c, roi)
        res = {'sharp': sharp, 'dep': dep, 'coc': coc, 'dp_l': dp_l, 'dp_r': dp_r, 'dp_c': dp_c}
        return res

    def __getitem__(self, idx):
        ## STEP-1: extract names
        blur_name = self.blur_list[idx]
        if blur_name[0] == '/':
            scene_name = os.path.join('/', *blur_name.split('/')[:-2])
        else:
            scene_name = os.path.join(*blur_name.split('/')[:-2])
        cam_id_str, F_number_str = blur_name.split('/')[-2][-1], blur_name.split('/')[-1][:-4].split('_')[1]
        ## STEP-2: load data
        sharp = load_img(os.path.join(scene_name, 'cam{}/FN_22.png'.format(cam_id_str)), normalize=True)
        dep = load_img(os.path.join(scene_name, 'cam{}/dep.png'.format(cam_id_str)), normalize=False)
        dp_l = load_img(os.path.join(scene_name, 'cam{}/FN_{}_l.png'.format(cam_id_str, F_number_str)), normalize=True)
        dp_r = load_img(os.path.join(scene_name, 'cam{}/FN_{}_r.png'.format(cam_id_str, F_number_str)), normalize=True)
        dp_c = load_img(os.path.join(scene_name, 'cam{}/FN_{}_c.png'.format(cam_id_str, F_number_str)), normalize=True)
        meta = load_h5py_file(os.path.join(scene_name, 'meta_data.h5'))
        focus_dis = meta['focus_dis_{}'.format(cam_id_str)]
        coc = generate_coc_map(dep, focus_dis=focus_dis, f_number=float(F_number_str), pixel_size=self.pixel_size, focal_len=self.focal_len_in_pixel)
        ## STEP-3: formulate data and do augmentation
        res = self._formulate_data(sharp, dep, coc, dp_l, dp_r, dp_c, cam_id_str)
        res['curr_name'] = '{}_cam{}_fn{}'.format(scene_name.split('/')[-1], cam_id_str, F_number_str)
        res['focus_dis'] = focus_dis
        res['focus_pt'] = meta['AF_point_{}'.format(cam_id_str)]
        if self.partition == 'train':
            res = self.transform(res)
            mask = torch.where(res['dep'] == 0, 0, 1)
            dep_percent = mask.sum() / (mask.shape[-1] * mask.shape[-2])
            if dep_percent <= self.required_dep_percent:
                res = None
        return res





class HypersimDataset(Dataset):
    def __init__(self, data_dir):
        super(HypersimDataset, self).__init__()
        self.all_frame_list = glob(os.path.join(data_dir, '*.h5'))

    def __len__(self):
        return len(self.all_frame_list)

    def __getitem__(self, idx):
        data = load_h5py_file(self.all_frame_list[idx])
        sharp, dep = data['rgb'] / 255.0, data['dep']
        focus_dis, thin_lens_focal_len_in_mm, f_number, pixel_size = data['focus_dis'], data['thin_lens_focal_len_in_mm'], data['F_number'], data['pixel_size']

        coc = generate_coc_map(depth=dep, focus_dis=focus_dis, f_number=f_number, pixel_size=pixel_size, focal_len=thin_lens_focal_len_in_mm)
        res = {'sharp': torch.tensor(sharp.transpose(2, 0, 1)).float(), 'dep': torch.tensor(dep[:, :, None].transpose(2, 0, 1)).float(),
               'coc': torch.tensor(coc[:, :, None].transpose(2, 0, 1)).float(), 'focus_dis': focus_dis, 'thin_lens_focal_len_in_mm': thin_lens_focal_len_in_mm, 'f_number': f_number,
               'M': data['M'], 'pixel_size': pixel_size, 'af_pt': data['af_pt'], 'normal': torch.tensor(data['normal'].transpose(2, 0, 1)).float(), 
               'curr_name': self.all_frame_list[idx].split('/')[-1][:-3]}
        return res


##-----------------------------------------------------------------



if __name__ == '__main__':
    np.random.seed(0)

    """ my DP dataset """
    data_dir = '/dataset/workspace2021/li/final_data'
    dataset = MyDataset(data_dir, partition='test', required_dep_percent=0.9)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=True,
                        collate_fn=functools.partial(collate_fn_replace_corrupted, torch_dataset=dataset))

    for data in tqdm(loader):
        sharp, dep, coc = data['sharp'], data['dep'], data['coc']
        dp_l, dp_r, dp_c = data['dp_l'], data['dp_r'], data['dp_c']
#         fig, ax = plt.subplots(1, 5, figsize=(12, 3))
#         for i, (x, name) in enumerate(zip([sharp, dp_l, dp_r, dep, coc], ['sharp', 'dp_l', 'dp_r', 'dep', 'coc'])):
#             ax[i].imshow(x[0].permute(1, 2, 0))
#             if name == 'dep':
#                 mask = torch.where(dep == 0, 0, 1)
#                 ax[i].set_title('cover%: {:.1%}'.format(mask.sum() / (mask.shape[-1] * mask.shape[-2])))
#             else:
#                 ax[i].set_title(name)
#         plt.tight_layout()
#         # plt.show(block=False)
#         # plt.pause(0.3)
#         # plt.close()
#         plt.show()


    """ hypersim rgbd dataset """
    # data_dir = '/media/li/52F3-84AF/selected_hypersim'
    # dataset = HypersimDataset(data_dir)
    # loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)
    #
    # for data in tqdm(loader):
    #     sharp, dep, coc, normal = data['sharp'], data['dep'], data['coc'], data['normal']
    #     fig, ax = plt.subplots(1, 4, figsize=(12, 3))
    #     for i, (x, name) in enumerate(zip([sharp, dep, coc, normal], ['sharp', 'dep', 'coc', 'normal'])):
    #         if name != 'normal':
    #             ax[i].imshow(x[0].permute(1, 2, 0))
    #         else:
    #             ax[i].imshow(x[0].permute(1, 2, 0) / 2 + 0.5)
    #     ax[0].set_title('{:.2f}'.format(data['thin_lens_focal_len_in_mm'][0]))
    #     ax[2].set_title('{:.2f}, {:.2f}'.format(coc.min(), coc.max()))
    #     # plt.tight_layout()
    #     # plt.show(block=False)
    #     # plt.pause(0.3)
    #     # plt.close()
    #     plt.show()















