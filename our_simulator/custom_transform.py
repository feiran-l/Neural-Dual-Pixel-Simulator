import torch
import torchvision.transforms as transforms
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image



def image_transforms(augment_parameters=[0.8, 1.2, 0.8, 1.2, 0.8, 1.2], tar_size=(512, 512)):
    """ input to transform should be images in [c, h, w], normalized to [0, 1] """
    data_transform = transforms.Compose([RandomVerticalFlip(), AugmentColor(augment_parameters), RandomCrop(tar_size)])
    # data_transform = transforms.Compose([RandomVerticalFlip(), RandomCrop(tar_h, tar_w)])
    return data_transform



class RandomVerticalFlip(object):
    def __init__(self):
        self.transform = transforms.RandomVerticalFlip(p=1)

    def __call__(self, data):
        sharp, dep, coc = data['sharp'], data['dep'], data['coc']
        dp_l, dp_r, dp_c = data['dp_l'], data['dp_r'], data['dp_c']
        k = np.random.uniform(0, 1, 1)
        if k > 0.5:
            sharp, dep, coc = self.transform(sharp), self.transform(dep), self.transform(coc)
            dp_l, dp_r, dp_c = self.transform(dp_l), self.transform(dp_r), self.transform(dp_c)
        res = {'sharp': sharp, 'dep': dep, 'coc': coc, 'dp_l': dp_l, 'dp_r': dp_r, 'dp_c': dp_c}
        return res



class AugmentColor(object):
    def __init__(self, augment_parameters):
        self.gamma_low = augment_parameters[0]  # 0.8
        self.gamma_high = augment_parameters[1]  # 1.2
        self.brightness_low = augment_parameters[2]  # 0.8
        self.brightness_high = augment_parameters[3]  # 1.2
        self.color_low = augment_parameters[4]  # 0.8
        self.color_high = augment_parameters[5]  # 1.2

    def __call__(self, data):
        sharp, dep, coc = data['sharp'], data['dep'], data['coc']
        dp_l, dp_r, dp_c = data['dp_l'], data['dp_r'], data['dp_c']
        k = np.random.uniform(0, 1, 1)
        if k > 0.5:
            # randomly shift gamma
            random_gamma = np.random.uniform(self.gamma_low, self.gamma_high)
            sharp, dp_l, dp_r, dp_c = sharp ** random_gamma, dp_l ** random_gamma, dp_r ** random_gamma, dp_c ** random_gamma

            # randomly shift brightness
            random_brightness = np.random.uniform(self.brightness_low, self.brightness_high)
            sharp, dp_l, dp_r, dp_c = sharp * random_brightness, dp_l * random_brightness, dp_r * random_brightness, dp_c * random_brightness

            # randomly shift color
            random_colors = np.random.uniform(self.color_low, self.color_high, 3)
            for i in range(3):
                sharp[i, :, :] *= random_colors[i]
                dp_l[i, :, :] *= random_colors[i]
                dp_r[i, :, :] *= random_colors[i]
                dp_c[i, :, :] *= random_colors[i]

            # saturate
            sharp, dp_l, dp_r, dp_c = torch.clamp(sharp, 0, 1), torch.clamp(dp_l, 0, 1), torch.clamp(dp_r, 0, 1), torch.clamp(dp_c, 0, 1)

        res = {'sharp': sharp, 'dep': dep, 'coc': coc, 'dp_l': dp_l, 'dp_r': dp_r, 'dp_c': dp_c}
        return res



class RandomCrop(object):
    def __init__(self, tar_size):
        self.tar_size = tar_size

    def __call__(self, data):
        if self.tar_size is not None:
            tar_w, tar_h = self.tar_size[0], self.tar_size[1]
            sharp, dep, coc = data['sharp'], data['dep'], data['coc']
            dp_l, dp_r, dp_c = data['dp_l'], data['dp_r'], data['dp_c']
            if tar_h >= sharp.shape[1]:
                init_h, end_h = 0, sharp.shape[1]
            else:
                init_h = np.random.randint(0, sharp.shape[1] - tar_h, 1)[0]
                end_h = init_h + tar_h
            if tar_w >= sharp.shape[2]:
                init_w, end_w = 0, sharp.shape[2]
            else:
                init_w = np.random.randint(0, sharp.shape[2] - tar_w, 1)[0]
                end_w = init_w + tar_w
            sharp, dep, coc = sharp[:, init_h:end_h, init_w:end_w], dep[:, init_h:end_h, init_w:end_w], coc[:, init_h:end_h, init_w:end_w]
            dp_l, dp_r, dp_c = dp_l[:, init_h:end_h, init_w:end_w], dp_r[:, init_h:end_h, init_w:end_w], dp_c[:, init_h:end_h, init_w:end_w]
            res = {'sharp': sharp, 'dep': dep, 'coc': coc, 'dp_l': dp_l, 'dp_r': dp_r, 'dp_c': dp_c}
            return res
        else:
            return data



##-----------------------------------------------------------------



if __name__ == '__main__':

    sharp, dep, coc = torch.rand(3, 1120, 1024), torch.rand(1, 1120, 1024), torch.rand(1, 1120, 1024)
    dp_l, dp_r, dp_c = torch.rand(3, 1120, 1024), torch.rand(3, 1120, 1024), torch.rand(3, 1120, 1024)
    data = {'sharp': sharp, 'dep': dep, 'coc': coc, 'dp_l': dp_l, 'dp_r': dp_r,  'dp_c': dp_c}
    trans = image_transforms(tar_size=None)

    res = trans(data)

    for k, v in res.items():
        print(k, v.shape)