import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import math



class DPLoss(nn.Module):
    def __init__(self, loss_type='charbonnier', use_mask=True):
        super(DPLoss, self).__init__()
        self.loss_type = loss_type
        self.eps = 1e-3
        self.use_mask = use_mask

    def forward(self, pred_l, pred_r, gt_l, gt_r, in_mask, pred_c=None, gt_c=None):
        mask = torch.ones_like(in_mask) if not self.use_mask else in_mask

        diff_l, diff_r = mask * torch.abs(pred_l - gt_l), mask * torch.abs(pred_r - gt_r)
        if self.loss_type == 'charbonnier':
            loss_l = torch.sum(torch.sqrt((diff_l * diff_l) + (self.eps * self.eps))) / torch.sum(mask)
            loss_r = torch.sum(torch.sqrt((diff_r * diff_r) + (self.eps * self.eps))) / torch.sum(mask)
            if pred_c is not None and gt_c is not None:
                diff_c = mask * torch.abs(pred_c - gt_c)
                loss_c = torch.sum(torch.sqrt((diff_c * diff_c) + (self.eps * self.eps))) / torch.sum(mask)
            else:
                loss_c = 0
        elif self.loss_type == 'l1':
            loss_l, loss_r = torch.sum(diff_l) / torch.sum(mask), torch.sum(diff_r) / torch.sum(mask)
            if pred_c is not None and gt_c is not None:
                diff_c = mask * torch.abs(pred_c - gt_c)
                loss_c = torch.sum(diff_c) / torch.sum(mask)
            else:
                loss_c = 0
        elif self.loss_type == 'l2':
            loss_l, loss_r = torch.sum(diff_l ** 2) / torch.sum(mask), torch.sum(diff_r ** 2) / torch.sum(mask)
            if pred_c is not None and gt_c is not None:
                diff_c = mask * torch.abs(pred_c - gt_c)
                loss_c = torch.sum(diff_c ** 2) / torch.sum(mask)
            else:
                loss_c = 0
        else:
            raise NotImplementedError('The specified loss type {} is not implemented'.format(self.loss_type))
        return loss_l + loss_r + loss_c



    
class EdgeLoss(nn.Module):
    def __init__(self, device='cpu', use_mask=True):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1).to(device)
        self.eps = 1e-3
        self.use_mask = use_mask
        
    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)  # filter
        down = filtered[:, :, ::2, ::2]  # downsample
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4  # upsample
        filtered = self.conv_gauss(new_filter)  # filter
        diff = current - filtered
        return diff

    def forward(self, x, y, in_mask):
        mask = torch.ones_like(in_mask) if not self.use_mask else in_mask
        
        filtered_x, filtered_y = mask * self.laplacian_kernel(x), mask * self.laplacian_kernel(y), 
        loss = torch.sum(torch.sqrt((filtered_x - filtered_y) ** 2 + (self.eps * self.eps))) / torch.sum(mask)
        return loss
    
    

    

##------------------------------------------------
    
    
    
if __name__ == '__main__':


    a = torch.rand(2, 3, 412, 412)
    b = torch.rand(2, 3, 412, 412)

    cri = DPLoss(use_mask=False)
    res = cri(a, a, b, b, a)
    print(res)