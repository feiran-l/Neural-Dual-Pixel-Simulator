import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, bn=False):
        super(BasicConv, self).__init__()
        if bn:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU())
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.ReLU())

    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False, bn=False):
        super(Down, self).__init__()
        m = [nn.MaxPool2d(2), BasicConv(in_channels, out_channels, bn)]
        if dropout:
            m.append(nn.Dropout2d())
        self.maxpool_conv = nn.Sequential(*m)

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bn=False):
        super(Up, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv = BasicConv(in_channels + in_channels // 2, out_channels, bn)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channel=3, out_channel=25, bn=False):
        super(UNet, self).__init__()
        if bn:
            self.inc = nn.Sequential(nn.Conv2d(in_channel, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                                     nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        else:
            self.inc = nn.Sequential(nn.Conv2d(in_channel, 64, 3, padding=1), nn.ReLU(),
                                     nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.down1 = Down(64, 128, bn=bn)
        self.down2 = Down(128, 256, bn=bn)
        self.down3 = Down(256, 512, bn=bn)
        self.down4 = Down(512, 1024, dropout=True, bn=bn)
        self.fm = nn.Sequential(BasicConv(1024, 1024, bn=bn), nn.Dropout2d())
        self.up1 = Up(1024, 512, bn=bn)
        self.up2 = Up(512, 256, bn=bn)
        self.up3 = Up(256, 128, bn=bn)
        self.up4 = Up(128, 64, bn=bn)
        self.outc = nn.Conv2d(64, out_channel, 3, padding=1)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight.data, 1)
                nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.fm(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


class KernelConv(nn.Module):
    def __init__(self, pred_k_size=5):
        super(KernelConv, self).__init__()
        self.k_size = pred_k_size

    def forward(self, fm, kernel, dilation=1):
        """
        :param fm: input feature map: [bs, c, h, w]
        :param kernel: the deconv_kernel: [bs, k*k, h, w]
        :param dilation: when dilation=1, the kernel size is the same as pred_k_size
        """
        bs, c, h, w = fm.shape
        pad_size = (self.k_size // 2) * dilation
        feat_padded = F.pad(fm, [pad_size, pad_size, pad_size, pad_size])
        kernel = torch.flip(kernel, dims=[1])  # change from correlation to convolution: [bs, k*k, h, w]
        # shift feature map to align the K*K kernel region
        img_stack = []
        for i in range(self.k_size):
            for j in range(self.k_size):
                iii = feat_padded[..., i * dilation:i * dilation + h, j * dilation:j * dilation + w]  # [bs, c, h, w]
                img_stack.append(iii)
        img_stack = torch.stack(img_stack, dim=1)  # [bs, k*k, c, h, w]
        # sum over k*k to get per-pixel conv res
        res = torch.sum(kernel.unsqueeze(2) * img_stack, dim=1, keepdim=False)  # [bs, c, h, w]
        return res



class DPSimulator(nn.Module):
    def __init__(self, k_size=5, bn=True):
        super(DPSimulator, self).__init__()
        self.k_size = k_size
        self.kpn = UNet(in_channel=5, out_channel=k_size**2 + 6, bn=bn)
        self.kernel_conv = KernelConv(pred_k_size=k_size)
        self.psf_norm = nn.Softmax(dim=1)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight.data, 1)
                nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, sharp, dep, coc):
        bs, _, h, w = sharp.shape
        k = self.kpn(torch.cat([sharp, dep, coc], dim=1))
        s = self.k_size ** 2
        k_l, adder_l, adder_r = k[:, :s, :, :],  k[:, s:s+3, :, :], k[:, s+3:, :, :]
        ## note that the following flip should be 1!
        k_r = torch.flip(k_l.reshape(bs, 5, 5, h, w), [1, ]).reshape(bs, 25, h, w)

        k_l, k_r = self.psf_norm(k_l), self.psf_norm(k_r)   # make sure that psfs are non-negative and sums to 1
        pred_l, pred_r = self.kernel_conv(sharp, k_l) + adder_l, self.kernel_conv(sharp, k_r) + adder_r
        return pred_l, pred_r, adder_l, adder_r



##----------------------------------------------------------------------------


if __name__ == '__main__':
    
    pass


    # net = DPSimulator(k_size=3, norm=True)
    # sharp, dep, coc = torch.rand(1, 3, 256, 256), torch.rand(1, 1, 256, 256), torch.rand(1, 1, 256, 256)
    # l, r = net(sharp, dep, coc)
    # print(l.shape, r.shape)


    # net = UNet()
    # inp = torch.rand(1, 3, 1680//4, 1120//4)
    # res = net(inp)
    #
    # print(res.shape)


   


