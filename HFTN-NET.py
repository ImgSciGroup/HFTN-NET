# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 10:22:51 2022

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 16:09:00 2022

@author: Administrator
"""


#from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.padding import ReplicationPad2d

class PDM(nn.Module):

    def __init__(self, in_channels):
        super(PDM, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        # self.batch1 = nn.BatchNorm2d(64)
        # self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # self.batch2 = nn.BatchNorm2d(64)
        # self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # self.batch3 = nn.BatchNorm2d(64)
        # self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        # self.batch4 = nn.BatchNorm2d(64)
        # self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        # self.batch5 = nn.BatchNorm2d(64)
        # self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(32, 3, kernel_size=3, padding=1)
        # self.batch6 = nn.BatchNorm2d(3)
        # self.relu6 = nn.ReLU(inplace=True)
        self.conv7 = nn.Conv2d(6, 3, kernel_size=3, padding=1)
        self.avg_pool = nn.AdaptiveAvgPool2d((None,None))
        self.max_pool = nn.AdaptiveMaxPool2d((None,None))

        self.conv8 = nn.Conv2d(3, 3, kernel_size=1, padding=0)
        self.conv9 = nn.Conv2d(3, 3, kernel_size=1, padding=0)
        self.conv10 = nn.Conv2d(3, 3, kernel_size=1, padding=0)
    def forward(self, x):

        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)

        x1_adpavg = self.avg_pool(x1)
        x1_adpmax = self.max_pool(x1)
        x1_max = F.max_pool2d(x1, kernel_size=1, stride=1)

        x1_adpavg = self.conv4(x1_adpavg)
        x1_adpavg = self.conv5(x1_adpavg)
        x1_adpavg = self.conv6(x1_adpavg)

        x1_adpmax = self.conv4(x1_adpmax)
        x1_adpmax = self.conv5(x1_adpmax)
        x1_adpmax = self.conv6(x1_adpmax)

        x1_max = self.conv4(x1_max)
        x1_max = self.conv5(x1_max)
        x1_max = self.conv6(x1_max)

        x_fusion1 = torch.cat([x1_max,x1_adpavg],dim=1)
        x_fusion2 = torch.cat([x1_max,x1_adpmax], dim=1)
        x_fusion3 = torch.cat([x1_adpavg, x1_adpmax], dim=1)

        x_denoise1 = self.conv7(x_fusion1)
        x_denoise2 = self.conv7(x_fusion2)
        x_denoise3 = self.conv7(x_fusion3)

        x_dnfusion1 = self.conv7(torch.cat([x_denoise1,x_denoise2],dim=1))
        x_dnfusion2 = self.conv7(torch.cat([x_denoise2,x_denoise3], dim=1))

        x_denoise = self.conv7(torch.cat([x_dnfusion1,x_dnfusion2], dim=1))

        after_x1 = self.conv8(x_denoise)
        after_x2 = self.conv8(after_x1)
        after_x3 = self.conv8(after_x2)
        # after_x = self.conv7(torch.cat([after_x3,after_x1],dim=1))
        return after_x3

class Conv_block1(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Conv_block1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    def forward(self, x):
        x1 = self.conv1(x)
        return x1
class Conv_block2(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Conv_block2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        return x1

class Conv_block3(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Conv_block3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)

    def forward(self, x):
        x1 = self.conv1(x)
        return x1

class FDM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FDM, self).__init__()

        self.conv1 = nn.Conv2d(6, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(2 * in_channels, out_channels, kernel_size=3, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x1, x2):
        x_diff = torch.abs(x1 - x2)

        x1_max = self.max_pool(x1)
        x2_max = self.max_pool(x2)

        x_1 = torch.cat([x1_max, x_diff], 1)

        x_1_conv = self.conv2(x_1)
        x_2 = torch.cat([x2_max, x_diff], 1)
        x_2_conv = self.conv2(x_2)

        res = torch.cat([x_1_conv, x_2_conv], 1)
        res = self.conv1(res)

        return res

class HFTNet(nn.Module):

    def __init__(self):
        super(HFTNet, self).__init__()

        # self.conv11_1 = Conv_block1(3, 16)
        # self.conv12_1 = Conv_block1(16, 16)
        #
        # self.conv21_1 = Conv_block1(16,32)
        # self.conv22_1 = Conv_block1(32, 32)
        #
        # self.conv31_1 = Conv_block1(32, 128)
        # self.conv32_1 = Conv_block1(128, 128)
        # self.conv33_1 = Conv_block1(128, 128)
        #
        # self.conv41_1 = Conv_block1(128, 256)
        # self.conv42_1 = Conv_block1(256, 256)
        # self.conv43_1 = Conv_block1(256, 256)
        #
        # self.conv11_3 = Conv_block2(3, 16)
        # self.conv12_3 = Conv_block2(16, 16)
        #
        # self.conv21_3 = Conv_block2(16,32)
        # self.conv22_3 = Conv_block2(32, 32)
        #
        # self.conv31_3 = Conv_block2(32, 128)
        # self.conv32_3 = Conv_block2(128, 128)
        # self.conv33_3 = Conv_block2(128, 128)
        #
        # self.conv41_3 = Conv_block2(128, 256)
        # self.conv42_3 = Conv_block2(256, 256)
        # self.conv43_3 = Conv_block2(256, 256)
        #
        # self.conv11_5 = Conv_block3(3, 16)
        # self.conv12_5 = Conv_block3(16, 16)
        #
        # self.conv21_5 = Conv_block3(16,32)
        # self.conv22_5 = Conv_block3(32, 32)
        #
        # self.conv31_5 = Conv_block3(32, 128)
        # self.conv32_5 = Conv_block3(128, 128)
        # self.conv33_5 = Conv_block3(128, 128)
        #
        # self.conv41_5 = Conv_block3(128, 256)
        # self.conv42_5 = Conv_block3(256, 256)
        # self.conv43_5 = Conv_block3(256, 256)

        # kernel 1
        self.conv11_1 = Conv_block1(3, 16)
        self.conv12_1 = Conv_block1(16, 16)

        self.conv21_1 = Conv_block1(16, 64)
        self.conv22_1 = Conv_block1(64, 64)

        self.conv31_1 = Conv_block1(64, 128)
        self.conv32_1 = Conv_block1(128, 128)
        self.conv33_1 = Conv_block1(128, 128)

        self.conv41_1 = Conv_block1(128, 512)
        self.conv42_1 = Conv_block1(512, 512)
        self.conv43_1 = Conv_block1(512, 512)
        # kernel 3
        self.conv11_3 = Conv_block2(3, 16)
        self.conv12_3 = Conv_block2(16, 16)

        self.conv21_3 = Conv_block2(16, 64)
        self.conv22_3 = Conv_block2(64, 64)

        self.conv31_3 = Conv_block2(64, 128)
        self.conv32_3 = Conv_block2(128, 128)
        self.conv33_3 = Conv_block2(128, 128)

        self.conv41_3 = Conv_block2(128, 512)
        self.conv42_3 = Conv_block2(512, 512)
        self.conv43_3 = Conv_block2(512, 512)
        # kernel 5
        self.conv11_5 = Conv_block3(3, 16)
        self.conv12_5 = Conv_block3(16, 16)

        self.conv21_5 = Conv_block3(16, 64)
        self.conv22_5 = Conv_block3(64, 64)

        self.conv31_5 = Conv_block3(64, 128)
        self.conv32_5 = Conv_block3(128, 128)
        self.conv33_5 = Conv_block3(128, 128)

        self.conv41_5 = Conv_block3(128, 512)
        self.conv42_5 = Conv_block3(512, 512)
        self.conv43_5 = Conv_block3(512, 512)
        #upsample
        self.upconv4 = nn.ConvTranspose2d(1536, 1536, kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv43d = nn.ConvTranspose2d(3072, 256, kernel_size=3, padding=1)
        self.conv42d = nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1)
        self.conv41d = nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1)


        self.upconv3 = nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv33d = nn.ConvTranspose2d(640, 128, kernel_size=3, padding=1)
        self.conv32d = nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1)
        self.conv31d = nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv22d = nn.ConvTranspose2d(320, 64, kernel_size=3, padding=1)
        self.conv21d = nn.ConvTranspose2d(64 ,32, kernel_size=3, padding=1)


        self.upconv1 = nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv12d = nn.ConvTranspose2d(64, 3, kernel_size=3, padding=1)
        self.conv11d = nn.ConvTranspose2d(3, 1, kernel_size=3, padding=1)


        self.feature_noise = PDM(3)
        self.enhance = FDM(3, 3)
        # self.net1 = HFTNet1()
        # self.net2 = HFTNet2()
        # self.net3 = HFTNet3()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(1)
        self.conv2 = nn.Conv2d(2, 1, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(1)
        self.conv3 = nn.Conv2d(2, 1, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(1)

        self.conv4 = nn.Conv2d(2, 1, kernel_size=1, padding=0)
        self.bn4 = nn.BatchNorm2d(1)
        self.conv5 = nn.Conv2d(2, 1, kernel_size=1, padding=0)
        self.bn5 = nn.BatchNorm2d(1)
        # self.upconv4 = nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1, stride=2, output_padding=1)
        #
        # self.conv43d = nn.ConvTranspose2d(512, 256, kernel_size=3, padding=1)
        # self.conv42d = nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1)
        # self.conv41d = nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1)
        #
        #
        # self.upconv3 = nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1, stride=2, output_padding=1)
        #
        # self.conv33d = nn.ConvTranspose2d(384, 128, kernel_size=3, padding=1)
        # self.conv32d = nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1)
        # self.conv31d = nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1)
        #
        #
        # self.upconv2 = nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1, stride=2, output_padding=1)
        #
        # self.conv22d = nn.ConvTranspose2d(160, 64, kernel_size=3, padding=1)
        # self.conv21d = nn.ConvTranspose2d(64 ,32, kernel_size=3, padding=1)
        #
        #
        # self.upconv1 = nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1, stride=2, output_padding=1)
        #
        # self.conv12d = nn.ConvTranspose2d(32, 3, kernel_size=3, padding=1)
        # self.conv11d = nn.ConvTranspose2d(3, 1, kernel_size=3, padding=1)
        #
        #
        # self.conv = nn.Conv2d(256,128,kernel_size=3,padding=1)
        #
        # self.outc1 = nn.Conv2d(3,8,kernel_size=3,padding=1)
        # self.outc2 = nn.Conv2d(1,1,kernel_size=3,padding=1)
        # self.outc3 = nn.Conv2d(3,16,kernel_size=3,padding=1)
        # self.outc4 = nn.Conv2d(16,128,kernel_size=3,padding=1)
        # self.outc5 = nn.Conv2d(128,128,kernel_size=3,padding=1)
        # self.outc6 = nn.Conv2d(128,512,kernel_size=3,padding=1)
        #
        # self.last_conv = nn.ConvTranspose2d(33, 1, kernel_size=3, padding=1, stride=2, output_padding=1)

    def forward(self, x1, x2):
        """Forward method"""
        x1_dn = self.feature_noise(x1)
        x2_dn = self.feature_noise(x2)

        x_diff = self.enhance(x1,x2)
        #kernel 1
        T1_x1_conv1_1 = F.relu(self.conv11_1(x1_dn))
        T1_x1_conv1_2 = F.relu(self.conv12_1(T1_x1_conv1_1))
        x1p_T1_kernel1_max1 = F.max_pool2d(T1_x1_conv1_2, kernel_size=2, stride=2)

        T1_x1_conv2_1 = F.relu(self.conv21_1(x1p_T1_kernel1_max1))
        T1_x1_conv2_2 = F.relu(self.conv22_1(T1_x1_conv2_1))
        x1p_T1_kernel1_max2 = F.max_pool2d(T1_x1_conv2_2, kernel_size=2, stride=2)

        T1_x1_conv3_1 = F.relu(self.conv31_1(x1p_T1_kernel1_max2))
        T1_x1_conv3_2 = F.relu(self.conv32_1(T1_x1_conv3_1))
        T1_x1_conv3_3 = F.relu(self.conv33_1(T1_x1_conv3_2))
        x1p_T1_kernel1_max3 = F.max_pool2d(T1_x1_conv3_3, kernel_size=2, stride=2)

        T1_x1_conv4_1 = F.relu(self.conv41_1(x1p_T1_kernel1_max3))
        T1_x1_conv4_2 = F.relu(self.conv42_1(T1_x1_conv4_1))
        T1_x1_conv4_3 = F.relu(self.conv43_1(T1_x1_conv4_2))
        x1p_T1_kernel1_max4 = F.max_pool2d(T1_x1_conv4_3, kernel_size=2, stride=2)

        T2_x1_conv1_1 = F.relu(self.conv11_1(x2_dn))
        T2_x1_conv1_2 = F.relu(self.conv12_1(T2_x1_conv1_1))
        x2p_T2_kernel1_max1 = F.max_pool2d(T2_x1_conv1_2, kernel_size=2, stride=2)

        T2_x1_conv2_1 = F.relu(self.conv21_1(x2p_T2_kernel1_max1))
        T2_x1_conv2_2 = F.relu(self.conv22_1(T2_x1_conv2_1))
        x2p_T2_kernel1_max2 = F.max_pool2d(T2_x1_conv2_2, kernel_size=2, stride=2)

        T2_x1_conv3_1 = F.relu(self.conv31_1(x2p_T2_kernel1_max2))
        T2_x1_conv3_2 = F.relu(self.conv32_1(T2_x1_conv3_1))
        T2_x1_conv3_3 = F.relu(self.conv33_1(T2_x1_conv3_2))
        x2p_T2_kernel1_max3 = F.max_pool2d(T2_x1_conv3_3, kernel_size=2, stride=2)

        T2_x1_conv4_1 = F.relu(self.conv41_1(x2p_T2_kernel1_max3))
        T2_x1_conv4_2 = F.relu(self.conv42_1(T2_x1_conv4_1))
        T2_x1_conv4_3 = F.relu(self.conv43_1(T2_x1_conv4_2))
        x2p_T2_kernel1_max4 = F.max_pool2d(T2_x1_conv4_3, kernel_size=2, stride=2)

        Diff_conv1_1 = F.relu(self.conv11_1(x_diff))
        Diff_conv1_2 = F.relu(self.conv12_1(Diff_conv1_1))
        Diff_kernel1_max1 = F.max_pool2d(Diff_conv1_2, kernel_size=2, stride=2)

        Diff_conv2_1 = F.relu(self.conv21_1(Diff_kernel1_max1))
        Diff_conv2_2 = F.relu(self.conv22_1(Diff_conv2_1))
        Diff_kernel1_max2 = F.max_pool2d(Diff_conv2_2, kernel_size=2, stride=2)

        Diff_conv3_1 = F.relu(self.conv31_1(Diff_kernel1_max2))
        Diff_conv3_2 = F.relu(self.conv32_1(Diff_conv3_1))
        Diff_conv3_3 = F.relu(self.conv33_1(Diff_conv3_2))
        Diff_kernel1_max3 = F.max_pool2d(Diff_conv3_3, kernel_size=2, stride=2)

        Diff_conv4_1 = F.relu(self.conv41_1(Diff_kernel1_max3))
        Diff_conv4_2 = F.relu(self.conv42_1(Diff_conv4_1))
        Diff_conv4_3 = F.relu(self.conv43_1(Diff_conv4_2))
        Diff_kernel1_max4 = F.max_pool2d(Diff_conv4_3, kernel_size=2, stride=2)

        x1_up = torch.cat([x1p_T1_kernel1_max4, x2p_T2_kernel1_max4, Diff_kernel1_max4], dim=1)

        #kernel 3
        T1_x3_conv1_1 = F.relu(self.conv11_3(x1_dn))
        T1_x3_conv1_2 = F.relu(self.conv12_3(T1_x3_conv1_1))
        x1p_T1_kernel3_max1 = F.max_pool2d(T1_x3_conv1_2, kernel_size=2, stride=2)

        T1_x3_conv2_1 = F.relu(self.conv21_3(x1p_T1_kernel3_max1))
        T1_x3_conv2_2 = F.relu(self.conv22_3(T1_x3_conv2_1))
        x1p_T1_kernel3_max2 = F.max_pool2d(T1_x3_conv2_2, kernel_size=2, stride=2)

        T1_x3_conv3_1 = F.relu(self.conv31_3(x1p_T1_kernel3_max2))
        T1_x3_conv3_2 = F.relu(self.conv32_3(T1_x3_conv3_1))
        T1_x3_conv3_3 = F.relu(self.conv33_3(T1_x3_conv3_2))
        x1p_T1_kernel3_max3 = F.max_pool2d(T1_x3_conv3_3, kernel_size=2, stride=2)

        T1_x3_conv4_1 = F.relu(self.conv41_3(x1p_T1_kernel3_max3))
        T1_x3_conv4_2 = F.relu(self.conv42_3(T1_x3_conv4_1))
        T1_x3_conv4_3 = F.relu(self.conv43_3(T1_x3_conv4_2))
        x1p_T1_kernel3_max4 = F.max_pool2d(T1_x3_conv4_3, kernel_size=2, stride=2)

        T2_x3_conv1_1 = F.relu(self.conv11_3(x2_dn))
        T2_x3_conv1_2 = F.relu(self.conv12_3(T2_x3_conv1_1))
        x2p_T2_kernel3_max1 = F.max_pool2d(T2_x3_conv1_2, kernel_size=2, stride=2)

        T2_x3_conv2_1 = F.relu(self.conv21_3(x2p_T2_kernel3_max1))
        T2_x3_conv2_2 = F.relu(self.conv22_3(T2_x3_conv2_1))
        x2p_T2_kernel3_max2 = F.max_pool2d(T2_x3_conv2_2, kernel_size=2, stride=2)

        T2_x3_conv3_1 = F.relu(self.conv31_3(x2p_T2_kernel3_max2))
        T2_x3_conv3_2 = F.relu(self.conv32_3(T2_x3_conv3_1))
        T2_x3_conv3_3 = F.relu(self.conv33_3(T2_x3_conv3_2))
        x2p_T2_kernel3_max3 = F.max_pool2d(T2_x3_conv3_3, kernel_size=2, stride=2)

        T2_x3_conv4_1 = F.relu(self.conv41_3(x2p_T2_kernel3_max3))
        T2_x3_conv4_2 = F.relu(self.conv42_3(T2_x3_conv4_1))
        T2_x3_conv4_3 = F.relu(self.conv43_3(T2_x3_conv4_2))
        x2p_T2_kernel3_max4 = F.max_pool2d(T2_x3_conv4_3, kernel_size=2, stride=2)

        DIFF_x3_conv1_1 = F.relu(self.conv11_3(x_diff))
        DIFF_x3_conv1_2 = F.relu(self.conv12_3(DIFF_x3_conv1_1))
        DIFF_kernel3_max1 = F.max_pool2d(DIFF_x3_conv1_2, kernel_size=2, stride=2)

        DIFF_x3_conv2_1 = F.relu(self.conv21_3(DIFF_kernel3_max1))
        DIFF_x3_conv2_2 = F.relu(self.conv22_3(DIFF_x3_conv2_1))
        DIFF_kernel3_max2 = F.max_pool2d(DIFF_x3_conv2_2, kernel_size=2, stride=2)

        DIFF_x3_conv3_1 = F.relu(self.conv31_3(DIFF_kernel3_max2))
        DIFF_x3_conv3_2 = F.relu(self.conv32_3(DIFF_x3_conv3_1))
        DIFF_x3_conv3_3 = F.relu(self.conv33_3(DIFF_x3_conv3_2))
        DIFF_kernel3_max3 = F.max_pool2d(DIFF_x3_conv3_3, kernel_size=2, stride=2)

        DIFF_x3_conv4_1 = F.relu(self.conv41_3(DIFF_kernel3_max3))
        DIFF_x3_conv4_2 = F.relu(self.conv42_3(DIFF_x3_conv4_1))
        DIFF_x3_conv4_3 = F.relu(self.conv43_3(DIFF_x3_conv4_2))
        DIFF_kernel3_max4 = F.max_pool2d(DIFF_x3_conv4_3, kernel_size=2, stride=2)

        x3_up = torch.cat([x1p_T1_kernel3_max4,x2p_T2_kernel3_max4,DIFF_kernel3_max4],dim=1)
        # kernel 5
        T1_x5_conv1_1 = F.relu(self.conv11_5(x1_dn))
        T1_x5_conv1_2 = F.relu(self.conv12_5(T1_x5_conv1_1))
        x1p_T1_kernel5_max1 = F.max_pool2d(T1_x5_conv1_2, kernel_size=2, stride=2)

        T1_x5_conv2_1 = F.relu(self.conv21_5(x1p_T1_kernel5_max1))
        T1_x5_conv2_2 = F.relu(self.conv22_5(T1_x5_conv2_1))
        x1p_T1_kernel5_max2 = F.max_pool2d(T1_x5_conv2_2, kernel_size=2, stride=2)

        T1_x5_conv3_1 = F.relu(self.conv31_5(x1p_T1_kernel5_max2))
        T1_x5_conv3_2 = F.relu(self.conv32_5(T1_x5_conv3_1))
        T1_x5_conv3_3 = F.relu(self.conv33_5(T1_x5_conv3_2))
        x1p_T1_kernel5_max3 = F.max_pool2d(T1_x5_conv3_3, kernel_size=2, stride=2)

        T1_x5_conv4_1 = F.relu(self.conv41_5(x1p_T1_kernel5_max3))
        T1_x5_conv4_2 = F.relu(self.conv42_5(T1_x5_conv4_1))
        T1_x5_conv4_3 = F.relu(self.conv43_5(T1_x5_conv4_2))
        x1p_T1_kernel5_max4 = F.max_pool2d(T1_x5_conv4_3, kernel_size=2, stride=2)

        T2_x5_conv1_1 = F.relu(self.conv11_5(x2_dn))
        T2_x5_conv1_2 = F.relu(self.conv12_5(T2_x5_conv1_1))
        x2p_T2_kernel5_max1 = F.max_pool2d(T2_x5_conv1_2, kernel_size=2, stride=2)

        T2_x5_conv2_1 = F.relu(self.conv21_5(x2p_T2_kernel5_max1))
        T2_x5_conv2_2 = F.relu(self.conv22_5(T2_x5_conv2_1))
        x2p_T2_kernel5_max2 = F.max_pool2d(T2_x5_conv2_2, kernel_size=2, stride=2)

        T2_x5_conv3_1 = F.relu(self.conv31_5(x2p_T2_kernel5_max2))
        T2_x5_conv3_2 = F.relu(self.conv32_5(T2_x5_conv3_1))
        T2_x5_conv3_3 = F.relu(self.conv33_5(T2_x5_conv3_2))
        x2p_T2_kernel5_max3 = F.max_pool2d(T2_x5_conv3_3, kernel_size=2, stride=2)

        T2_x5_conv4_1 = F.relu(self.conv41_5(x2p_T2_kernel5_max3))
        T2_x5_conv4_2 = F.relu(self.conv42_5(T2_x5_conv4_1))
        T2_x5_conv4_3 = F.relu(self.conv43_5(T2_x5_conv4_2))
        x2p_T2_kernel5_max4 = F.max_pool2d(T2_x5_conv4_3, kernel_size=2, stride=2)

        DIFF_x5_conv1_1 = F.relu(self.conv11_5(x_diff))
        DIFF_x5_conv1_2 = F.relu(self.conv12_5(DIFF_x5_conv1_1))
        DIFF_kernel5_max1 = F.max_pool2d(DIFF_x5_conv1_2, kernel_size=2, stride=2)

        DIFF_x5_conv2_1 = F.relu(self.conv21_5(DIFF_kernel5_max1))
        DIFF_x5_conv2_2 = F.relu(self.conv22_5(DIFF_x5_conv2_1))
        DIFF_kernel5_max2 = F.max_pool2d(DIFF_x5_conv2_2, kernel_size=2, stride=2)

        DIFF_x5_conv3_1 = F.relu(self.conv31_5(DIFF_kernel5_max2))
        DIFF_x5_conv3_2 = F.relu(self.conv32_5(DIFF_x5_conv3_1))
        DIFF_x5_conv3_3 = F.relu(self.conv33_5(DIFF_x5_conv3_2))
        DIFF_kernel5_max3 = F.max_pool2d(DIFF_x5_conv3_3, kernel_size=2, stride=2)

        DIFF_x5_conv4_1 = F.relu(self.conv41_5(DIFF_kernel5_max3))
        DIFF_x5_conv4_2 = F.relu(self.conv42_5(DIFF_x5_conv4_1))
        DIFF_x5_conv4_3 = F.relu(self.conv43_5(DIFF_x5_conv4_2))
        DIFF_kernel5_max4 = F.max_pool2d(DIFF_x5_conv4_3, kernel_size=2, stride=2)

        x5_up = torch.cat([x1p_T1_kernel5_max4,x2p_T2_kernel5_max4,DIFF_kernel5_max4],dim=1)

        #kernel 1 up
        T1_x4d_kernel1 = self.upconv4(x1_up)
        pad4 = ReplicationPad2d(
            (0, T1_x1_conv4_3.size(3) - T1_x4d_kernel1.size(3), 0, T1_x1_conv4_3.size(2) - T1_x4d_kernel1.size(2)))
        T1_x4d_kernel1 = torch.cat((pad4(T1_x4d_kernel1), T1_x1_conv4_3, T2_x1_conv4_3, Diff_conv4_3), 1)
        T1_x43d_kernel1 = F.relu(self.conv43d(T1_x4d_kernel1))
        T1_x42d_kernel1 = F.relu(self.conv42d(T1_x43d_kernel1))
        T1_x41d_kernel1 = F.relu(self.conv41d(T1_x42d_kernel1))

        # Stage 3d
        T1_x3d_kernel1 = self.upconv3(T1_x41d_kernel1)
        pad3 = ReplicationPad2d(
            (0, T1_x1_conv3_3.size(3) - T1_x3d_kernel1.size(3), 0, T1_x1_conv3_3.size(2) - T1_x3d_kernel1.size(2)))
        T1_x3d_kernel1 = torch.cat((pad3(T1_x3d_kernel1), T1_x1_conv3_3, T2_x1_conv3_3, Diff_conv3_3), 1)
        T1_x33d_kernel1 = F.relu(self.conv33d(T1_x3d_kernel1))
        T1_x32d_kernel1 = F.relu(self.conv32d(T1_x33d_kernel1))
        T1_x31d_kernel1 = F.relu(self.conv31d(T1_x32d_kernel1))

        # Stage 2d
        T1_x2d_kernel1 = self.upconv2(T1_x31d_kernel1)
        pad2 = ReplicationPad2d(
            (0, T1_x1_conv2_2.size(3) - T1_x2d_kernel1.size(3), 0, T1_x1_conv2_2.size(2) - T1_x2d_kernel1.size(2)))
        T1_x2d_kernel1 = torch.cat((pad2(T1_x2d_kernel1), T1_x1_conv2_2, T2_x1_conv2_2, Diff_conv2_2), 1)
        T1_x22d_kernel1 = F.relu(self.conv22d(T1_x2d_kernel1))
        T1_x21d_kernel1 = F.relu(self.conv21d(T1_x22d_kernel1))

        # Stage 1d
        T1_x1d_kernel1 = self.upconv1(T1_x21d_kernel1)
        pad1 = ReplicationPad2d(
            (0, T1_x1_conv1_2.size(3) - T1_x1d_kernel1.size(3), 0, T1_x1_conv1_2.size(2) - T1_x1d_kernel1.size(2)))
        T1_x11d_kernel1 = torch.cat((pad1(T1_x1_conv1_2), T1_x1_conv1_2, T2_x1_conv1_2, Diff_conv1_2), 1)
        T1_x12d = F.relu(self.conv12d(T1_x11d_kernel1))
        feature1_map = self.conv11d(T1_x12d)

        #kernel 3 up
        T1_x4d_kernel3 = self.upconv4(x3_up)
        pad4 = ReplicationPad2d(
            (0, T1_x3_conv4_3.size(3) - T1_x4d_kernel3.size(3), 0, T1_x3_conv4_3.size(2) - T1_x4d_kernel3.size(2)))
        T1_x4d_kernel3 = torch.cat((pad4(T1_x4d_kernel3), T1_x3_conv4_3, T2_x3_conv4_3, DIFF_x3_conv4_3), 1)
        T1_x43d_kernel3 = F.relu(self.conv43d(T1_x4d_kernel3))
        T1_x42d_kernel3 = F.relu(self.conv42d(T1_x43d_kernel3))
        T1_x41d_kernel3 = F.relu(self.conv41d(T1_x42d_kernel3))

        # Stage 3d
        T1_x3d_kernel3 = self.upconv3(T1_x41d_kernel3)
        pad3 = ReplicationPad2d(
            (0, T1_x3_conv3_3.size(3) - T1_x3d_kernel3.size(3), 0, T1_x3_conv3_3.size(2) - T1_x3d_kernel3.size(2)))
        T1_x3d_kernel3 = torch.cat(
            (pad3(T1_x3d_kernel3), T1_x3_conv3_3, T2_x3_conv3_3, DIFF_x3_conv3_3), 1)
        T1_x33d_kernel3 = F.relu(self.conv33d(T1_x3d_kernel3))
        T1_x32d_kernel3 = F.relu(self.conv32d(T1_x33d_kernel3))
        T1_x31d_kernel3 = F.relu(self.conv31d(T1_x32d_kernel3))

        # Stage 2d
        T1_x2d_kernel3 = self.upconv2(T1_x31d_kernel3)
        pad2 = ReplicationPad2d(
            (0, T1_x3_conv2_2.size(3) - T1_x2d_kernel3.size(3), 0, T1_x3_conv2_2.size(2) - T1_x2d_kernel3.size(2)))
        T1_x2d_kernel3 = torch.cat((pad2(T1_x2d_kernel3), T1_x3_conv2_2, T2_x3_conv2_2, DIFF_x3_conv2_2), 1)
        T1_x22d_kernel3 = F.relu(self.conv22d(T1_x2d_kernel3))
        T1_x21d_kernel3 = F.relu(self.conv21d(T1_x22d_kernel3))

        # Stage 1d
        T1_x1d_kernel3 = self.upconv1(T1_x21d_kernel3)
        pad1 = ReplicationPad2d(
            (0, T1_x3_conv1_2.size(3) - T1_x1d_kernel3.size(3), 0, T1_x3_conv1_2.size(2) - T1_x1d_kernel3.size(2)))
        T1_x11d_kernel3 = torch.cat((pad1(T1_x3_conv1_2), T1_x3_conv1_2, T2_x3_conv1_2, DIFF_x3_conv1_2), 1)
        T1_x12d = F.relu(self.conv12d(T1_x11d_kernel3))
        feature2_map = self.conv11d(T1_x12d)


        #kernel 5 up
        T1_x4d_kernel5 = self.upconv4(x5_up) #torch.cat([x1p_T1_kernel5_max4,x2p_T2_kernel5_max4,DIFF_kernel5_max4],dim=1)
        pad4 = ReplicationPad2d(
            (0, T1_x5_conv4_3.size(3) - T1_x4d_kernel5.size(3), 0, T1_x5_conv4_3.size(2) - T1_x4d_kernel5.size(2)))
        T1_x4d_kernel5 = torch.cat((pad4(T1_x4d_kernel5), T1_x5_conv4_3, T2_x5_conv4_3, DIFF_x5_conv4_3), 1)
        T1_x43d_kernel5 = F.relu(self.conv43d(T1_x4d_kernel5))
        T1_x42d_kernel5 = F.relu(self.conv42d(T1_x43d_kernel5))
        T1_x41d_kernel5 = F.relu(self.conv41d(T1_x42d_kernel5))


        T1_x3d_kernel5 = self.upconv3(T1_x41d_kernel5)
        pad3 = ReplicationPad2d(
            (0, T1_x5_conv3_3.size(3) - T1_x3d_kernel5.size(3), 0, T1_x5_conv3_3.size(2) - T1_x3d_kernel5.size(2)))
        T1_x3d_kernel5 = torch.cat((pad3(T1_x3d_kernel5), T1_x5_conv3_3, T2_x5_conv3_3, DIFF_x5_conv3_3), 1)
        T1_x33d_kernel5 = F.relu(self.conv33d(T1_x3d_kernel5))
        T1_x32d_kernel5 = F.relu(self.conv32d(T1_x33d_kernel5))
        T1_x31d_kernel5 = F.relu(self.conv31d(T1_x32d_kernel5))


        T1_x2d_kernel5 = self.upconv2(T1_x31d_kernel5)
        pad2 = ReplicationPad2d(
            (0, T1_x5_conv2_2.size(3) - T1_x2d_kernel5.size(3), 0, T1_x5_conv2_2.size(2) - T1_x2d_kernel5.size(2)))
        T1_x2d_kernel5 = torch.cat((pad2(T1_x2d_kernel5), T1_x5_conv2_2, T2_x5_conv2_2, DIFF_x5_conv2_2), 1)
        T1_x22d_kernel5 = F.relu(self.conv22d(T1_x2d_kernel5))
        T1_x21d_kernel5 = F.relu(self.conv21d(T1_x22d_kernel5))

        T1_x1d_kernel5 = self.upconv1(T1_x21d_kernel5)
        pad1 = ReplicationPad2d(
            (0, T1_x5_conv1_2.size(3) - T1_x1d_kernel5.size(3), 0, T1_x5_conv1_2.size(2) - T1_x1d_kernel5.size(2)))
        T1_x11d_kernel5 = torch.cat((pad1(T1_x5_conv1_2), T1_x5_conv1_2, T2_x5_conv1_2, DIFF_x5_conv1_2), 1)
        T1_x12d = F.relu(self.conv12d(T1_x11d_kernel5))
        feature3_map = self.conv11d(T1_x12d)


        # kernel1_fusion_map = F.relu(self.bn1(self.conv1(feature1_map)))
        # kernel2_fusion_map = F.relu(self.bn2(self.conv2(feature2_map)))
        # kernel3_fusion_map = F.relu(self.bn3(self.conv3(feature3_map)))

        feature_1_2_fusion = torch.cat([feature1_map,feature2_map],dim=1)
        feature_2_3_fusion = torch.cat([feature2_map,feature3_map],dim=1)
        feature_1_3_fusion = torch.cat([feature1_map, feature3_map], dim=1)

        feature1_2 = F.relu(self.bn1(self.conv1(feature_1_2_fusion)))
        feature2_3 = F.relu(self.bn2(self.conv2(feature_2_3_fusion)))
        feature1_3 = F.relu(self.bn3(self.conv3(feature_1_3_fusion)))

        fusion_map1 = F.relu(self.bn4(self.conv4(torch.cat([feature1_2,feature2_3],dim=1))))
        fusion_map2 = F.relu(self.bn4(self.conv4(torch.cat([feature2_3,feature1_3],dim=1))))
        fusion_map = torch.cat([fusion_map1,fusion_map2],dim=1)
        feature_map = F.relu(self.bn4(self.conv4(fusion_map)))
        return feature_map

if __name__=="__main__":
    
    net = HFTNet()
    x1 = torch.randn(1,3,64,64)
    x2 = torch.randn(1,3,64,64)
    x3 = torch.randn(1,3,64,64)
    img = net(x1,x2)

    
    
    
    
    
