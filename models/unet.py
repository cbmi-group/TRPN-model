
import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

currentdir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(currentdir)

from utils.base_module import DoubleConv, Down, Up, OutConv, SELayer, ConvLayer
from utils.utils import init_weights
from .resnet import BasicBlock, conv1x1


class UNet(nn.Module):
    def __init__(self, input_channels=1, base_dim=64, n_classes=1, bilinear=False, logits=True):
        super(UNet, self).__init__()
        self.n_channels = input_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.fuse_mode = 'cat'
        self.logits = logits

        # encoder
        self.inc = DoubleConv(input_channels, base_dim)
        self.down1 = Down(base_dim, base_dim*2)
        self.down2 = Down(base_dim*2, base_dim*4)
        self.down3 = Down(base_dim*4, base_dim*8)
        self.down4 = Down(base_dim*8, base_dim*16)

        # decoder
        self.up1 = Up(base_dim*16, base_dim*8, bilinear, fuse_mode=self.fuse_mode)
        self.up2 = Up(base_dim*8,  base_dim*4, bilinear, fuse_mode=self.fuse_mode)
        self.up3 = Up(base_dim*4,  base_dim*2, bilinear, fuse_mode=self.fuse_mode)
        self.up4 = Up(base_dim*2,  base_dim,   bilinear, fuse_mode=self.fuse_mode)
        self.out = OutConv(base_dim, n_classes)


    def forward(self, x):
        # encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        o_4 = self.up1(x5, x4)
        o_3 = self.up2(o_4, x3)
        o_2 = self.up3(o_3, x2)
        o_1 = self.up4(o_2, x1)
        o_seg = self.out(o_1)

        if self.logits:
            pred = torch.sigmoid(o_seg)
        else:
            pred = o_seg

        return pred


class UNet_Multi_Output(UNet):
    def __init__(self, input_channels=1, base_dim=64, n_classes=1, bilinear=False):
        super(UNet_Multi_Output, self).__init__(input_channels=input_channels, base_dim=base_dim, n_classes=n_classes)
        self.out_down = OutConv(base_dim, n_classes)
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        o_4 = self.up1(x5, x4)
        o_3 = self.up2(o_4, x3)
        o_2 = self.up3(o_3, x2)
        o_1 = self.up4(o_2, x1)
        o_seg = self.out(o_1)
        o_seg_down = self.out_down(o_1)

        pred_up = torch.sigmoid(o_seg)
        pred_down = torch.sigmoid(o_seg_down)

        return pred_up, pred_down


class UNet3(nn.Module):
    def __init__(self, input_channels=1, base_dim=64, n_classes=1, bilinear=False, logits=True, is_res_loss=False):
        super(UNet3, self).__init__()
        self.n_channels = input_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.fuse_mode = 'cat'
        self.logits = logits
        self.is_res_loss = is_res_loss

        # encoder
        self.inc = DoubleConv(input_channels, base_dim)
        self.down1 = Down(base_dim, base_dim*2)
        self.down2 = Down(base_dim*2, base_dim*4)
        self.down3 = Down(base_dim*4, base_dim*8)

        # decoder
        self.up1 = Up(base_dim*8,  base_dim*4, bilinear, fuse_mode=self.fuse_mode)
        self.up2 = Up(base_dim*4,  base_dim*2, bilinear, fuse_mode=self.fuse_mode)
        self.up3 = Up(base_dim*2,  base_dim,   bilinear, fuse_mode=self.fuse_mode)
        self.out = OutConv(base_dim, n_classes)

        if is_res_loss:
            self.out1 = OutConv(base_dim, n_classes)

    def forward(self, x):
        # encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        o_3 = self.up1(x4, x3)
        o_2 = self.up2(o_3, x2)
        o_1 = self.up3(o_2, x1)
        o_seg = self.out(o_1)

        if self.logits:
            if self.n_classes > 1:
                pred = torch.softmax(o_seg, dim=1)
            else:
                pred = torch.sigmoid(o_seg)
        else:
            pred = o_seg

        if self.is_res_loss:
            o_seg_1 = self.out1(o_1)
            if self.n_classes > 1:
                pred1 = torch.softmax(o_seg_1, dim=1)
            else:
                pred1 = torch.sigmoid(o_seg_1)
            return pred, pred1
        else: 
            return pred

class UNet2(nn.Module):
    def __init__(self, input_channels=1, base_dim=64, n_classes=1, bilinear=False, logits=True, is_res_loss=False):
        super(UNet2, self).__init__()
        self.n_channels = input_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.fuse_mode = 'cat'
        self.logits = logits
        self.is_res_loss = is_res_loss

        # encoder
        self.inc = DoubleConv(input_channels, base_dim)
        self.down1 = Down(base_dim, base_dim*2)
        self.down2 = Down(base_dim*2, base_dim*4)

        # decoder
        self.up1 = Up(base_dim*4,  base_dim*2, bilinear, fuse_mode=self.fuse_mode)
        self.up2 = Up(base_dim*2,  base_dim,   bilinear, fuse_mode=self.fuse_mode)
        self.out = OutConv(base_dim, n_classes)

        if is_res_loss:
            self.out1 = OutConv(base_dim, n_classes)

    def forward(self, x):
        # encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        o_2 = self.up1(x3, x2)
        o_1 = self.up2(o_2, x1)
        o_seg = self.out(o_1)

        if self.logits:
            if self.n_classes > 1:
                pred = torch.softmax(o_seg, dim=1)
            else:
                pred = torch.sigmoid(o_seg)
        else:
            pred = o_seg

        if self.is_res_loss:
            o_seg_1 = self.out1(o_1)
            if self.n_classes > 1:
                pred1 = torch.softmax(o_seg_1, dim=1)
            else:
                pred1 = torch.sigmoid(o_seg_1)
            return pred, pred1
        else: 
            return pred


class UNet1(nn.Module):
    def __init__(self, input_channels=1, base_dim=64, n_classes=1, bilinear=False, logits=True, is_res_loss=False):
        super(UNet1, self).__init__()
        self.n_channels = input_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.fuse_mode = 'cat'
        self.logits = logits
        self.is_res_loss = is_res_loss

        # encoder
        self.inc = DoubleConv(input_channels, base_dim)
        self.down1 = Down(base_dim, base_dim*2)

        # decoder
        self.up1 = Up(base_dim*2,  base_dim,   bilinear, fuse_mode=self.fuse_mode)
        self.out = OutConv(base_dim, n_classes)

        if is_res_loss:
            self.out1 = OutConv(base_dim, n_classes)

    def forward(self, x):
        # encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)

        o_1 = self.up1(x2, x1)
        o_seg = self.out(o_1)

        if self.logits:
            if self.n_classes > 1:
                pred = torch.softmax(o_seg, dim=1)
            else:
                pred = torch.sigmoid(o_seg)
        else:
            pred = o_seg

        if self.is_res_loss:
            o_seg_1 = self.out1(o_1)
            if self.n_classes > 1:
                pred1 = torch.softmax(o_seg_1, dim=1)
            else:
                pred1 = torch.sigmoid(o_seg_1)
            return pred, pred1
        else: 
            return pred



class UNet_Resnet(nn.Module):
    def __init__(self, input_channels=1, base_dim=32, n_classes=1, layers=[1,1,1,2], 
                       bilinear=False, block=BasicBlock,
                       logits=True, is_res_loss=False):
        super(UNet_Resnet, self).__init__()
        
        self.n_channels = input_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inplanes = base_dim # initial feature dim is 64
        self.logits = logits
        self.is_res_loss = is_res_loss

        # input layer
        self.in_conv = self._DoubleConv(input_channels, base_dim)

        # encoder
        self.encoder_layer1 = self._make_layer(block, base_dim*2, layers[0], stride=2) # 128
        self.encoder_layer2 = self._make_layer(block, base_dim*4, layers[1], stride=2) # 256
        self.encoder_layer3 = self._make_layer(block, base_dim*8, layers[2], stride=2) # 512
        self.encoder_layer4 = self._make_layer(block, base_dim*16, layers[3], stride=2) # 1024
        

        # decoder
        self.up_layer4 = self._Up(base_dim*16, base_dim*8, bilinear=bilinear)
        self.inplanes = base_dim * 16
        self.decoder_layer4 = self._make_layer(block, base_dim*8, layers[3], stride=1)
        
        self.up_layer3 = self._Up(base_dim*8, base_dim*4, bilinear=bilinear)
        self.inplanes = base_dim * 8
        self.decoder_layer3 = self._make_layer(block, base_dim*4, layers[2], stride=1)

        self.up_layer2 = self._Up(base_dim*4, base_dim*2, bilinear=bilinear)
        self.inplanes = base_dim * 4
        self.decoder_layer2 = self._make_layer(block, base_dim*2, layers[1], stride=1)

        self.up_layer1 = self._Up(base_dim*2, base_dim, bilinear=bilinear)
        self.inplanes = base_dim * 2
        self.decoder_layer1 = self._make_layer(block, base_dim, layers[0], stride=1)
        
        self.out = OutConv(base_dim, n_classes)

        if is_res_loss:
            self.out1 = OutConv(base_dim, n_classes)

    def _DoubleConv(self, in_channels, out_channels):
        double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), 
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=True))
        return double_conv
    
    def _Up(self, in_channels, out_channels, bilinear=False):
        if bilinear:
            up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=(1,1), stride=1)
            )
        else:
            up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        return up
    
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # encoder
        x1 = self.in_conv(x)
        x2 = self.encoder_layer1(x1)
        x3 = self.encoder_layer2(x2)
        x4 = self.encoder_layer3(x3)

        # decoder
        up_3 = self.up_layer3(x4)
        y3 = self.decoder_layer3(torch.cat((x3, up_3), dim=1))
        
        up_2 = self.up_layer2(y3)
        y2 = self.decoder_layer2(torch.cat((x2, up_2), dim=1))
        
        up_1 = self.up_layer1(y2)
        y1 = self.decoder_layer1(torch.cat((x1, up_1), dim=1))
        
        o_seg = self.out(y1)

        if self.logits:
            if self.n_classes > 1:
                pred = torch.softmax(o_seg, dim=1)
            else:
                pred = torch.sigmoid(o_seg)
        else:
            pred = o_seg

        if self.is_res_loss:
            o_seg_1 = self.out1(y1)
            if self.n_classes > 1:
                pred1 = torch.softmax(o_seg_1, dim=1)
            else:
                pred1 = torch.sigmoid(o_seg_1)
            return pred, pred1
        else: 
            return pred


