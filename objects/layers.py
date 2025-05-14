import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from time import time



def get_layer_dimensions(latent_dim):
    l3 = latent_dim//2
    l2 = l3//2
    l1 = l2//2
    return l1, l2, l3

# Layers

class MaskedConv2d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=5, mask_size=3):
        super().__init__()
        filter_mask = torch.ones(kernel_size,kernel_size)
        for i in range((kernel_size-mask_size)//2,(kernel_size+mask_size)//2):
            for j in range((kernel_size-mask_size)//2,(kernel_size+mask_size)//2):
                filter_mask[i,j] = 0
        self.register_buffer('filter_mask', filter_mask)
        self.conv = nn.Conv2d(in_dim,out_dim,kernel_size=kernel_size,padding='same')

    def forward(self, x):
        self._mask_conv_filter()
        return self.conv(x)
    def _mask_conv_filter(self):
        with torch.no_grad():
            self.conv.weight = nn.Parameter(self.conv.weight * self.filter_mask)


class ConvBlock2d(nn.Module):
    def __init__(self,in_dim,out_dim,kernels=[3,5,7],dropout=.0,decode=False,last_channel=False):
        super().__init__()
        k_in, k_out = (4, 1) if decode == True else (1, 4)
        self.last_channel = last_channel
        self.convs = nn.ModuleList([nn.Conv2d(in_dim,k_in*out_dim,kernel_size=kernel,padding='same') for kernel in kernels])
        self.batch_norm_1 = nn.BatchNorm2d(len(kernels)*k_in*out_dim)
        self.relu_1 = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.resize = Rearrange('b (c p1 p2) h w -> b c (h p1) (w p2)',p1=2,p2=2) if decode == True else Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w',p1=2,p2=2)
        self.conv_last = nn.Conv2d(len(kernels)*k_out*out_dim,out_dim,kernel_size=1,padding='same')
        self.batch_norm_2 = nn.BatchNorm2d(out_dim)
        self.relu_2 = nn.ReLU()
        
    
    def forward(self,x):
        x = [conv(x) for conv in self.convs]
        x = torch.cat(x,dim=1)
        x = self.batch_norm_1(x)
        x = self.relu_1(x)
        x = self.drop(x)
        x = self.resize(x) 
        x = self.conv_last(x)
        if self.last_channel == False:
            x = self.batch_norm_2(x)
            x = self.relu_2(x)
        return x

class EncoderBlock2d(nn.Module):
    def __init__(self,latent_dim=64,channels=5,kernels=[3,5,7]):
        super().__init__()
        l1, l2, l3 = get_layer_dimensions(latent_dim)
        self.conv_1 = ConvBlock2d(channels,l1,kernels)
        self.conv_2 = ConvBlock2d(l1,l2,kernels)
        self.conv_3 = ConvBlock2d(l2,l3,kernels)
        self.conv_4 = ConvBlock2d(l3,latent_dim,kernels)


    def forward(self,x):
       x1 = self.conv_1(x)
       x2 = self.conv_2(x1)
       x3 = self.conv_3(x2)
       x4 = self.conv_4(x3)
       return x1, x2, x3, x4

class DecoderBlock2d(nn.Module):
    def __init__(self,latent_dim=64,channels=5,kernels=[3,5,7]):
        super().__init__()
        l1, l2, l3 = get_layer_dimensions(latent_dim)
        self.conv_1 = ConvBlock2d(latent_dim,l3,kernels,decode=True)
        self.conv_2 = ConvBlock2d(l3,l2,kernels,decode=True)
        self.conv_3 = ConvBlock2d(l2,l1,kernels,decode=True)
        self.conv_4 = ConvBlock2d(l1,channels,kernels,decode=True,last_channel=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):

        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x) 
        return self.sigmoid(x)

class SkipDecoderBlock2d(nn.Module):
    def __init__(self,latent_dim=64,channels=5,kernels=[3,5,7]):
        super().__init__()
        l1, l2, l3 = get_layer_dimensions(latent_dim)
        self.conv_1 = ConvBlock2d(latent_dim,l3,kernels,decode=True)
        self.conv_2 = ConvBlock2d(2*l3,l2,kernels,decode=True)
        self.conv_3 = ConvBlock2d(2*l2,l1,kernels,decode=True)
        self.conv_4 = ConvBlock2d(2*l1,channels,kernels,decode=True,last_channel=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x1, x2, x3, x4):

        x = self.conv_1(x4)
        x = torch.cat([x,x3],dim=1)
        x = self.conv_2(x)
        x = torch.cat([x,x2],dim=1)
        x = self.conv_3(x)
        x = torch.cat([x,x1],dim=1)
        x = self.conv_4(x) 
        return self.sigmoid(x)


class MaskedConvBlock2d(nn.Module):
    def __init__(self,in_dim,out_dim,kernels=[3,5,7],dropout=.0,decode=False,last_channel=False):
        super().__init__()
        k_in, k_out = (4, 1) if decode == True else (1, 4)
        self.last_channel = last_channel
        self.convs = nn.ModuleList([MaskedConv2d(in_dim,k_in*out_dim,kernel_size=kernel,mask_size=kernel-2) for kernel in kernels])
        self.batch_norm_1 = nn.BatchNorm2d(len(kernels)*k_in*out_dim)
        self.relu_1 = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.resize = Rearrange('b (c p1 p2) h w -> b c (h p1) (w p2)',p1=2,p2=2) if decode == True else Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w',p1=2,p2=2)
        self.conv_last = nn.Conv2d(len(kernels)*k_out*out_dim,out_dim,kernel_size=1,padding='same')
        self.batch_norm_2 = nn.BatchNorm2d(out_dim)
        self.relu_2 = nn.ReLU()
        
    
    def forward(self,x):
        x = [conv(x) for conv in self.convs]
        x = torch.cat(x,dim=1)
        x = self.batch_norm_1(x)
        x = self.relu_1(x)
        x = self.drop(x)
        x = self.resize(x) 
        x = self.conv_last(x)
        if self.last_channel == False:
            x = self.batch_norm_2(x)
            x = self.relu_2(x)
        return x

class MaskedEncoderBlock2d(nn.Module):
    def __init__(self,latent_dim=64,channels=5,kernels=[3,5,7]):
        super().__init__()
        l1, l2, l3 = get_layer_dimensions(latent_dim)
        self.conv_1 = MaskedConvBlock2d(channels,l1,kernels)
        self.conv_2 = MaskedConvBlock2d(l1,l2,kernels)
        self.conv_3 = MaskedConvBlock2d(l2,l3,kernels)
        self.conv_4 = MaskedConvBlock2d(l3,latent_dim,kernels)


    def forward(self,x):
       x1 = self.conv_1(x)
       x2 = self.conv_2(x1)
       x3 = self.conv_3(x2)
       x4 = self.conv_4(x3)
       return x1, x2, x3, x4

class MaskedDecoderBlock2d(nn.Module):
    def __init__(self,latent_dim=64,channels=5,kernels=[3,5,7]):
        super().__init__()
        l1, l2, l3 = get_layer_dimensions(latent_dim)
        self.conv_1 = MaskedConvBlock2d(latent_dim,l3,kernels,decode=True)
        self.conv_2 = MaskedConvBlock2d(l3,l2,kernels,decode=True)
        self.conv_3 = MaskedConvBlock2d(l2,l1,kernels,decode=True)
        self.conv_4 = MaskedConvBlock2d(l1,channels,kernels,decode=True,last_channel=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):

        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x) 
        return self.sigmoid(x)

class MaskedSkipDecoderBlock2d(nn.Module):
    def __init__(self,latent_dim=64,channels=5,kernels=[3,5,7]):
        super().__init__()
        l1, l2, l3 = get_layer_dimensions(latent_dim)
        self.conv_1 = MaskedConvBlock2d(latent_dim,l3,kernels,decode=True)
        self.conv_2 = MaskedConvBlock2d(2*l3,l2,kernels,decode=True)
        self.conv_3 = MaskedConvBlock2d(2*l2,l1,kernels,decode=True)
        self.conv_4 = MaskedConvBlock2d(2*l1,channels,kernels,decode=True,last_channel=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x1, x2, x3, x4):

        x = self.conv_1(x4)
        x = torch.cat([x,x3],dim=1)
        x = self.conv_2(x)
        x = torch.cat([x,x2],dim=1)
        x = self.conv_3(x)
        x = torch.cat([x,x1],dim=1)
        x = self.conv_4(x) 
        return self.sigmoid(x)


class MaskedConv3d(nn.Module):
    def __init__(self, in_dim, out_dim, temporal=3, kernel_size=5, mask_size=3):
        super().__init__()
        filter_mask = torch.ones(temporal,kernel_size,kernel_size)
        t = (temporal-1)//2
        for i in range((kernel_size-mask_size)//2,(kernel_size+mask_size)//2):
            for j in range((kernel_size-mask_size)//2,(kernel_size+mask_size)//2):
                filter_mask[t,i,j] = 0
        self.register_buffer('filter_mask', filter_mask)
        self.conv = nn.Conv3d(in_dim,out_dim,kernel_size=(temporal,kernel_size,kernel_size),padding='same')

    def forward(self, x):
        self._mask_conv_filter()
        return self.conv(x)

    def _mask_conv_filter(self):
        with torch.no_grad():
            self.conv.weight = nn.Parameter(self.conv.weight * self.filter_mask)

class ConvBlock3d(nn.Module):
    def __init__(self,in_dim,out_dim,temporal=3,kernels=[3,5,7],dropout=.0,decode=False,last_channel=False):
        super().__init__()
        k_in, k_out = (4, 1) if decode == True else (1, 4)
        self.last_channel = last_channel
        self.convs = nn.ModuleList([nn.Conv3d(in_dim,k_in*out_dim,kernel_size=(temporal,kernel,kernel),padding='same') for kernel in kernels])
        self.batch_norm_1 = nn.BatchNorm3d(len(kernels)*k_in*out_dim)
        self.relu_1 = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.resize = Rearrange('b (c p1 p2) t h w -> b c t (h p1) (w p2)',p1=2,p2=2) if decode == True else Rearrange('b c t (h p1) (w p2) -> b (c p1 p2) t h w',p1=2,p2=2)
        self.conv_last = nn.Conv3d(len(kernels)*k_out*out_dim,out_dim,kernel_size=1,padding='same')
        self.batch_norm_2 = nn.BatchNorm3d(out_dim)
        self.relu_2 = nn.ReLU()
        
    
    def forward(self,x):
        x = [conv(x) for conv in self.convs]
        x = torch.cat(x,dim=1)
        x = self.batch_norm_1(x)
        x = self.relu_1(x)
        x = self.drop(x)
        x = self.resize(x) 
        x = self.conv_last(x)
        if self.last_channel == False:
            x = self.batch_norm_2(x)
            x = self.relu_2(x)
        return x

class EncoderBlock3d(nn.Module):
    def __init__(self,latent_dim=64,channels=5,temporal=3,kernels=[3,5,7]):
        super().__init__()
        l1, l2, l3 = get_layer_dimensions(latent_dim)
        self.conv_1 = ConvBlock3d(channels,l1,temporal,kernels)
        self.conv_2 = ConvBlock3d(l1,l2,temporal,kernels)
        self.conv_3 = ConvBlock3d(l2,l3,temporal,kernels)
        self.conv_4 = ConvBlock3d(l3,latent_dim,temporal,kernels)


    def forward(self,x):
       x1 = self.conv_1(x)
       x2 = self.conv_2(x1)
       x3 = self.conv_3(x2)
       x4 = self.conv_4(x3)
       return x1, x2, x3, x4

class DecoderBlock3d(nn.Module):
    def __init__(self,latent_dim=64,channels=5,temporal=3,kernels=[3,5,7]):
        super().__init__()
        l1, l2, l3 = get_layer_dimensions(latent_dim)
        self.conv_1 = ConvBlock3d(latent_dim,l3,temporal,kernels,decode=True)
        self.conv_2 = ConvBlock3d(l3,l2,temporal,kernels,decode=True)
        self.conv_3 = ConvBlock3d(l2,l1,temporal,kernels,decode=True)
        self.conv_4 = ConvBlock3d(l1,channels,temporal,kernels,decode=True,last_channel=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):

        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x) 
        return self.sigmoid(x)

class SkipDecoderBlock3d(nn.Module):
    def __init__(self,latent_dim=64,channels=5,temporal=3,kernels=[3,5,7]):
        super().__init__()
        l1, l2, l3 = get_layer_dimensions(latent_dim)
        self.conv_1 = ConvBlock3d(latent_dim,l3,temporal,kernels,decode=True)
        self.conv_2 = ConvBlock3d(2*l3,l2,temporal,kernels,decode=True)
        self.conv_3 = ConvBlock3d(2*l2,l1,temporal,kernels,decode=True)
        self.conv_4 = ConvBlock3d(2*l1,channels,temporal,kernels,decode=True,last_channel=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x1, x2, x3, x4):

        x = self.conv_1(x4)
        x = torch.cat([x,x3],dim=1)
        x = self.conv_2(x)
        x = torch.cat([x,x2],dim=1)
        x = self.conv_3(x)
        x = torch.cat([x,x1],dim=1)
        x = self.conv_4(x) 
        return self.sigmoid(x)
