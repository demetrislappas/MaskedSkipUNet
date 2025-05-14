import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

# Encoders
from layers import EncoderBlock2d, EncoderBlock3d, MaskedEncoderBlock2d

# Decoders
from layers import DecoderBlock2d, SkipDecoderBlock2d, SkipDecoderBlock3d, MaskedSkipDecoderBlock2d

# Masked Convolutions
from layers import MaskedConv2d, MaskedConv3d

# import layer_dimesions function
from layers import get_layer_dimensions

# Models

class AutoEncoder(nn.Module):
    def __init__(self,latent_dim=64,channels=3,temporal=1):
        super().__init__()
        assert temporal==1, 'Temporal must be 1 for this model'
        self.encoder = EncoderBlock2d(latent_dim=latent_dim,channels=channels)
        self.decoder = DecoderBlock2d(latent_dim=latent_dim,channels=channels)

    def forward(self,x):
        if len(x.shape) == 5:
            x = x.squeeze(dim=2)
        _, _, _, x4 = self.encoder(x)
        x = self.decoder(x4)
        return x
    
class UNet(nn.Module):
    def __init__(self,latent_dim=64,channels=3,temporal=1):
        super().__init__()
        assert temporal==1, 'Temporal must be 1 for this model'
        self.encoder = EncoderBlock2d(latent_dim=latent_dim,channels=channels)
        self.decoder = SkipDecoderBlock2d(latent_dim=latent_dim,channels=channels)

    def forward(self,x):
        if len(x.shape) == 5:
            x = x.squeeze(dim=2)
        x1, x2, x3, x4 = self.encoder(x)
        x = self.decoder(x1, x2, x3, x4)
        return x

class Conv3dSkipUNet(nn.Module):
    def __init__(self,latent_dim=64,channels=3,temporal=3):
        super().__init__()
        l1, l2, l3 = get_layer_dimensions(latent_dim)
        self.target_frame = (temporal-1)//2
        self.encoder = EncoderBlock2d(latent_dim=latent_dim,channels=channels)
        self.decoder = SkipDecoderBlock2d(latent_dim=latent_dim,channels=channels)

        self.to_2d = Rearrange('b c t h w -> (b t) c h w')
        self.to_3d = Rearrange('(b t) c h w -> b c t h w',t=temporal)

        kernel_sizes = [17,11,7,3]
        self.conv1 = nn.Conv3d(l1,l1,kernel_size=(temporal,kernel_sizes[0],kernel_sizes[0]),padding='same')
        self.conv2 = nn.Conv3d(l2,l2,kernel_size=(temporal,kernel_sizes[1],kernel_sizes[1]),padding='same')
        self.conv3 = nn.Conv3d(l3,l3,kernel_size=(temporal,kernel_sizes[2],kernel_sizes[2]),padding='same')
        self.conv4 = nn.Conv3d(latent_dim,latent_dim,kernel_size=(temporal,kernel_sizes[3],kernel_sizes[3]),padding='same')

        
    def forward(self,x):
        
        # Stack temporal onto batch
        x = self.to_2d(x)

        # Pass through 2D encoder
        x1, x2, x3, x4 = self.encoder(x)

        # Convert each skip connection to 3D
        x1 = self.to_3d(x1)
        x2 = self.to_3d(x2)
        x3 = self.to_3d(x3)
        x4 = self.to_3d(x4)
        
        # Pass skip connections through masked convolutions
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        x4 = self.conv4(x4)

        # Convert each skip connection back to 2D
        x1 = x1[:,:,self.target_frame]
        x2 = x2[:,:,self.target_frame]
        x3 = x3[:,:,self.target_frame]
        x4 = x4[:,:,self.target_frame]

        # Pass through 2D decoder
        x = self.decoder(x1, x2, x3, x4)

        return x   


class OneMaskedSkipUNet(nn.Module):
    def __init__(self,latent_dim=64,channels=3,temporal=3):
        super().__init__()
        self.target_frame = (temporal-1)//2
        self.encoder = EncoderBlock2d(latent_dim=latent_dim,channels=channels)
        self.decoder = SkipDecoderBlock2d(latent_dim=latent_dim,channels=channels)

        self.to_2d = Rearrange('b c t h w -> (b t) c h w')
        self.to_3d = Rearrange('(b t) c h w -> b c t h w',t=temporal)

        kernel_sizes = [17,11,7,3]
        mask_sizes = [7,5,3,1]
        self.mconv4 = MaskedConv3d(latent_dim,latent_dim,temporal,kernel_size=kernel_sizes[3],mask_size=mask_sizes[3])

        
    def forward(self,x):
        
        # Stack temporal onto batch
        x = self.to_2d(x)

        # Pass through 2D encoder
        x1, x2, x3, x4 = self.encoder(x)

        # Convert each skip connection to 3D
        x1 = self.to_3d(x1)
        x2 = self.to_3d(x2)
        x3 = self.to_3d(x3)
        x4 = self.to_3d(x4)
        
        # Pass skip connections through masked convolutions
        x4 = self.mconv4(x4)

        # Convert each skip connection back to 2D
        x1 = torch.zeros_like(x1[:,:,self.target_frame])
        x2 = torch.zeros_like(x2[:,:,self.target_frame])
        x3 = torch.zeros_like(x3[:,:,self.target_frame])
        x4 = x4[:,:,self.target_frame]

        # Pass through 2D decoder
        x = self.decoder(x1, x2, x3, x4)

        return x   

class TwoMaskedSkipUNet(nn.Module):
    def __init__(self,latent_dim=64,channels=3,temporal=3):
        super().__init__()
        _, _, l3 = get_layer_dimensions(latent_dim)
        self.target_frame = (temporal-1)//2
        self.encoder = EncoderBlock2d(latent_dim=latent_dim,channels=channels)
        self.decoder = SkipDecoderBlock2d(latent_dim=latent_dim,channels=channels)

        self.to_2d = Rearrange('b c t h w -> (b t) c h w')
        self.to_3d = Rearrange('(b t) c h w -> b c t h w',t=temporal)

        kernel_sizes = [17,11,7,3]
        mask_sizes = [7,5,3,1]
        self.mconv3 = MaskedConv3d(l3,l3,temporal,kernel_size=kernel_sizes[2],mask_size=mask_sizes[2])
        self.mconv4 = MaskedConv3d(latent_dim,latent_dim,temporal,kernel_size=kernel_sizes[3],mask_size=mask_sizes[3])

        
    def forward(self,x):
        
        # Stack temporal onto batch
        x = self.to_2d(x)

        # Pass through 2D encoder
        x1, x2, x3, x4 = self.encoder(x)

        # Convert each skip connection to 3D
        x1 = self.to_3d(x1)
        x2 = self.to_3d(x2)
        x3 = self.to_3d(x3)
        x4 = self.to_3d(x4)
        
        # Pass skip connections through masked convolutions
        x3 = self.mconv3(x3)
        x4 = self.mconv4(x4)

        # Convert each skip connection back to 2D
        x1 = torch.zeros_like(x1[:,:,self.target_frame])
        x2 = torch.zeros_like(x2[:,:,self.target_frame])
        x3 = x3[:,:,self.target_frame]
        x4 = x4[:,:,self.target_frame]

        # Pass through 2D decoder
        x = self.decoder(x1, x2, x3, x4)

        return x   

class ThreeMaskedSkipUNet(nn.Module):
    def __init__(self,latent_dim=64,channels=3,temporal=3):
        super().__init__()
        _, l2, l3 = get_layer_dimensions(latent_dim)
        self.target_frame = (temporal-1)//2
        self.encoder = EncoderBlock2d(latent_dim=latent_dim,channels=channels)
        self.decoder = SkipDecoderBlock2d(latent_dim=latent_dim,channels=channels)

        self.to_2d = Rearrange('b c t h w -> (b t) c h w')
        self.to_3d = Rearrange('(b t) c h w -> b c t h w',t=temporal)

        kernel_sizes = [17,11,7,3]
        mask_sizes = [7,5,3,1]
        self.mconv2 = MaskedConv3d(l2,l2,temporal,kernel_size=kernel_sizes[1],mask_size=mask_sizes[1])
        self.mconv3 = MaskedConv3d(l3,l3,temporal,kernel_size=kernel_sizes[2],mask_size=mask_sizes[2])
        self.mconv4 = MaskedConv3d(latent_dim,latent_dim,temporal,kernel_size=kernel_sizes[3],mask_size=mask_sizes[3])

        
    def forward(self,x):
        
        # Stack temporal onto batch
        x = self.to_2d(x)

        # Pass through 2D encoder
        x1, x2, x3, x4 = self.encoder(x)

        # Convert each skip connection to 3D
        x1 = self.to_3d(x1)
        x2 = self.to_3d(x2)
        x3 = self.to_3d(x3)
        x4 = self.to_3d(x4)
        
        # Pass skip connections through masked convolutions
        x2 = self.mconv2(x2)
        x3 = self.mconv3(x3)
        x4 = self.mconv4(x4)

        # Convert each skip connection back to 2D
        x1 = torch.zeros_like(x1[:,:,self.target_frame])
        x2 = x2[:,:,self.target_frame]
        x3 = x3[:,:,self.target_frame]
        x4 = x4[:,:,self.target_frame]

        # Pass through 2D decoder
        x = self.decoder(x1, x2, x3, x4)

        return x   


class FullConv2DMaskedSkipUNet(nn.Module):
    def __init__(self,latent_dim=64,channels=3,temporal=1):
        super().__init__()
        assert temporal==1, 'Temporal must be 1 for this model'
        l1, l2, l3 = get_layer_dimensions(latent_dim)
        self.encoder = EncoderBlock2d(latent_dim=latent_dim,channels=channels)
        self.decoder = SkipDecoderBlock2d(latent_dim=latent_dim,channels=channels)

        kernel_sizes = [17,11,7,3]
        mask_sizes = [7,5,3,1]
        self.mconv1 = MaskedConv2d(l1,l1,kernel_size=kernel_sizes[0],mask_size=mask_sizes[0])
        self.mconv2 = MaskedConv2d(l2,l2,kernel_size=kernel_sizes[1],mask_size=mask_sizes[1])
        self.mconv3 = MaskedConv2d(l3,l3,kernel_size=kernel_sizes[2],mask_size=mask_sizes[2])
        self.mconv4 = MaskedConv2d(latent_dim,latent_dim,kernel_size=kernel_sizes[3],mask_size=mask_sizes[3])

        
    def forward(self,x):
        if len(x.shape) == 5:
            x = x.squeeze(dim=2)
        
        # Pass through 2D encoder
        x1, x2, x3, x4 = self.encoder(x)

        # Pass skip connections through masked convolutions
        x1 = self.mconv1(x1)
        x2 = self.mconv2(x2)
        x3 = self.mconv3(x3)
        x4 = self.mconv4(x4)

        # Pass through 2D decoder
        x = self.decoder(x1, x2, x3, x4)

        return x   

class FullConv3DMaskedSkipUNet(nn.Module):
    def __init__(self,latent_dim=64,channels=3,temporal=3):
        super().__init__()
        l1, l2, l3 = get_layer_dimensions(latent_dim)
        self.target_frame = (temporal-1)//2
        self.encoder = EncoderBlock3d(latent_dim=latent_dim,channels=channels)
        self.decoder = SkipDecoderBlock3d(latent_dim=latent_dim,channels=channels)

        kernel_sizes = [17,11,7,3]
        mask_sizes = [7,5,3,1]
        self.mconv1 = MaskedConv3d(l1,l1,temporal,kernel_size=kernel_sizes[0],mask_size=mask_sizes[0])
        self.mconv2 = MaskedConv3d(l2,l2,temporal,kernel_size=kernel_sizes[1],mask_size=mask_sizes[1])
        self.mconv3 = MaskedConv3d(l3,l3,temporal,kernel_size=kernel_sizes[2],mask_size=mask_sizes[2])
        self.mconv4 = MaskedConv3d(latent_dim,latent_dim,temporal,kernel_size=kernel_sizes[3],mask_size=mask_sizes[3])

        
    def forward(self,x):
        
        # Pass through 3D encoder
        x1, x2, x3, x4 = self.encoder(x)

        # Pass skip connections through masked convolutions
        x1 = self.mconv1(x1)
        x2 = self.mconv2(x2)
        x3 = self.mconv3(x3)
        x4 = self.mconv4(x4)

        # Pass through 3D decoder
        x = self.decoder(x1, x2, x3, x4)
        x = x[:,:,self.target_frame]

        return x   


class FullMaskedSkipUNet(nn.Module):
    def __init__(self,latent_dim=64,channels=3,temporal=3):
        super().__init__()
        l1, l2, l3 = get_layer_dimensions(latent_dim)
        self.target_frame = (temporal-1)//2
        self.encoder = MaskedEncoderBlock2d(latent_dim=latent_dim,channels=channels)
        self.decoder = MaskedSkipDecoderBlock2d(latent_dim=latent_dim,channels=channels)

        self.to_2d = Rearrange('b c t h w -> (b t) c h w')
        self.to_3d = Rearrange('(b t) c h w -> b c t h w',t=temporal)

        kernel_sizes = [17,11,7,3]
        mask_sizes = [7,5,3,1]
        self.mconv1 = MaskedConv3d(l1,l1,temporal,kernel_size=kernel_sizes[0],mask_size=mask_sizes[0])
        self.mconv2 = MaskedConv3d(l2,l2,temporal,kernel_size=kernel_sizes[1],mask_size=mask_sizes[1])
        self.mconv3 = MaskedConv3d(l3,l3,temporal,kernel_size=kernel_sizes[2],mask_size=mask_sizes[2])
        self.mconv4 = MaskedConv3d(latent_dim,latent_dim,temporal,kernel_size=kernel_sizes[3],mask_size=mask_sizes[3])


    def forward(self,x):
        
        # Stack temporal onto batch
        x = self.to_2d(x)

        # Pass through 2D encoder
        x1, x2, x3, x4 = self.encoder(x)

        # Convert each skip connection to 3D
        x1 = self.to_3d(x1)
        x2 = self.to_3d(x2)
        x3 = self.to_3d(x3)
        x4 = self.to_3d(x4)
        
        # Pass skip connections through masked convolutions
        x1 = self.mconv1(x1)
        x2 = self.mconv2(x2)
        x3 = self.mconv3(x3)
        x4 = self.mconv4(x4)

        # Convert each skip connection back to 2D
        x1 = x1[:,:,self.target_frame]
        x2 = x2[:,:,self.target_frame]
        x3 = x3[:,:,self.target_frame]
        x4 = x4[:,:,self.target_frame]

        # Pass through 2D decoder
        x = self.decoder(x1, x2, x3, x4)

        return x   


class MaskedSkipUNet(nn.Module):
    def __init__(self,latent_dim=64,channels=3,temporal=3):
        super().__init__()
        l1, l2, l3 = get_layer_dimensions(latent_dim)
        self.target_frame = (temporal-1)//2
        self.encoder = EncoderBlock2d(latent_dim=latent_dim,channels=channels)
        self.decoder = SkipDecoderBlock2d(latent_dim=latent_dim,channels=channels)

        self.to_2d = Rearrange('b c t h w -> (b t) c h w')
        self.to_3d = Rearrange('(b t) c h w -> b c t h w',t=temporal)

        kernel_sizes = [17,11,7,3]
        mask_sizes = [7,5,3,1]
        self.mconv1 = MaskedConv3d(l1,l1,temporal,kernel_size=kernel_sizes[0],mask_size=mask_sizes[0])
        self.mconv2 = MaskedConv3d(l2,l2,temporal,kernel_size=kernel_sizes[1],mask_size=mask_sizes[1])
        self.mconv3 = MaskedConv3d(l3,l3,temporal,kernel_size=kernel_sizes[2],mask_size=mask_sizes[2])
        self.mconv4 = MaskedConv3d(latent_dim,latent_dim,temporal,kernel_size=kernel_sizes[3],mask_size=mask_sizes[3])

        
    def forward(self,x):
        
        # Stack temporal onto batch
        x = self.to_2d(x)

        # Pass through 2D encoder
        x1, x2, x3, x4 = self.encoder(x)

        # Convert each skip connection to 3D
        x1 = self.to_3d(x1)
        x2 = self.to_3d(x2)
        x3 = self.to_3d(x3)
        x4 = self.to_3d(x4)
        
        # Pass skip connections through masked convolutions
        x1 = self.mconv1(x1)
        x2 = self.mconv2(x2)
        x3 = self.mconv3(x3)
        x4 = self.mconv4(x4)

        # Convert each skip connection back to 2D
        x1 = x1[:,:,self.target_frame]
        x2 = x2[:,:,self.target_frame]
        x3 = x3[:,:,self.target_frame]
        x4 = x4[:,:,self.target_frame]

        # Pass through 2D decoder
        x = self.decoder(x1, x2, x3, x4)

        return x   

