import torch
import torch.nn as nn

class NormLoss(nn.Module):
    def __init__(self,temporal=3):
        super().__init__()
        self.target_frame = (temporal-1)//2

    def forward(self,x_true,x_pred):
        if len(x_true.shape) == 5:
            x_true = x_true[:,:,self.target_frame] 
        
        recon_loss = torch.norm(x_true-x_pred,dim=1).mean()
        return recon_loss