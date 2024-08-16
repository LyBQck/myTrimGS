import torch
from torch import nn

class SdfNet(nn.Module):
    
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.model = nn.Sequential(
            nn.Linear(3, latent_dim),
            nn.ReLU(True),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(True),
            nn.Linear(latent_dim, 1)
        )
        
        
        # for name, param in self.named_parameters():
        #     # if 'weight' in name:
        #     nn.init.constant_(param, 0.)

    def forward(self, x):
        x = self.model(x)
        return torch.abs(x)
        # return torch.exp(x)
