import torch
from torch import nn
from utils.graphics_utils import getProjectionMatrix

class PseudoCamera(nn.Module):
    def __init__(self, w2c, FoVx, FoVy, height, width, name):
    
        super(PseudoCamera, self).__init__()

        self.name = name

        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_width = width
        self.image_height = height

        # arguments for the transformation matrices
        self.zfar = 100.0
        self.znear = 0.01

        # transformation matrices
        self.world_view_transform = torch.tensor(w2c).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]