import torch
import numpy as np
from gaussian_renderer import render
from geometry_regularization.pseudo_cameras import PseudoCamera
from geometry_regularization.cam_utils import get_transform

# geometry_loss = geometry_regularization(viewpoint_cam, gaussians, pipe, torch.zeros_like(background))
def geometry_regularization(cam, gaussians, pipe, background):
    ratio_rec = pipe.depth_ratio
    pipe.depth_ratio = 0
    render_pkg = render(cam, gaussians, pipe, background)
    pipe.depth_ratio = ratio_rec
    depth_map_orig = render_pkg['surf_depth']

    w2c = cam.world_view_transform.cpu().transpose(0, 1).numpy()
    trans = np.array([0.0, 0.0, 0.0])
    rots = np.array([0.0, 0.0, 0.0])
    loss = 0.0
    for i in range(3):
        for theta in np.linspace(-30, 30, 3):
            rots[i] = theta
            w2c_pseudo = get_transform(w2c=w2c, trans=trans, rots=rots)
            pseudo_cam = PseudoCamera(w2c_pseudo, cam.FoVx, cam.FoVy, cam.image_height, cam.image_width, cam.name)
            ratio_rec = pipe.depth_ratio
            pipe.depth_ratio = 0
            render_pkg = render(pseudo_cam, gaussians, pipe, background)
            pipe.depth_ratio = ratio_rec
            depth_map = render_pkg['surf_depth']

            # loss += torch.sum(torch.abs(depth_map - depth_map_orig))

    return loss