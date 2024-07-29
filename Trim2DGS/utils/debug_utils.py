import numpy as np
import torch
from scene.cameras import Camera
from scene.gaussian_model import BasicPointCloud

def create_debug_cam():
    return Camera(colmap_id=1, R=np.eye(3), T=np.zeros(3), FoVx=0.525, FoVy=0.398, image=torch.zeros((3, 581, 777)), gt_alpha_mask=None, image_name="debug_cam", uid=0, data_device="cuda")

def create_debug_ply(cam):
    w2c = cam.world_view_transform.transpose(0, 1).cpu().numpy()
    c2w = np.linalg.inv(w2c)
    num = 10
    pos_c = np.array([
                        [0.0, 0.0, 1.0],
                        [0.0, 0.0, 2.0],
                        [0.0, 0.0, 3.0],
                        [0.0, 0.0, 4.0],
                        [0.0, 0.0, 5.0],
                        [0.0, 0.0, 6.0],
                        [0.0, 0.0, 7.0],
                        [0.0, 0.0, 8.0],
                        [0.0, 0.0, 9.0],
                        [0.0, 0.0, 10.0]
                        ])
    # pos_c = np.concatenate([pos_c] * num, axis=0)
    pos_w = c2w[:3, :3] @ pos_c.T + c2w[:3, 3:4]
    color_arr = np.concatenate([np.array([[1.0, 1.0, 1.0]])] * num, axis=0)
    norm_arr = np.concatenate([np.array([[0.0, 0.0, 0.0]])] * num, axis=0)
    return BasicPointCloud(points=pos_w.T, colors=color_arr, normals=norm_arr)