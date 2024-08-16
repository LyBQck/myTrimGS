import os
from tqdm import tqdm
import open3d as o3d
import numpy as np
import sklearn.neighbors as skln
from scipy.io import loadmat
import argparse

if __name__ == '__main__':
    pbar = tqdm(total=1)
    parser = argparse.ArgumentParser()
    parser.add_argument('--scan', type=int, default=24)
    parser.add_argument('--dataset_dir', type=str, default='data/dtu_dataset/Official_DTU_Dataset')
    parser.add_argument('--out_dir', type=str, default='nn_sdf/data')
    parser.add_argument('--downsample_density', type=float, default=0.2)
    parser.add_argument('--min_dist', type=float, default=1.0)
    parser.add_argument('--boundary_length', type=float, default=1.3)
    args = parser.parse_args()
    thresh = args.downsample_density

    stl_pcd = o3d.io.read_point_cloud(f'{args.dataset_dir}/Points/stl/stl{args.scan:03}_total.ply')
    stl = np.asarray(stl_pcd.points)
    ground_plane = loadmat(f'{args.dataset_dir}/ObsMask/Plane{args.scan}.mat')['P']
    stl_hom = np.concatenate([stl, np.ones_like(stl[:,:1])], -1)
    above = (ground_plane.reshape((1,4)) * stl_hom).sum(-1) > 0
    stl = stl[above]
    nn_engine = skln.NearestNeighbors(n_neighbors=1, radius=thresh, algorithm='kd_tree', n_jobs=-1)
    nn_engine.fit(stl)

    # neg_pcd = torch.rand(1000, 3) # random points range from 0 to 1
    neg_pcd = np.random.rand(stl.shape[0], 3)
    for i in range(3):
        # neg_pcd[:, i] = neg_pcd[:, i] * (stl[:, i].max() - stl[:, i].min()) + stl[:, i].min()
        neg_pcd[:, i] = neg_pcd[:, i] * args.boundary_length * (stl[:, i].max() - stl[:, i].min()) + args.boundary_length * stl[:, i].min()
    pbar.set_description('compute data2stl')
    dist_d2s, idx_d2s = nn_engine.kneighbors(neg_pcd, n_neighbors=1, return_distance=True)
    min_dist = args.min_dist
    dist_d2s = dist_d2s[min_dist < dist_d2s]
    neg_pcd = neg_pcd[min_dist < dist_d2s]
    
    pbar.update(1)
    print(f'neg_pcd.shape: {neg_pcd.shape}, dist_d2s.shape: {dist_d2s.shape}')
    os.makedirs(f'{args.out_dir}/scan{args.scan}', exist_ok=True)
    np.save(f'{args.out_dir}/scan{args.scan}/stl.npy', stl)
    np.save(f'{args.out_dir}/scan{args.scan}/neg.npy', neg_pcd)
    np.save(f'{args.out_dir}/scan{args.scan}/neg_dist.npy', dist_d2s)