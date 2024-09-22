import json
import os
from argparse import ArgumentParser


def report_dtu(path, iteration, render_only=False):
    print(f'Results of {path}')
    scans = os.listdir(path)
    if not render_only:
        print("***************** mesh *****************")
        sum_overall = 0
        n = 0
        for scan in sorted(scans, key=lambda x: int(x.replace('scan', ''))):
            p = os.path.join(path, scan, f'train/ours_{iteration}/results.json')
            if not os.path.exists(p):
                continue
            with open(p, 'r') as f:
                data = json.load(f)
                print(scan, data, scan)
            sum_overall += data['overall']
            n += 1
        print(f"Overall: {sum_overall / n}")

        sum_overall = 0
        n = 0
        print("***************** pcd *****************")
        for scan in sorted(scans, key=lambda x: int(x.replace('scan', ''))):
            p = os.path.join(path, scan, f'train/ours_{iteration}/results_pcd.json')
            if not os.path.exists(p):
                continue
            with open(p, 'r') as f:
                data = json.load(f)
                print(scan, data, scan)
            sum_overall += data['overall']
            n += 1
        print(f"Overall: {sum_overall / n}")

    sum_psnr, sum_ssim, sum_lpips = [], [], []
    n = 0
    print("***************** Render *****************")
    for scan in sorted(scans, key=lambda x: int(x.replace('scan', ''))):
        p = os.path.join(path, scan, 'results.json')
        if not os.path.exists(p):
            continue
        with open(p, 'r') as f:
            data = json.load(f)
            data = data[f'ours_{iteration}']
            print(scan, data, scan)
        sum_psnr.append(data['PSNR'])
        sum_ssim.append(data['SSIM'])
        sum_lpips.append(data['LPIPS'])
        n += 1
    sum_psnr.sort()
    sum_ssim.sort()
    sum_lpips.sort()
    print(f"n-3 SSIM: {sum(sum_ssim[3:]) / (n - 3)} PSNR: {sum(sum_psnr[3:]) / (n - 3)} LPIPS: {sum(sum_lpips[3:]) / (n - 3)}")
    print(f"SSIM: {sum(sum_ssim) / (n)} PSNR: {sum(sum_psnr) / (n)} LPIPS: {sum(sum_lpips) / (n)}")
    
def report_mipnerf360(path, iteration):
    print(f'Results of {path}')
    scans = os.listdir(path)
    sum_overall = 0
    n = 0
    for scan in sorted(scans):
        p = os.path.join(path, scan, f'point_cloud/iteration_{iteration}/point_cloud.ply')
        if not os.path.exists(p):
            print(f"Missing {p}")
            continue
        # check the storage size of the point cloud
        size = os.path.getsize(p)
        mb_size = size / 1024 / 1024
        print(scan, f"{mb_size:.2f} MB")
        sum_overall += mb_size
        n += 1
    print(f"Overall: {sum_overall / n:.2f} MB")

    indoor = ['room', 'counter', 'kitchen', 'bonsai']
    outdoor = ['bicycle', 'flowers', 'garden', 'stump', 'treehill']
    sum_overall_indoor = dict()
    sum_overall_outdoor = dict()
    n_indoor = 0
    n_outdoor = 0
    for scan in sorted(scans):
        p = os.path.join(path, scan, 'results.json')
        if not os.path.exists(p):
            continue
        with open(p, 'r') as f:
            data = json.load(f)
            print(scan, data, scan)
        if scan in indoor:
            for k, v in data[f'ours_{iteration}'].items():
                if k not in sum_overall_indoor:
                    sum_overall_indoor[k] = 0.0
                sum_overall_indoor[k] += v
            n_indoor += 1
        if scan in outdoor:
            for k, v in data[f'ours_{iteration}'].items():
                if k not in sum_overall_outdoor:
                    sum_overall_outdoor[k] = 0.0
                sum_overall_outdoor[k] += v
            n_outdoor += 1

    print("Outdoor")
    for k, v in sum_overall_outdoor.items():
        print(f"{k}: {v / n_outdoor:.3f}")

    print("Indoor")
    for k, v in sum_overall_indoor.items():
        print(f"{k}: {v / n_indoor:.3f}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--output_path', '-o', type=str)
    parser.add_argument('--iteration', type=int, default=7000)
    parser.add_argument('--dataset', type=str, choices=['dtu', 'mipnerf360'])
    parser.add_argument('--render_only', action='store_true')
    args = parser.parse_args()

    if args.dataset == 'dtu':
        report_dtu(args.output_path, args.iteration, args.render_only)
    elif args.dataset == 'mipnerf360':
        report_mipnerf360(args.output_path, args.iteration)
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")