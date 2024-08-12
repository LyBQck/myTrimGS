# training script for DTU dataset
# adapted from https://github.com/autonomousvision/gaussian-opacity-fields/blob/main/scripts/run_dtu.py

import os, sys
from concurrent.futures import ThreadPoolExecutor
import subprocess
import time
import torch
from argparse import ArgumentParser
# if __name__ == "__main__":
parser = ArgumentParser()
parser.add_argument("--gpu", type=int, default=-1)
parser.add_argument("--eval_mode", action="store_true")
parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])

# train args
parser.add_argument("--tune_depth", action="store_true", help="Whether to tune depth weight during first 30k training iterations")
parser.add_argument("--tune_depth_from_iter", type=int, default=1000)
parser.add_argument("--lambda_dist_alpha", type=float, default=1e-6)
parser.add_argument("--lambda_dist", type=float, default=0.0)

# 0: mapped, full; 1: mapped, simplified; 2: orig, full; 3: orig, simplified
parser.add_argument("--control_id", type=int, default=0)
parser.add_argument("--debug_depth", action="store_true", help="Whether to debug depth weight")
parser.add_argument("--skip_train", action="store_true", help="Whether to skip training")
parser.add_argument("--extra_name", type=str, default="")
args = parser.parse_args(sys.argv[1:])

scenes = ['scan24', 'scan37', 'scan40', 'scan55', 'scan63', 'scan65', 'scan69', 'scan83', 'scan97', 'scan105', 'scan106', 'scan110', 'scan114', 'scan118', 'scan122']
# scenes = ['scan69', 'scan106', 'scan110', 'scan114', 'scan118', 'scan122']
if args.debug_depth:
    scenes = ['scan24']

eval_str = args.eval_mode * " --eval"
debug_depth = args.debug_depth * " --debug_depth --densify_from_iter=10000 --opacity_reset_interval=10000 --save_images"

factors = [2] * len(scenes)

full_gpus = set([0, 1, 2, 3, 4, 5, 6, 7])
if args.gpu == -1:
    excluded_gpus = set([])
else:
    excluded_gpus = full_gpus - set([args.gpu])

output_dir = "output0/dtu_alpha_depth_multi_scenes/" + args.tune_depth * ("tune_" + str(args.tune_depth_from_iter)+ "_") + args.extra_name + "/DTU_2DGS"
test_iterations = " ".join([str(i) for i in args.test_iterations])

jobs = list(zip(scenes, factors))

position_lr_init = {"scan63": 0.0000016}
iteration = 30000

def train_scene(gpu, scene, factor):
    cmds = [
            f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python train.py -s data/dtu_dataset/DTU/{scene} -m {output_dir}/{scene} " + args.tune_depth * (" --tune_depth --tune_depth_from_iter " + str(args.tune_depth_from_iter)) + " --test_iterations " + test_iterations + f" --depth_ratio 1.0 -r {factor} --save_images --lambda_dist {args.lambda_dist} --lambda_dist_alpha {args.lambda_dist_alpha} --control_id {args.control_id}"  + eval_str + debug_depth,
        ]

    for cmd in cmds:
        print(cmd)
        subprocess.run(cmd, shell=True, check=True)
    return True

def extract_and_eval(gpu, scene):
    scan_id = scene[4:]
    cmds = [
            f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python cull_pcd.py -m {output_dir}/{scene} --iteration 30000",
            f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python render.py -s data/dtu_dataset/DTU/{scene} -m {output_dir}/{scene} --skip_train --depth_ratio 1.0 --num_cluster 1 --iteration {iteration} --voxel_size 0.004 --voxel_size 0.004 --sdf_trunc 0.016 --depth_trunc 3.0",
            f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python scripts/eval_dtu/evaluate_single_scene.py --input_mesh {output_dir}/{scene}/train/ours_{iteration}/fuse_post.ply --scan_id {scan_id} --output_dir {output_dir}/{scene}/train/ours_{iteration} --mask_dir data/dtu_dataset/DTU --DTU data/dtu_dataset/Official_DTU_Dataset",
            f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python scripts/eval_dtu_pcd/evaluate_single_scene.py --input_pcd {output_dir}/{scene}/point_cloud/iteration_{iteration}/point_cloud.ply --scan_id {scan_id} --output_dir {output_dir}/{scene}/train/ours_{iteration} --mask_dir data/dtu_dataset/DTU --DTU data/dtu_dataset/Official_DTU_Dataset",
            # eval render
            f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python render.py -s data/dtu_dataset/DTU/{scene} -m {output_dir}/{scene} --skip_mesh --iteration {iteration}",
            f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python metrics.py -m {output_dir}/{scene}"
        ]

    for cmd in cmds:
        print(cmd)
        subprocess.run(cmd, shell=True, check=True)
    return True

def worker(gpu, scene, factor):
    print(f"Starting job on GPU {gpu} with scene {scene}\n")
    if not args.skip_train:
        train_scene(gpu, scene, factor)
    extract_and_eval(gpu, scene)
    print(f"Finished job on GPU {gpu} with scene {scene}\n")
    # This worker function starts a job and returns when it's done.

def dispatch_jobs(jobs, executor):
    future_to_job = {}
    reserved_gpus = set()  # GPUs that are slated for work but may not be active yet

    while jobs or future_to_job:
        # Get the list of available GPUs, not including those that are reserved.
        all_available_gpus = set(range(torch.cuda.device_count()))
        available_gpus = list(all_available_gpus - reserved_gpus - excluded_gpus)

        # Launch new jobs on available GPUs
        while available_gpus and jobs:
            gpu = available_gpus.pop(0)
            job = jobs.pop(0)
            future = executor.submit(worker, gpu, *job)  # Unpacking job as arguments to worker
            future_to_job[future] = (gpu, job)

            reserved_gpus.add(gpu)  # Reserve this GPU until the job starts processing

        # Check for completed jobs and remove them from the list of running jobs.
        # Also, release the GPUs they were using.
        done_futures = [future for future in future_to_job if future.done()]
        for future in done_futures:
            job = future_to_job.pop(future)  # Remove the job associated with the completed future
            gpu = job[0]  # The GPU is the first element in each job tuple
            reserved_gpus.discard(gpu)  # Release this GPU
            print(f"Job {job} has finished., rellasing GPU {gpu}")
        # (Optional) You might want to introduce a small delay here to prevent this loop from spinning very fast
        # when there are no GPUs available.
        time.sleep(5)

    print("All jobs have been processed.")

# Using ThreadPoolExecutor to manage the thread pool
with ThreadPoolExecutor(max_workers=8) as executor:
    dispatch_jobs(jobs, executor)

