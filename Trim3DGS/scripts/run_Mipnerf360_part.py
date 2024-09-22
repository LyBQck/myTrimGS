# training script for MipNeRF360 dataset
# adapted from https://github.com/autonomousvision/gaussian-opacity-fields/blob/main/scripts/run_mipnerf360.py

import os, sys
from concurrent.futures import ThreadPoolExecutor
import subprocess
import time
import torch
from argparse import ArgumentParser
# if __name__ == "__main__":
parser = ArgumentParser()
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--scene", type=str, default="bicycle")
parser.add_argument("--depth_weight_init", type=str, default="0.0")
parser.add_argument("--depth_weight_tune", type=str, default="0.0")
parser.add_argument("--tune_depth", action="store_true", help="Whether to tune depth weight during first 30k training iterations")

parser.add_argument("--extract_and_eval", action="store_true", help="Whether to extract and evaluate only")
parser.add_argument("--extra_name", type=str, default="")
args = parser.parse_args(sys.argv[1:])

scenes = [args.scene]

# scenes = ["bicycle", "bonsai", "counter", "flowers", "garden", "stump", "treehill", "kitchen", "room"]

factors = [4, 2, 2, 4, 4, 4, 4, 2, 2]

normal_weights = [0.1, 0.01, 0.01, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01]

full_gpus = set([0, 1, 2, 3, 4, 5, 6, 7])
if args.gpu == -1:
    excluded_gpus = set([])
else:
    excluded_gpus = full_gpus - set([args.gpu])
depth_weight_init = args.depth_weight_init
depth_weight_tune = args.depth_weight_tune
output_dir = "output_l2_" + depth_weight_init.replace(".", "-") + args.tune_depth * "tune" + "_" + depth_weight_tune.replace(".", "-") + args.extra_name + "/MipNeRF360_3DGS"
tune_output_dir = "output_l2_" + depth_weight_init.replace(".", "-") + "_" + depth_weight_tune.replace(".", "-") + args.extra_name + "/MipNeRF360_Trim3DGS"

split = "scale"
iteration = 7000

jobs = list(zip(scenes, factors, normal_weights))


def train_scene(gpu, scene, factor, weight):
    cmds = [
            f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python train.py -s data/MipNeRF360/{scene} -m {output_dir}/{scene} --eval -i images_{factor} --test_iterations -1 --quiet --depth_weight=" + depth_weight_init + args.tune_depth * " --tune_depth",
            f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python tune.py -s data/MipNeRF360/{scene} -m {tune_output_dir}/{scene} --eval -i images_{factor} --pretrained_ply {output_dir}/{scene}/point_cloud/iteration_30000/point_cloud.ply --test_iterations -1 --quiet --split {split} --position_lr_init 0.0000016 --densification_interval 1000 --opacity_reset_interval 999999 --normal_regularity_param {weight} --contribution_prune_from_iter 0 --contribution_prune_interval 1000 --depth_weight=" + depth_weight_tune,
            f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python render.py --iteration {iteration} -m {tune_output_dir}/{scene} --eval --skip_train --render_other",
            f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python metrics.py -m {tune_output_dir}/{scene}",
            # f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python extract_mesh_tsdf.py -s data/MipNeRF360/{scene} -m {tune_output_dir}/{scene} --eval -i images_{factor} --iteration {iteration} --voxel_size 0.004 --sdf_trunc 0.04",
        ]

    for cmd in cmds:
        print(cmd)
        subprocess.run(cmd, shell=True, check=True)
    return True

def extract_and_eval(gpu, scene):
    iteration_local = 30000
    cmds = [
            # extract
            # f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python extract_mesh_tsdf.py -s data/MipNeRF360/{scene} -m {tune_output_dir}/{scene} --eval -i images_{factor} --iteration {iteration} --voxel_size 0.004 --sdf_trunc 0.04",
            # f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python extract_mesh_tsdf.py -s data/MipNeRF360/{scene} -m {output_dir}/{scene} --iteration {iteration_local} --num_cluster 1 --voxel_size 0.004 --sdf_trunc 0.016 --depth_trunc 3.0",
            # eval
            # PASS
            # eval render
            f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python render.py -m {output_dir}/{scene} --skip_train",
            f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python metrics.py -m {output_dir}/{scene}"
        ]

    for cmd in cmds:
        print(cmd)
        subprocess.run(cmd, shell=True, check=True)
    return True

def worker(gpu, scene, factor, weight):
    print(f"Starting job on GPU {gpu} with scene {scene}\n")
    if not args.extract_and_eval:
        train_scene(gpu, scene, factor, weight)
    else:
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

