# training script for DTU dataset
# adapted from https://github.com/autonomousvision/gaussian-opacity-fields/blob/main/scripts/run_dtu.py

import os, sys
from concurrent.futures import ThreadPoolExecutor
import subprocess
import time
import torch
from argparse import ArgumentParser

scenes = [24, 40, 55, 63, 106, 110, 118, 122]

factors = [2] * len(scenes)

excluded_gpus = set([])
depth_weight_init = "0.0"
depth_weight_tune = "0.0"
output_dir = "output_orig/DTU_3DGS"
tune_output_dir = "output_orig/DTU_Trim3DGS"

iteration = 7000

jobs = list(zip(scenes, factors))

normal_weight = {24:0.1, 63:0.1, 65:0.1, 97:0.01, 110:0.01, 118:0.02}
prune_ratio = {55:0.2, 114:0.2}

def train_scene(gpu, scene, factor):
    cmds = [
            # eval
            f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python eval_dtu/evaluate_single_scene.py --input_mesh {tune_output_dir}/scan{scene}/tsdf/ours_{iteration}/mesh_post.ply --scan_id {scene} --output_dir {tune_output_dir}/scan{scene}/tsdf/ours_{iteration} --mask_dir data/dtu_dataset/DTU --DTU data/dtu_dataset/Official_DTU_Dataset",
            f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python eval_dtu_pcd/evaluate_single_scene.py --input_pcd {tune_output_dir}/scan{scene}/point_cloud/iteration_{iteration}/point_cloud.ply --scan_id {scene} --output_dir {tune_output_dir}/scan{scene}/train/ours_{iteration} --mask_dir data/dtu_dataset/DTU --DTU data/dtu_dataset/Official_DTU_Dataset",
            # eval render
            f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python render.py -m {tune_output_dir}/scan{scene}",
            f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python metrics.py -m {tune_output_dir}/scan{scene}"
        ]

    for cmd in cmds:
        print(cmd)
        subprocess.run(cmd, shell=True, check=True)
    return True


def worker(gpu, scene, factor):
    print(f"Starting job on GPU {gpu} with scene {scene}\n")
    train_scene(gpu, scene, factor)
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

