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
parser.add_argument("--depth_weight", type=str, default="0.0")
parser.add_argument("--depth_opac_weight", type=str, default="0.0")
parser.add_argument("--tune_depth", action="store_true", help="Whether to tune depth weight during first 30k training iterations")
parser.add_argument("--tune_depth_from_iter", type=int, default=7000)

parser.add_argument("--skip_train", action="store_true", help="Whether to skip training")
parser.add_argument("--extra_name", type=str, default="")
args = parser.parse_args(sys.argv[1:])

scenes = [24, 40, 55, 63, 106, 110, 118, 122]
eval_str = args.eval_mode * " --eval"

factors = [2] * len(scenes)

full_gpus = set([0, 1, 2, 3, 4, 5, 6, 7])
if args.gpu == -1:
    excluded_gpus = set([])
else:
    excluded_gpus = full_gpus - set([args.gpu])

depth_weight = args.depth_weight
depth_opac_weight = args.depth_opac_weight

output_dir = "output0/dtu_alpha_depth_multi_scenes/" + "d_wgt_" + depth_weight.replace(".", "-") + "_d-opac_wgt_" + depth_opac_weight.replace(".", "-") + args.tune_depth * ("_tune_" + str(args.tune_depth_from_iter)+ "_") + args.extra_name + "/DTU_3DGS"
test_iterations = " ".join([str(i) for i in args.test_iterations])

jobs = list(zip(scenes, factors))

normal_weight = {24:0.1, 63:0.1, 65:0.1, 97:0.01, 110:0.01, 118:0.02}
prune_ratio = {55:0.2, 114:0.2}

def train_scene(gpu, scene, factor):
    cmds = [
            f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python train.py -s data/dtu_dataset/DTU/scan{scene} -m {output_dir}/scan{scene} -r {factor} --test_iterations " + test_iterations + " --depth_weight=" + depth_weight + " --depth_opac_weight=" + depth_opac_weight + args.tune_depth * (" --tune_depth --tune_depth_from_iter " + str(args.tune_depth_from_iter)) + eval_str
        ]

    for cmd in cmds:
        print(cmd)
        subprocess.run(cmd, shell=True, check=True)
    return True

def extract_and_eval(gpu, scene):
    iteration_local = 30000
    cmds = [
            # extract
            f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python extract_mesh_tsdf.py -s data/dtu_dataset/DTU/scan{scene} -m {output_dir}/scan{scene} --iteration {iteration_local} --num_cluster 1 --voxel_size 0.004 --sdf_trunc 0.016 --depth_trunc 3.0" + eval_str,
            # eval
            f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python eval_dtu/evaluate_single_scene.py --input_mesh {output_dir}/scan{scene}/tsdf/ours_{iteration_local}/mesh_post.ply --scan_id {scene} --output_dir {output_dir}/scan{scene}/tsdf/ours_{iteration_local} --mask_dir data/dtu_dataset/DTU --DTU data/dtu_dataset/Official_DTU_Dataset",
            f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python eval_dtu_pcd/evaluate_single_scene.py --input_pcd {output_dir}/scan{scene}/point_cloud/iteration_{iteration_local}/point_cloud.ply --scan_id {scene} --output_dir {output_dir}/scan{scene}/train/ours_{iteration_local} --mask_dir data/dtu_dataset/DTU --DTU data/dtu_dataset/Official_DTU_Dataset",
            # eval render
            f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python render.py -m {output_dir}/scan{scene} --skip_train" + eval_str,
            f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python metrics.py -m {output_dir}/scan{scene}"
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

