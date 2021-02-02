import sys
import subprocess
import os
import socket
from argparse import ArgumentParser, REMAINDER

import torch


def parse_args():
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser = ArgumentParser(description="PyTorch distributed training launch "
                                        "helper utilty that will spawn up "
                                        "multiple distributed processes")

    # Optional arguments for the launch helper
    parser.add_argument("--master_addr", default="127.0.0.1", type=str,
                        help="Master node (rank 0)'s address, should be either "
                             "the IP address or the hostname of node 0, for "
                             "single node multi-proc training, the "
                             "--master_addr can simply be 127.0.0.1")
    parser.add_argument("--master_port", default=29500, type=int,
                        help="Master node (rank 0)'s free port that needs to "
                             "be used for communciation during distributed "
                             "training")
    
    parser.add_argument("--log_dir", default="./", type=str,
                        help="output directory for log file")

    # positional
    parser.add_argument("training_script", type=str,
                        help="The full path to the single GPU training "
                             "program/script to be launched in parallel, "
                             "followed by all the arguments for the "
                             "training script")

    # rest from the training program
    parser.add_argument('training_script_args', nargs=REMAINDER)
    return parser.parse_args()


def main():
    args = parse_args()

    # world size in terms of number of processes
    world_size = torch.cuda.device_count()

    # set PyTorch distributed related environmental variables
    current_env = os.environ.copy()
    current_env["MASTER_ADDR"] = args.master_addr
    current_env["MASTER_PORT"] = str(args.master_port)
    current_env["WORLD_SIZE"] = str(world_size)
    gpu_ids = current_env["CUDA_VISIBLE_DEVICES"].split(",")

    processes = []

    for local_rank in range(world_size):
        # each process's rank
        current_env["RANK"] = str(local_rank)

        stdout = None if local_rank == 0 else open(os.path.join(args.log_dir, "GPU_"+str(local_rank)+".log"), "w")

        # spawn the processes
        cmd = [sys.executable,
               "-u",
               args.training_script,
               "--local_rank={}".format(local_rank),
               "--world_size={}".format(world_size)
               ] + args.training_script_args

        #current_env["CUDA_VISIBLE_DEVICES"] = gpu_ids[local_rank]
        
        print(cmd)
        process = subprocess.Popen(cmd, env=current_env, stdout=stdout)
        processes.append(process)

    for process in processes:
        process.wait()


if __name__ == "__main__":
    main()
