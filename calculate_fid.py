import subprocess
import os
from argparse import ArgumentParser
import sys

def run_command(cmd):
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=1, universal_newlines=True) as p:
        for line in p.stdout:
            print(line, end='')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="CelebA", choices=["CelebA", "PCam"])
    parser.add_argument("--input_image_format", type=str, default="jpg", choices=["jpg", "png"], help="Only relevant for CelebA")
    # parser.add("--output_image_format", type=str, default="jpg", choices=["jpg", "png"])
    parser.add_argument("--test_iters", type=int, help="Model iterations to load")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--class_distribution", default=0.06511878143524893, type=float, help="Distribution of activated/inactivated label in data. Assumes usage of a single label with binary output. Default is CelebA eyeglasses distribution.")
    parser.add_argument("--python", default="python3", type=str, help="Python interpreter")
    config = parser.parse_args()

    fid_stats_path = f"fid_stats_{config.dataset.lower()}.npz"
    if not os.path.exists(fid_stats_path):
        run_command(f"{config.python} precalc_stats_example.py --dataset {config.dataset} --image_format {config.input_image_format}")

    if config.dataset == "PCam":
        command = f"""
            {config.python} main.py --mode test --dataset PCam
            --image_size 96 --crop_size 96 --c_dim 1  
            --model_save_dir pretrained_models/celeba 
            --result_dir pcam/results --test_iters {config.test_iters}
            --batch_size {config.batch_size} --random_target 0.5 --exclude_source --single_image_output
        """ .replace("\n", "")
    elif config.dataset == "CelebA":
        command = f"""
            {config.python} main.py --mode test --dataset CelebA 
            --image_size 128 --crop_size 128 --c_dim 1  
            --model_save_dir pretrained_models/celeba 
            --result_dir celeba/results --test_iters {config.test_iters}
            --batch_size {config.batch_size} --random_target {config.class_distribution} 
            --exclude_source --selected_attrs Eyeglasses --single_image_output
        """ .replace("\n", "")
    else:
        print("Unsupported dataset")
        sys.exit(1)
    run_command(command)

    run_command(f"{config.python} fix_example.py --dataset {config.dataset}")