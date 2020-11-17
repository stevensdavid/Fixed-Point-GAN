import subprocess
import os
from argparse import ArgumentParser
import sys
import glob

def run_command(cmd):
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=1, universal_newlines=True) as p:
        for line in p.stdout:
            print(line, end='')

# NOTE: MOST LIKELY NEED TO ADD FID CAPABILITIES FOR PCAM AS WELL, BUT I DON'T HAVE TIME
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="CelebA", choices=["CelebA", "PCam"])
    parser.add_argument("--input_image_format", type=str, default="jpg", choices=["jpg", "png"], help="Only relevant for CelebA")    
    parser.add_argument('--c_dim', type=int, default=1, help='dimension of domain labels')
    parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset',
                        default=['Eyeglasses'])
    # parser.add("--output_image_format", type=str, default="jpg", choices=["jpg", "png"])
    parser.add_argument("--test_iters", type=int, help="Model iterations to load")
    parser.add_argument("--batch_size", type=int, help="Batch size to use during generation of images from test set")
    parser.add_argument("--class_distribution", default=0.07053526763381691, type=float, help="Distribution of activated/inactivated label in data. Assumes usage of a single label with binary output. Default is CelebA eyeglasses distribution.")
    parser.add_argument("--python", default="python3", type=str, help="Python interpreter")
    parser.add_argument("--reuse_generated", type=bool, default=False, help="try to reuse generated images instead of generating new ones")
    config = parser.parse_args()

    files = glob.glob('./celeba/results/*')
    for f in files:
        os.remove(f)

    # fid stats for total, for only glasses and for only non-glasses
    fid_stats_path = f"fid_stats_{config.dataset.lower()}.npz"
    if not os.path.exists(fid_stats_path):
        run_command(f"{config.python} precalc_fid_stats.py --dataset {config.dataset} --image_format {config.input_image_format} --from_files".split())
    
    fid_stats_path = f"fid_stats_{config.dataset.lower()}_eyeglasses.npz"
    if not os.path.exists(fid_stats_path):
        run_command(f"{config.python} precalc_fid_stats.py --dataset {config.dataset} --image_format {config.input_image_format} --eyeglasses --from_files".split())
    
    fid_stats_path = f"fid_stats_{config.dataset.lower()}_no_eyeglasses.npz"
    if not os.path.exists(fid_stats_path):
        run_command(f"{config.python} precalc_fid_stats.py --dataset {config.dataset} --image_format {config.input_image_format} --no_eyeglasses --from_files".split())

    # # fid stats for generated total
    # if config.reuse_generated == False:
    #   if config.dataset == "PCam":
    #       command = f"""
    #           {config.python} main.py --mode test --dataset PCam --image_dir data/pcam
    #           --image_size 96 --crop_size 96 --c_dim 1  
    #           --model_save_dir pretrained_models/pcam 
    #           --result_dir pcam/results --test_iters {config.test_iters}
    #           --batch_size {config.batch_size} --random_target 0.5 --exclude_source --single_image_output
    #       """ .replace("\n", "").split()
    #   elif config.dataset == "CelebA":
    #       # not sure if c_dim == selected_attrs in the case of hair related attributes, 
    #       # but we are not dealing with them anyway.
    #       command = f"""
    #           {config.python} main.py --mode test --dataset CelebA --image_dir data/celeba/images
    #           --image_size 128 --c_dim {config.c_dim}
    #           --model_save_dir pretrained_models/celeba 
    #           --result_dir celeba/results --test_iters {config.test_iters}
    #           --batch_size {config.batch_size} --random_target {config.class_distribution} 
    #           --random_target_class Eyeglasses
    #           --exclude_source --selected_attrs {" ".join(str(x) for x in config.selected_attrs)} 
    #           --single_image_output
    #       """ .replace("\n", "").split()
    #   else:
    #       print("Unsupported dataset")
    #       sys.exit(1)
    # run_command(command)
    # print("FID ON COMPLETE TEST DATA: ", end='')
    # run_command(f"{config.python} gan_fid_calculator.py --dataset {config.dataset} --stats_path fid_stats_{config.dataset.lower()}.npz".split())

    files = glob.glob('./celeba/results/*')
    for f in files:
      os.remove(f)

    # fid stats for eyeglass images generated from eyeglass data 
    if config.reuse_generated == False:
      if config.dataset == "PCam":
          command = f"""
              {config.python} main.py --mode test --dataset PCam --image_dir data/pcam
              --image_size 96 --crop_size 96 --c_dim 1  
              --model_save_dir pretrained_models/pcam 
              --result_dir pcam/results --test_iters {config.test_iters}
              --batch_size {config.batch_size} --random_target 0.5 --exclude_source --single_image_output 
          """ .replace("\n", "").split()
      elif config.dataset == "CelebA":
          # not sure if c_dim == selected_attrs in the case of hair related attributes, 
          # but we are not dealing with them anyway.
          command = f"""
              {config.python} main.py --mode test --dataset CelebA --image_dir data/celeba/images
              --image_size 128 --c_dim {config.c_dim}
              --model_save_dir pretrained_models/celeba 
              --result_dir celeba/results --test_iters {config.test_iters}
              --batch_size {config.batch_size} --random_target 1 
              --random_target_class Eyeglasses
              --exclude_source --selected_attrs {" ".join(str(x) for x in config.selected_attrs)} 
              --single_image_output --eyeglasses
          """ .replace("\n", "").split()
      else:
          print("Unsupported dataset")
          sys.exit(1)
    run_command(command)
    print("FID BETWEEN REAL EYEGLASS IMAGES AND FAKE EYEGLASS IMAGES GENERATED FROM IDENTITY TRANSFORM: ", end='')
    run_command(f"{config.python} gan_fid_calculator.py --dataset {config.dataset} --stats_path fid_stats_{config.dataset.lower()}_eyeglasses.npz".split())

    files = glob.glob('./celeba/results/*')
    for f in files:
      os.remove(f)

    # fid stats for eyeglass images generated from non-eyeglass data 
    if config.reuse_generated == False:
      if config.dataset == "PCam":
          command = f"""
              {config.python} main.py --mode test --dataset PCam --image_dir data/pcam
              --image_size 96 --crop_size 96 --c_dim 1  
              --model_save_dir pretrained_models/pcam 
              --result_dir pcam/results --test_iters {config.test_iters}
              --batch_size {config.batch_size} --random_target 0.5 --exclude_source --single_image_output
          """ .replace("\n", "").split()
      elif config.dataset == "CelebA":
          # not sure if c_dim == selected_attrs in the case of hair related attributes, 
          # but we are not dealing with them anyway.
          command = f"""
              {config.python} main.py --mode test --dataset CelebA --image_dir data/celeba/images
              --image_size 128 --c_dim {config.c_dim}
              --model_save_dir pretrained_models/celeba 
              --result_dir celeba/results --test_iters {config.test_iters}
              --batch_size {config.batch_size} --random_target 1 
              --random_target_class Eyeglasses
              --exclude_source --selected_attrs {" ".join(str(x) for x in config.selected_attrs)} 
              --single_image_output --no_eyeglasses
          """ .replace("\n", "").split()
      else:
          print("Unsupported dataset")
          sys.exit(1)
    run_command(command)
    print("FID BETWEEN REAL EYEGLASS IMAGES AND FAKE EYEGLASS IMAGES GENERATED FROM NON EYEGLASS DATA: ", end='')
    run_command(f"{config.python} gan_fid_calculator.py --dataset {config.dataset} --stats_path fid_stats_{config.dataset.lower()}_eyeglasses.npz".split())

    files = glob.glob('./celeba/results/*')
    for f in files:
      os.remove(f)

    # fid stats for non-eyeglass images generated from eyeglass data 
    if config.reuse_generated == False:
      if config.dataset == "PCam":
          command = f"""
              {config.python} main.py --mode test --dataset PCam --image_dir data/pcam
              --image_size 96 --crop_size 96 --c_dim 1  
              --model_save_dir pretrained_models/pcam 
              --result_dir pcam/results --test_iters {config.test_iters}
              --batch_size {config.batch_size} --random_target 0.5 --exclude_source --single_image_output 
          """ .replace("\n", "").split()
      elif config.dataset == "CelebA":
          # not sure if c_dim == selected_attrs in the case of hair related attributes, 
          # but we are not dealing with them anyway.
          command = f"""
              {config.python} main.py --mode test --dataset CelebA --image_dir data/celeba/images
              --image_size 128 --c_dim {config.c_dim}
              --model_save_dir pretrained_models/celeba 
              --result_dir celeba/results --test_iters {config.test_iters}
              --batch_size {config.batch_size} --random_target 0 
              --random_target_class Eyeglasses
              --exclude_source --selected_attrs {" ".join(str(x) for x in config.selected_attrs)} 
              --single_image_output --eyeglasses
          """ .replace("\n", "").split()
      else:
          print("Unsupported dataset")
          sys.exit(1)
    run_command(command)
    print("FID BETWEEN REAL NON-EYEGLASS IMAGES AND FAKE NON-EYEGLASS IMAGES GENERATED FROM EYEGLASS DATA: ", end='')
    run_command(f"{config.python} gan_fid_calculator.py --dataset {config.dataset} --stats_path fid_stats_{config.dataset.lower()}_no_eyeglasses.npz".split())

    files = glob.glob('./celeba/results/*')
    for f in files:
      os.remove(f)

    # fid stats for non-eyeglass images generated from non-eyeglass data 
    if config.reuse_generated == False:
      if config.dataset == "PCam":
          command = f"""
              {config.python} main.py --mode test --dataset PCam --image_dir data/pcam
              --image_size 96 --crop_size 96 --c_dim 1  
              --model_save_dir pretrained_models/pcam 
              --result_dir pcam/results --test_iters {config.test_iters}
              --batch_size {config.batch_size} --random_target 0.5 --exclude_source --single_image_output
          """ .replace("\n", "").split()
      elif config.dataset == "CelebA":
          # not sure if c_dim == selected_attrs in the case of hair related attributes, 
          # but we are not dealing with them anyway.
          command = f"""
              {config.python} main.py --mode test --dataset CelebA --image_dir data/celeba/images
              --image_size 128 --c_dim {config.c_dim}
              --model_save_dir pretrained_models/celeba 
              --result_dir celeba/results --test_iters {config.test_iters}
              --batch_size {config.batch_size} --random_target 0 
              --random_target_class Eyeglasses
              --exclude_source --selected_attrs {" ".join(str(x) for x in config.selected_attrs)} 
              --single_image_output --no_eyeglasses
          """ .replace("\n", "").split()
      else:
          print("Unsupported dataset")
          sys.exit(1)
    run_command(command)
    print("FID BETWEEN REAL NON-EYEGLASS IMAGES AND FAKE NON-EYEGLASS IMAGES GENERATED FROM IDENTITY TRANSFORM: ", end='')
    run_command(f"{config.python} gan_fid_calculator.py --dataset {config.dataset} --stats_path fid_stats_{config.dataset.lower()}_no_eyeglasses.npz".split())

    files = glob.glob('./celeba/results/*')
    for f in files:
      os.remove(f)

    # FID ON COMPLETE TEST DATA: FID: 9.166371487022872
    # 