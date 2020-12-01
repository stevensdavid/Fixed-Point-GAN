# Fixed-Point GAN <img src="https://img.shields.io/badge/Patent-Pending-yellow"/>
This repository provides the official PyTorch implementation of Fixed-Point GAN. Fixed-Point GAN introduces fixed-point translation which dramatically reduces artifacts in image-to-image translation and introduces a novel method for disease detection and localization using image-level annotation only.

<img src="images/image-to-image_translation_example.png" alt="Example of image-to-image translation"/>
<img src="images/disease_detection_localization_example.png" alt="Example of disease detection and localization"/>

## Paper
[**Learning Fixed Points in Generative Adversarial Networks:<br/>From Image-to-Image Translation to Disease Detection and Localization**](https://arxiv.org/abs/1908.06965)

[Md Mahfuzur Rahman Siddiquee](https://github.com/mahfuzmohammad)<sup>1</sup>, [Zongwei Zhou](https://github.com/MrGiovanni)<sup>1,3</sup>, [Nima Tajbakhsh](https://www.linkedin.com/in/nima-tajbakhsh-b5454376/)<sup>1</sup>, [Ruibin Feng](https://chs.asu.edu/ruibin-feng)<sup>1</sup>,<br/>
[Michael B. Gotway](https://www.mayoclinic.org/biographies/gotway-michael-b-m-d/bio-20055566)<sup>2</sup>, [Yoshua Bengio](https://mila.quebec/en/yoshua-bengio/)<sup>3</sup>, [Jianming Liang](https://chs.asu.edu/jianming-liang)<sup>1,3</sup><br/>

<sup>1</sup>Arizona State University; <sup>2</sup>Mayo Clinic; <sup>3</sup>Mila – Quebec Artificial Intelligence Institute<br/>
International Conference on Computer Vision ([ICCV](http://iccv2019.thecvf.com)), 2019

## Abstract
Generative adversarial networks (GANs) have ushered in a revolution in image-to-image translation. The development and proliferation of GANs raises an interesting question: can we train a GAN to remove an object, if present, from an image while otherwise preserving the image? Specifically, can a GAN "virtually heal" anyone by turning his medical image, with an unknown health status (diseased or healthy), into a healthy one, so that diseased regions could be revealed by subtracting those two images? Such a task requires a GAN to identify a minimal subset of target pixels for domain translation, an ability that we call fixed-point translation, which no GAN is equipped with yet. Therefore, we propose a new GAN, called Fixed-Point GAN, trained by (1) supervising same-domain translation through a conditional identity loss, and (2) regularizing cross-domain translation through revised adversarial, domain classification, and cycle consistency loss. Based on fixed-point translation, we further derive a novel framework for disease detection and localization using only image-level annotation. Qualitative and quantitative evaluations demonstrate that the proposed method outperforms the state of the art in multi-domain image-to-image translation and that it surpasses predominant weakly-supervised localization methods in both disease detection and localization.

## Usage

### 0. Cloning the repository

```bash
$ git clone https://github.com/mahfuzmohammad/Fixed-Point-GAN.git
$ cd Fixed-Point-GAN/
```

### 1. Creating python environment

```bash
$ conda env create -f conda_env/conda_env_pytorch0.2.yml
$ source activate pytorch0.2
$ cat conda_env/pip_pytorch0.2.txt | xargs -n 1 pip install
```

### 2. Downloading the dataset

To download the CelebA dataset:

```bash
$ bash download.sh celeba
```

To download the processed BRATS 2013 synthetic dataset:

```bash
$ bash download.sh brats
```

To download the PatchChamelyon dataset:

```bash
$ bash download.sh pcam
```

### 3. Training

Training on CelebA dataset

```bash
$ python main.py --mode train --dataset CelebA --image_size 128 --c_dim 5 \
                 --sample_dir celeba/samples \
                 --log_dir celeba/logs \
                 --model_save_dir celeba/models \
                 --result_dir celeba/results \
                 --selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young --lambda_id 10
```

Training on CelebA with glasses as attribute:

```bash
$ python main.py --mode train --dataset CelebA --image_size 128 --c_dim 1 \
                 --sample_dir celeba/samples \
                 --log_dir celeba/logs \
                 --model_save_dir celeba/models \
                 --result_dir celeba/results \
                 --selected_attrs Eyeglasses --lambda_id 10
```

In PowerShell:
```powershell
$ python main.py --mode train --dataset CelebA --image_size 128 --c_dim 5 `
                 --sample_dir celeba/samples `
                 --log_dir celeba/logs `
                 --model_save_dir celeba/models `
                 --result_dir celeba/results `
                 --selected_attrs Eyeglasses --lambda_id 10
```

Training on BRATS dataset

```bash
$ python main.py --mode train --dataset BRATS --crop_size 256 --image_size 256 --c_dim 1 \
                 --image_dir data/brats/syn \
                 --sample_dir brats_syn_256_lambda0.1/samples \
                 --log_dir brats_syn_256_lambda0.1/logs \
                 --model_save_dir brats_syn_256_lambda0.1/models \
                 --result_dir brats_syn_256_lambda0.1/results \
                 --batch_size 8 --num_workers 4 --lambda_id 0.1 --num_iters 10000
```

Training on PatchCamelyon dataset

```bash
$ python main.py --mode train --dataset PCam --crop_size 96 --image_size 96 --c_dim 1 \
                 --image_dir data/pcam \
                 --sample_dir pcam/samples \
                 --log_dir pcam/logs \
                 --model_save_dir pcam/models \
                 --result_dir pcam/results \
                 --batch_size 32 --num_workers 8 --lambda_id 0.1 --num_iters 10000
```

In PowerShell:
```powershell
$ python main.py --mode train --dataset PCam --crop_size 96 --image_size 96 --c_dim 1 `
                 --image_dir data/pcam `
                 --sample_dir pcam/samples `
                 --log_dir pcam/logs `
                 --model_save_dir pcam/models `
                 --result_dir pcam/results `
                 --batch_size 32 --num_workers 8 --lambda_id 0.1 --num_iters 10000
```

### 4. Testing

Testing on CelebA dataset

```bash
$ python main.py --mode test --dataset CelebA --image_size 128 --c_dim 5 \
                 --sample_dir celeba/samples \
                 --log_dir celeba/logs \
                 --model_save_dir celeba/models \
                 --result_dir celeba/results \
                 --selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young --lambda_id 10
```

Testing on CelebA dataset with glasses as attribute

```bash
$ python main.py --mode test --dataset CelebA --image_size 128 --c_dim 5 \
                 --sample_dir celeba/samples \
                 --log_dir celeba/logs \
                 --model_save_dir celeba/models \
                 --result_dir celeba/results \
                 --selected_attrs Eyeglasses --lambda_id 10
```

In PowerShell

```powershell
$ python main.py --mode test --dataset CelebA --image_size 128 --c_dim 5 `
                 --sample_dir celeba/samples `
                 --log_dir celeba/logs `
                 --model_save_dir celeba/models `
                 --result_dir celeba/results `
                 --selected_attrs Eyeglasses --lambda_id 10
```

Testing on BRATS dataset

```bash
$ python main.py --mode test_brats --dataset BRATS --crop_size 256 --image_size 256 --c_dim 1 \
                 --image_dir data/brats/syn \
                 --sample_dir brats_syn_256_lambda0.1/samples \
                 --log_dir brats_syn_256_lambda0.1/logs \
                 --model_save_dir brats_syn_256_lambda0.1/models \
                 --result_dir brats_syn_256_lambda0.1/results \
                 --batch_size 16 --num_workers 4 --lambda_id 0.1 --test_iters 300000
```

### 5. Testing using pretrained models

Testing on CelebA dataset using pretrained models

```bash
$ bash download.sh pretrained_celeba_128
$ python main.py --mode test --dataset CelebA --image_size 128 --c_dim 5 \
                 --sample_dir celeba/samples \
                 --log_dir celeba/logs \
                 --model_save_dir pretrained_models/celeba \
                 --result_dir celeba/results \
                 --selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young --lambda_id 10
```

Testing on BRATS dataset using pretrained models

```bash
$ bash download.sh pretrained_brats_256
$ python main.py --mode test_brats --dataset BRATS --crop_size 256 --image_size 256 --c_dim 1 \
                 --image_dir data/brats/syn --sample_dir brats_syn_256_lambda0.1/samples \
                 --log_dir brats_syn_256_lambda0.1/logs \
                 --model_save_dir pretrained_models/brats_syn_256_lambda0.1 \
                 --result_dir brats_syn_256_lambda0.1/results \
                 --batch_size 16 --num_workers 4 --lambda_id 0.1 --test_iters 300000
$ python brats_auc.py
```

CelebA

1. Single ID utvärderad på verkliga trainingbilder / Single Tilde utvärderad på verkliga trainingbilder
```bash
python main.py --mode test --dataset CelebA --image_size 128 --c_dim 1 --sample_dir celeba/samples --log_dir celeba/logs --model_save_dir celeba/models --result_dir celeba/results/1 --selected_attrs Eyeglasses --lambda_id 5  --test_iters 4200000 --eval_dataset train --eval_resnet_id_name celeba_id_resnet_single.ckpt --eval_resnet_tilde_name celeba_tilde_resnet_single.ckpt
```
2. Dual ID utvärderad på verkliga träningsbilder /  Dual Tilde utvärderad på verkliga träningsbilder
```bash
python main.py --mode test --dataset CelebA --image_size 128 --c_dim 2 --sample_dir celeba/samples --log_dir celeba/logs --model_save_dir celeba/models --result_dir celeba/results/2 --selected_attrs Eyeglasses Narrow_Eyes  --lambda_id 5  --test_iters 2550000 --eval_dataset train --eval_resnet_id_name celeba_id_resnet_dual.ckpt --eval_resnet_tilde_name celeba_tilde_resnet_dual.ckpt
```
3. Single, Actual utvärderad på genererade tilde från validering / Actual utvärderad på genererade ID från validering
```bash
python main.py --mode test --dataset CelebA --image_size 128 --c_dim 1 --sample_dir celeba/samples --log_dir celeba/logs --model_save_dir celeba/models --result_dir celeba/results/3 --selected_attrs Eyeglasses --lambda_id 5  --test_iters 4200000 --eval_dataset test --eval_resnet_id_name celeba_resnet_actual.ckpt --eval_resnet_tilde_name celeba_resnet_actual.ckpt
```
4. Dual, Actual utvärderad på genererade tilde från validering / Actual utvärderad på genererade ID från validering
```bash
python main.py --mode test --dataset CelebA --image_size 128 --c_dim 2 --sample_dir celeba/samples --log_dir celeba/logs --model_save_dir celeba/models --result_dir celeba/results/4 --selected_attrs Eyeglasses Narrow_Eyes  --lambda_id 5  --test_iters 2550000 --eval_dataset test --eval_resnet_id_name celeba_resnet_actual.ckpt --eval_resnet_tilde_name celeba_resnet_actual.ckpt
```

PCAM

1. Single ID utvärderad på verkliga trainingbilder / Single Tilde utvärderad på verkliga trainingbilder
python main.py --mode test --dataset PCam --crop_size 96 --image_size 96 --c_dim 1                  --image_dir data/pcam                  --sample_dir pcam/samples                  --log_dir pcam/logs                  --model_save_dir pcam/models                  --result_dir pcam/results/1                  --batch_size 32 --num_workers 8 --lambda_id 0.1 --num_iters 10000 --test_iters 4600000 --eval_dataset train --eval_resnet_id_name pcam_id_resnet.ckpt --eval_resnet_tilde_name pcam_tilde_resnet.ckpt
2. Single, Actual utvärderad på genererade tilde från validering / Actual utvärderad på genererade ID från validering
python main.py --mode test --dataset PCam --crop_size 96 --image_size 96 --c_dim 1                  --image_dir data/pcam                  --sample_dir pcam/samples                  --log_dir pcam/logs                  --model_save_dir pcam/models                  --result_dir pcam/results/2                  --batch_size 32 --num_workers 8 --lambda_id 0.1 --num_iters 10000 --test_iters 4600000 --eval_dataset test --eval_resnet_id_name pcam_actual_resnet.ckpt --eval_resnet_tilde_name pcam_actual_resnet.ckpt








### 6. Calculating FID (EXPERIMENTAL, only for PCam)
See the official [FID implementation](https://github.com/bioinf-jku/TTUR) for more information.

1. Execute `python3 precalc_stats_example.py` (will calculate and store the distribution of real 
   samples in a file `fix_stats.npz`). Assumes usage of tensorflow 1.x (1.15 was used during experimentation).
2. Execute `python3 main.py --mode test --dataset PCam --image_size 96 --crop_size 96 --c_dim 1  --log_dir pcam/logs --model_save_dir pretrained_models/pcam --result_dir pcam/results --image_dir data/pcam --test_iters 620000 --lambda_id 0.1 --batch_size 1` (will produce and store generated images from the test dataset). Assumes that a model that has run for 620000 iterations is available for testing.
2. Execute `python3 fid_example.py` (calculates final FID based on distribution of generated values and estimation of the real distribution). Assumes usage of tensorflow 1.x (1.15 was used during experimentation).

## Citation

Please cite this work as following:

```
@inproceedings{siddiquee2019learning,
  title={Learning Fixed Points in Generative Adversarial Networks: From Image-to-Image Translation to Disease Detection and Localization},
  author={Rahman Siddiquee, Md Mahfuzur and Zhou, Zongwei and Tajbakhsh, Nima and Feng, Ruibin and Gotway, Michael B and Bengio, Yoshua and Liang, Jianming},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={191--200},
  year={2019}
}
```

## Acknowledgements

This is a patent-pending technology. This research has been supported partially by ASU and Mayo Clinic through a Seed Grant and an Innovation Grant, and partially by NIH under Award Number R01HL128785. The content is solely the responsibility of the authors and does not necessarily represent the official views of NIH. This repository has been built upon [yunjey/stargan](https://github.com/yunjey/stargan).
