import os
import argparse
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn


def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    # Data loader.
    data_loader = None

    if config.dataset in ['CelebA']:
        data_loader = get_loader(config.image_dir, config.attr_path, config.selected_attrs,
                                    config.crop_size, config.image_size, config.batch_size,
                                    'CelebA', config.mode, config.num_workers, config.eval_dataset,
                                    match_distribution=config.mode == "test",
                                    subsample_offset=1 if config.mode == "test" else None)
    elif config.dataset in ['PCam']:
        data_loader = get_loader(config.image_dir, None, None,
                                   config.crop_size, config.image_size, config.batch_size,
                                   'PCam', config.mode, config.num_workers, config.eval_dataset)
    elif config.dataset in ['BRATS']:
        data_loader = get_loader(config.image_dir, None, None,
                                   config.crop_size, config.image_size, config.batch_size,
                                   'BRATS', config.mode, config.num_workers,config.eval_dataset)
    elif config.dataset in ['Directory']:
        data_loader = get_loader(config.image_dir, None, None,
                                 config.crop_size, config.image_size, config.batch_size,
                                 'Directory', config.mode, config.num_workers, config.eval_dataset)

        
    # Solver for training and testing Fixed-Point GAN.
    solver = Solver(data_loader, config)



    if config.mode == 'train':
        if config.dataset in ['CelebA', 'PCam', 'BRATS', 'Directory']:
            solver.train()
    elif config.mode == 'test':
        if config.dataset in ['Directory']:
            solver.test()
        elif config.dataset in ['PCam']:
            solver.test_pcam()
        elif config.dataset in ['CelebA']:
            if config.c_dim > 1:
                solver.test_celeba_multi()
            else:
                solver.test_celeba();    
    elif config.mode == 'test_brats':
        if config.dataset in ['BRATS']:
            solver.test_brats()
    elif config.mode == 'generate_dataset':
        if config.dataset in ['PCam']:
            solver.generate_dataset()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--c_dim', type=int, default=5, help='dimension of domain labels (1st dataset)')
    parser.add_argument('--c2_dim', type=int, default=8, help='dimension of domain labels (2nd dataset)')
    parser.add_argument('--crop_size', type=int, default=178, help='crop size for the images')
    parser.add_argument('--image_size', type=int, default=128, help='image resolution')
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
    parser.add_argument('--lambda_id', type=float, default=10, help='weight for identity loss')
    
    # Training configuration.
    parser.add_argument('--dataset', type=str, default='PCam', choices=['CelebA', 'BRATS', 'PCam', 'Directory'])
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=200000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
    parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset',
                        default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'])

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=200000, help='test model from this step')
    parser.add_argument('--eval_resnet_id_name', type=str, default=200000, help='name of resnet in /models to load for evaluation')
    parser.add_argument('--eval_resnet_tilde_name', type=str, default=200000, help='name of resnet in /models to load for evaluation')
    parser.add_argument('--eval_dataset', type=str, default='test', choices=["test","train", "val"], help='Valid datasets are default and training')


    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'test_brats', ])
    parser.add_argument('--use_tensorboard', type=str2bool, default=False)

    # Directories.
    parser.add_argument('--image_dir', type=str, default='data/celeba/images')
    parser.add_argument('--attr_path', type=str, default='data/celeba/list_attr_celeba.txt')
    parser.add_argument('--log_dir', type=str, default='celeba/logs')
    parser.add_argument('--model_save_dir', type=str, default='celeba/models')
    parser.add_argument('--sample_dir', type=str, default='celeba/samples')
    parser.add_argument('--result_dir', type=str, default='celeba/results')
    parser.add_argument('--resnet_save_dir', type=str, default="resnets")

    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=1000)
    parser.add_argument('--lr_update_step', type=int, default=1000)

    config = parser.parse_args()
    print(config)
    main(config)