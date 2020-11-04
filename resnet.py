# Ignore futurewarnings from using old version of tensorboard
from solver import Solver
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
from torchvision.models.resnet import resnet50
from data_loader import get_loader
from argparse import ArgumentParser
from main import str2bool
import torch.nn as nn
import torch.optim as optim
from numpy import inf
from logger import Logger
from tqdm import tqdm
import os


def add_missing_solver_args(config):
    # Model configurations.
    expected_attrs = [
        "c_dim", "c2_dim", "image_size", "g_conv_dim", "d_conv_dim", "g_repeat_num",
        "d_repeat_num", "lambda_cls", "lambda_rec", "lambda_gp", "lambda_id",
        "dataset", "batch_size", "num_iters", "num_iters_decay", "g_lr", "d_lr",
        "n_critic", "beta1", "beta2", "resume_iters", "selected_attrs", "test_iters",
        "use_tensorboard", "log_dir", "sample_dir", "model_save_dir", "result_dir",
        "log_step", "sample_step", "model_save_step", "lr_update_step"
    ]
    for attr in expected_attrs:
        if not hasattr(config, attr):
            setattr(config, attr, None)
    # Set model hyperparameters to their defaults
    config.d_conv_dim = 64
    config.g_conv_dim = 64
    config.g_repeat_num = 6
    config.d_repeat_num = 6
    return config

class ResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = resnet50(pretrained=False)
        self.model.fc = nn.Linear(2048, num_classes)
        self.activation = nn.Sigmoid()
        # self.activation = nn.Identity()
        # Make AVG pooling input shape independent
        self.model.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.model(x)
        return self.activation(x)

    def predict(self, x):
        x = self.forward(x)
        return self.activation(x)



def train_resnet(config):
    logger = Logger(config.log_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # device = "cpu"

    os.makedirs(config.resnet_save_dir, exist_ok=True)
    model = ResNet(num_classes=1)
    train_data = get_loader(
        config.image_dir,
        config.attr_path,
        config.selected_attrs,
        config.crop_size,
        config.image_size,
        config.batch_size,
        config.dataset,
        "train",
        config.num_workers,
        config.in_memory,
        weighted=config.dataset == "CelebA",
    )
    val_data = get_loader(
        config.image_dir,
        config.attr_path,
        config.selected_attrs,
        config.crop_size,
        config.image_size,
        config.batch_size,
        config.dataset,
        "val",
        config.num_workers,
        config.in_memory,
        weighted=config.dataset == "CelebA",
    )

    if config.generator_iters is not None:
        config = add_missing_solver_args(config)
        generator = Solver(train_data, config, train_mode=False)
        generator.restore_model(config.generator_iters)
    else:
        generator = None

    loss_function = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(
        model.parameters(), lr=config.lr, betas=[config.beta1, config.beta2]
    )
    start_iters = 0

    scaler = torch.cuda.amp.GradScaler()
    best_val_loss = inf
    tests_since_best = 0

    loss_log = {}
    model.to(device)

    for epoch in tqdm(range(1, config.epochs + 1), desc="Epochs", disable=not config.progress_bar):
        model.train()
        running_loss = 0.0
        correct_classifications = 0
        n_positive_samples = 0
        n_negative_samples = 0
        correct_positive = 0
        correct_negative = 0
        # Training
        tqdm.write("Training")
        for batch_idx, (x, y) in enumerate(train_data, 1):
            if generator is not None:
                x, y = generator.invert_batch(x, y)

            model.train()
            optimizer.zero_grad()
            x = x.to(device)
            if y.shape[1] > 1:
                # Only extract first attribute.
                y = y[:,0:1]
            y = y.to(device)
            with torch.cuda.amp.autocast():
                output = model(x).to(device)
                loss = loss_function(output, y).to(device)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pred = output >= 0.5
            running_loss += loss.item()
            correct_classifications += pred.eq(y).sum().item()
            n_positive_samples += y.sum()
            n_negative_samples += (1-y).sum()
            correct_positive += pred.eq(y).logical_and(y).sum().item()
            correct_negative += pred.eq(y).logical_and(1-y).sum().item()
            if config.verbose:
                n_samples = batch_idx * config.batch_size
                current_loss = running_loss / batch_idx
                current_acc = correct_classifications / n_samples
                current_sensitivity = correct_positive / max(n_positive_samples, 1)
                current_specificity = correct_negative / max(n_negative_samples, 1)
                tqdm.write(
                        f"Training batch {batch_idx}/{len(train_data)} Current loss: {current_loss:.4f} Current accuracy: {current_acc:.4f} Current sensitivity: {current_sensitivity:.4f} Current specificity: {current_specificity:.4f}"
                )
        train_loss = running_loss / len(train_data)
        train_acc = correct_classifications / len(train_data.dataset)
        train_sensitivity = correct_positive / max(n_positive_samples, 1)
        train_specificity = correct_negative / max(n_negative_samples, 1)

        # Validating
        tqdm.write("Validating")
        with torch.no_grad():
            model.eval()
            val_loss = 0
            correct_classifications = 0
            n_positive_samples = 0
            n_negative_samples = 0
            correct_positive = 0
            correct_negative = 0
            for batch_idx, (x, y) in enumerate(val_data, 1):
                if generator is not None:
                    x, y = generator.invert_batch(x, y)
                x = x.to(device)
                if y.shape[1] > 1:
                    # Only extract first attribute.
                    y = y[:,0:1]
                y = y.to(device)
                with torch.cuda.amp.autocast():
                    output = model(x).to(device)
                    loss = loss_function(output, y).to(device)
                pred = output >= 0.5
                correct_classifications += pred.eq(y).sum().item()
                n_positive_samples += y.sum()
                n_negative_samples += (1-y).sum()
                correct_positive += pred.eq(y).logical_and(y).sum().item()
                correct_negative += pred.eq(y).logical_and(1-y).sum().item()
                val_loss += loss.item()
                if config.verbose:
                    n_samples = batch_idx * config.batch_size
                    cur_val_loss = val_loss / batch_idx
                    cur_val_acc = correct_classifications / n_samples
                    current_sensitivity = correct_positive / max(n_positive_samples, 1)
                    current_specificity = correct_negative / max(n_negative_samples, 1)
                    tqdm.write(
                            f"Validating batch {batch_idx}/{len(val_data)} Current loss: {cur_val_loss:.4f} Current accuracy: {cur_val_acc:.4f} Current sensitivity: {current_sensitivity:.4f} Current specificity: {current_specificity:.4f}"
                    )
            val_loss = val_loss / len(val_data)
            val_acc = correct_classifications / len(val_data.dataset)
            val_specificty = correct_positive / max(n_positive_samples, 1)
            val_sensitivity = correct_negative / max(n_negative_samples, 1)

        # Logging
        loss_log["train_loss"] = train_loss
        loss_log["train_acc"] = train_acc
        loss_log["train_spec"] = train_specificity
        loss_log["train_sens"] = train_sensitivity
        loss_log["val_loss"] = val_loss
        loss_log["val_acc"] = val_acc
        loss_log["val_spec"] = val_specificty
        loss_log["val_sens"] = val_sensitivity

        tqdm.write(
            f"Epoch: {epoch} Training loss: {train_loss:.4f} Training accuracy: {train_acc:.4f} Training sensitivty: {train_sensitivity:.4f} Training specificity: {train_specificity:.4f} Validation loss: {val_loss:.4f} Validation accuracy: {val_acc:.4f} Validation sensitivity: {val_sensitivity:.4f} Validation specificity: {val_specificty:.4f}"
        )

        if config.use_tensorboard:
            for tag, value in loss_log.items():
                logger.scalar_summary(tag, value, epoch)

        # Early stopping
        if config.use_early_stopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                tests_since_best = 0
                best_model_save_path = os.path.join(config.resnet_save_dir, f"{config.dataset}_resnet_best.ckpt")
                torch.save(model.state_dict(), best_model_save_path)
            else:
                tests_since_best += 1
                if tests_since_best >= config.patience:
                    tqdm.write(
                        f"Reached early stopping threshold with patience {config.patience}."
                    )
                    break
    model_save_path = os.path.join(config.resnet_save_dir, f"{config.dataset}_resnet.ckpt")
    torch.save(model.state_dict(), model_save_path)
    print(f"Saved model into path {model_save_path}")


def main(config):
    if config.mode == "train":
        train_resnet(config)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--c_dim", type=int, default=5, help="dimension of domain labels (1st dataset)"
    )
    parser.add_argument(
        "--crop_size", type=int, default=178, help="crop size for the images"
    )
    parser.add_argument("--image_size", type=int, default=128, help="image resolution")

    # Training configuration.
    parser.add_argument("--in_memory", action="store_true", help="Store dataset in RAM")
    parser.add_argument("--dataset", type=str, default="PCam", choices=["PCam", "CelebA"])
    parser.add_argument("--batch_size", type=int, default=16, help="mini-batch size")
    parser.add_argument(
        "--epochs", type=int, default=200000, help="number of total epochs for training"
    )
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument(
        "--beta1", type=float, default=0.5, help="beta1 for Adam optimizer"
    )
    parser.add_argument(
        "--beta2", type=float, default=0.999, help="beta2 for Adam optimizer"
    )
    parser.add_argument("--use_early_stopping", action="store_true", default=True)
    parser.add_argument(
        "--patience",
        type=int,
        default=2,
        help="patience for early stopping measured in validation loss tracking",
    )
    parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset', default=None)    
    parser.add_argument("--attr_path", type=str, default=None)

    # Miscellaneous.
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument(
        "--mode", type=str, default="train", choices=["train", "test", "test_brats"]
    )
    parser.add_argument("--use_tensorboard", action="store_true")
    parser.add_argument(
        "--verbose", action="store_true", help="Print loss after each batch"
    )
    parser.add_argument("--progress_bar", action="store_true", help="Print progress bars")

    # Directories.
    parser.add_argument("--image_dir", type=str, default="data/pcam")
    parser.add_argument("--log_dir", type=str, default="pcam/resnet/logs")
    parser.add_argument("--resnet_save_dir", type=str, default="pcam/resnet/models")
    parser.add_argument(
        "--model_save_dir", 
        type=str, 
        default="pcam/models", 
        help="Directory where GAN models are saved"
    )

    parser.add_argument(
        "--generator_iters",
        type=int,
        help="Train on generated data from a model restored form the supplied iteration."
    )

    config = parser.parse_args()
    main(config)
