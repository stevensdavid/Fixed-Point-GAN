import torch
from torchvision.models import resnet50
from data_loader import get_loader
from argparse import ArgumentParser
from main import str2bool
import torch.nn as nn
import torch.optim as optim
from numpy import inf
from logger import Logger
from tqdm import tqdm
import os


class ResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = resnet50(pretrained=False, num_classes=num_classes)
        self.activation = nn.Softmax()
        # self.activation = nn.Identity()
        # Make AVG pooling input shape independent
        self.model.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.model(x)
        return x

    def predict(self, x):
        x = self.forward(x)
        return self.activation(x)



def train_resnet(config):
    logger = Logger(config.log_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # device = "cpu"

    model = ResNet(num_classes=1)
    train_data = get_loader(
        config.image_dir,
        None,
        None,
        config.crop_size,
        config.image_size,
        config.batch_size,
        "PCam",
        "train",
        config.num_workers,
        config.in_memory,
    )
    val_data = get_loader(
        config.image_dir,
        None,
        None,
        config.crop_size,
        config.image_size,
        config.batch_size,
        "PCam",
        "val",
        config.num_workers,
        config.in_memory,
    )

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
        # Training
        for batch_idx, (x, y) in enumerate(train_data, 1):
            model.train()
            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)
            with torch.cuda.amp.autocast():
                output = model(x).to(device)
                # class_pred = output >= 0.5
                # correct_classifications += class_pred.eq(y).sum()
                loss = loss_function(output, y).to(device)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pred = nn.functional.softmax(output) >= 0.5
            running_loss += loss.item()
            correct_classifications += pred.eq(y).sum().item()
            if config.verbose:
                n_samples = batch_idx * config.batch_size
                current_loss = running_loss / batch_idx
                current_acc = correct_classifications / n_samples
                tqdm.write(
                        f"Training batch {batch_idx}/{len(train_data)} Current loss: {current_loss:.4f} Current accuracy: {current_acc:.4f}"
                )
        train_loss = running_loss / len(train_data)
        train_acc = correct_classifications / len(train_data.dataset)

        # Validating
        with torch.no_grad():
            model.eval()
            val_loss = 0
            correct_classifications = 0
            for batch_idx, (x, y) in tqdm(enumerate(val_data, 1), desc="Validating", disable=not config.progress_bar):
                x = x.to(device)
                y = y.to(device)
                with torch.cuda.amp.autocast():
                    output = model(x).to(device)
                    pred = nn.functional.softmax(output) >= 0.5
                    correct_classifications += pred.eq(y).sum().item()
                    loss = loss_function(output, y).to(device)
                val_loss += loss.item()
                if config.verbose:
                    n_samples = batch_idx * config.batch_size
                    cur_val_loss = val_loss / batch_idx
                    cur_val_acc = correct_classifications / n_samples
                    tqdm.write(
                            f"Validating batch {batch_idx}/{len(val_data)} Current loss: {cur_val_loss:.4f} Current accuracy: {cur_val_acc:.4f}"
                    )
            val_loss = val_loss / len(val_data)
            val_acc = correct_classifications / len(val_data.dataset)

        # Logging
        loss_log["train_loss"] = train_loss
        loss_log["train_acc"] = train_acc
        loss_log["val_loss"] = val_loss
        loss_log["val_acc"] = val_acc

        tqdm.write(
            f"Epoch: {epoch} Training loss: {train_loss:.4f} Training accuracy: {train_acc:.4f} Validation loss: {val_loss:.4f} Validation accuracy: {val_acc:.4f}"
        )

        if config.use_tensorboard:
            for tag, value in loss_log.items():
                logger.scalar_summary(tag, value, epoch)

        # Early stopping
        if config.use_early_stopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                tests_since_best = 0
                best_model_save_path = os.path.join(config.model_save_dir, "pcam_resnet_best.ckpt")
                torch.save(model.state_dict(), best_model_save_path)
            else:
                tests_since_best += 1
                if tests_since_best > config.patience:
                    tqdm.write(
                        f"Reached early stopping threshold with patience {config.patience}."
                    )
                    break
    model_save_path = os.path.join(config.model_save_dir, "pcam_resnet.ckpt")
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
    parser.add_argument("--dataset", type=str, default="PCam", choices=["PCam"])
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
    parser.add_argument("--use_early_stopping", type=str2bool, default=True)
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="patience for early stopping measured in validation loss tracking",
    )

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
    parser.add_argument("--model_save_dir", type=str, default="pcam/resnet/models")

    config = parser.parse_args()
    main(config)
