# Ignore futurewarnings from using old version of tensorboard
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from collections import namedtuple
import os
from tqdm import tqdm
from logger import Logger
from numpy import inf
import torch.optim as optim
import torch.nn as nn
from main import str2bool
from argparse import ArgumentParser
from data_loader import get_loader
from torchvision.models.resnet import resnet50
import torch
from solver import Solver


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


def _transform_batch(generator, x, y, device, op):
    if generator is None:
        raise ValueError("Must have provided generator")
    if op == "id":
        y_trg = y
    elif op == "tilde":
        # Only flip first attribute. Assumed to be glasses.
        y_trg = y.clone().to(device)
        y_trg[:, 0] = (~y_trg[:, 0].bool()).float()
    elif op == "random":
        y_trg = y.clone().to(device)
        distribution = torch.distributions.bernoulli.Bernoulli(
            0.5*torch.ones_like(y_trg[:, 0]))
        y_trg[:, 0] = distribution.sample()
    else:
        raise ValueError("Invalid generator_op")
    x_out = generator.transform_batch(x, y_trg)
    return x_out, y_trg


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
        config.train_on,
        config.num_workers,
        config.in_memory,
        weighted=config.dataset == "CelebA",
        augment=config.generator_iters is None,
        match_distribution=True,
        subsample_offset=0,
    )
    val_data = get_loader(
        config.image_dir,
        config.attr_path,
        config.selected_attrs,
        config.crop_size,
        config.image_size,
        config.batch_size,
        config.dataset,
        config.early_stopping_split,
        config.num_workers,
        config.in_memory,
        weighted=config.dataset == "CelebA",
        augment=config.generator_iters is not None,
        match_distribution=True,
        subsample_offset=1,
    )

    if config.generator_iters is None:
        generator = None
    else:
        config = add_missing_solver_args(config)
        generator = Solver(train_data, config, train_mode=False)
        generator.restore_model(config.generator_iters)
        # Train on generated images from validation set.
        # This is not optimal, but better than training on
        # the training set.

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
                x, y = _transform_batch(generator, x, y, device, config.generator_op)
            model.train()
            optimizer.zero_grad()
            x = x.to(device)
            if y.shape[1] > 1:
                # Only extract first attribute.
                y = y[:, 0:1]
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
                current_sensitivity = correct_positive / \
                    max(n_positive_samples, 1)
                current_specificity = correct_negative / \
                    max(n_negative_samples, 1)
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
                    x, y = _transform_batch(generator, x, y, device, config.generator_op)
                x = x.to(device)
                if y.shape[1] > 1:
                    # Only extract first attribute.
                    y = y[:, 0:1]
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
                    current_sensitivity = correct_positive / \
                        max(n_positive_samples, 1)
                    current_specificity = correct_negative / \
                        max(n_negative_samples, 1)
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
                best_model_save_path = os.path.join(
                    config.resnet_save_dir, f"{config.dataset}_resnet_best.ckpt")
                torch.save(model.state_dict(), best_model_save_path)
            else:
                tests_since_best += 1
                if tests_since_best >= config.patience:
                    tqdm.write(
                        f"Reached early stopping threshold with patience {config.patience}."
                    )
                    break
    if not config.use_early_stopping:
        model_save_path = os.path.join(
            config.resnet_save_dir, f"{config.dataset}_resnet.ckpt")
        torch.save(model.state_dict(), model_save_path)
        print(f"Saved model into path {model_save_path}")


def resnet_accuracy(model: ResNet, dataset, device, generator=None, generator_op=None, progress_bar=True) -> dict:
    tp = tn = fp = fn = 0
    n_correct = 0
    n_total = 0
    for x, y in tqdm(dataset, total=len(dataset), disable=not progress_bar):
        x = x.to(device)
        y = y.to(device)
        if generator is not None:
            x, y = _transform_batch(generator, x, y, device, generator_op)
        if y.shape[1] > 1:
            y = y[:, 0:1]
        x = x.to(device)
        y = y.to(device)
        with torch.cuda.amp.autocast():
            output = model(x).to(device)
        pred = output >= 0.5
        # tricky bitmagic, but I am almost confident it works
        # EXAMPLE:
        # >>> y
        # tensor([1, 0, 0, 1])
        # >>> pred
        # tensor([1, 0, 1, 0])
        # >>> ~pred.eq(y) # This is the elements that were misclassified
        # tensor([False, False,  True,  True])
        # >>> (~pred.eq(y)).logical_and(1-y) # Elements that are misclassified as 1
        # tensor([False, False,  True, False])
        # >>> (~pred.eq(y)).logical_and(y) # Elements that are misclassified and 0
        # tensor([False, False, False,  True])
        result = pred.eq(y)
        true_positive = result.logical_and(y).sum().item()
        true_negative = result.logical_and(1-y).sum().item()
        false_positive = (~result).logical_and(1-y).sum().item()
        false_negative = (~result).logical_and(y).sum().item()
        tp += true_positive
        tn += true_negative
        fp += false_positive
        fn += false_negative
        n_correct += result.sum().item()
        n_total += y.shape[0]
    accuracy = n_correct / n_total
    sensitivity = tp / float(tp + fn)
    specificity = tn / float(tn + fp)
    return {
        "acc": round(accuracy, 4),
        "TPR": round(sensitivity, 4),
        "TNR": round(specificity, 4),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "n_correct": n_correct,
        "n_total": n_total
    }


def evaluate_resnet(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        # Counters for (true|false) (positives|negatives)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    dataset = get_loader(
        config.image_dir,
        config.attr_path,
        config.selected_attrs,
        config.crop_size,
        config.image_size,
        config.batch_size,
        config.dataset,
        config.eval_split,
        config.num_workers,
        config.in_memory,
        weighted=False,
        augment=False,
        match_distribution=True,
        subsample_offset=0,
    )
    resnet_path = os.path.join(config.resnet_save_dir, config.resnet_filename)
    resnet = ResNet(num_classes=1)
    resnet.load_state_dict(torch.load(resnet_path), strict=False)
    resnet.eval()
    resnet.to(device)
    if config.generator_iters is None:
        generator = None
    else:
        config = add_missing_solver_args(config)
        generator = Solver(dataset, config, train_mode=False)
        generator.restore_model(config.generator_iters)
    results = resnet_accuracy(
        resnet, dataset, device, generator, config.generator_op, config.eval_progress_bar
    )
    print(results)


def main(config):
    if config.mode == "train":
        train_resnet(config)
    elif config.mode == "evaluate":
        evaluate_resnet(config)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--c_dim", type=int, default=5, help="dimension of domain labels (1st dataset)"
    )
    parser.add_argument(
        "--crop_size", type=int, default=178, help="crop size for the images"
    )
    parser.add_argument("--image_size", type=int,
                        default=128, help="image resolution")

    # Training configuration.
    parser.add_argument("--in_memory", action="store_true",
                        help="Store dataset in RAM")
    parser.add_argument("--dataset", type=str,
                        default="PCam", choices=["PCam", "CelebA"])
    parser.add_argument("--batch_size", type=int,
                        default=16, help="mini-batch size")
    parser.add_argument(
        "--epochs", type=int, default=200000, help="number of total epochs for training"
    )
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="learning rate")
    parser.add_argument(
        "--beta1", type=float, default=0.5, help="beta1 for Adam optimizer"
    )
    parser.add_argument(
        "--beta2", type=float, default=0.999, help="beta2 for Adam optimizer"
    )
    parser.add_argument("--use_early_stopping",
                        action="store_true", default=True)
    parser.add_argument(
        "--patience",
        type=int,
        default=2,
        help="patience for early stopping measured in validation loss tracking",
    )
    parser.add_argument('--selected_attrs', '--list', nargs='+',
                        help='selected attributes for the CelebA dataset', default=None)
    parser.add_argument("--attr_path", type=str, default=None)

    parser.add_argument("--early_stopping_split", type=str, choices=["train", "val"])
    # Evaluation configuration
    parser.add_argument("--eval_split", type=str, choices=["train", "val", "test"])
    parser.add_argument(
        "--resnet_filename",
        type=str,
        help="File name (not path) of resnet to evaluate. Directory is given by resnet_save_dir"
    )

    # Miscellaneous.
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument(
        "--mode", type=str, default="train", choices=["train", "evaluate"]
    )
    parser.add_argument("--use_tensorboard", action="store_true")
    parser.add_argument(
        "--verbose", action="store_true", help="Print loss after each batch"
    )
    parser.add_argument("--progress_bar", action="store_true",
                        help="Print progress bars")

    # Directories.
    parser.add_argument("--image_dir", type=str, default="data/pcam")
    parser.add_argument("--log_dir", type=str, default="pcam/resnet/logs")
    parser.add_argument("--resnet_save_dir", type=str,
                        default="pcam/resnet/models")
    parser.add_argument(
        "--model_save_dir",
        type=str,
        default="pcam/models",
        help="Directory where GAN models are saved"
    )
    parser.add_argument("--train_on", type=str,
                        default="train", choices=["train", "val"])

    parser.add_argument(
        "--generator_iters",
        type=int,
        help="Train on generated data from a model restored form the supplied iteration."
    )
    parser.add_argument("--generator_op", type=str, choices=[
                        "id", "tilde", "random"], help="The generator transformation to apply.")
    parser.add_argument("--eval_progress_bar", action="store_true")

    config = parser.parse_args()
    main(config)
