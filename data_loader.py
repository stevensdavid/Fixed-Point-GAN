from torch.utils import data
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random
from glob import glob
import h5py
import numpy as np


class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, attr_path, selected_attrs, transform, mode, eval_dataset, match_distribution, subsample_offset):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.match_distribution = match_distribution
        self.subsample_offset = subsample_offset
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        random.seed(1234)
        random.shuffle(lines)
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')

            if (i+1) < 2000:
                self.test_dataset.append([filename, label])
            else:
                self.train_dataset.append([filename, label])
        if self.match_distribution:
            # Match the distribution of the first label in test and training set 
            # and the dataset size.
            test_labels = [l[0] for _, l in self.test_dataset]
            num_zeroes = len([l for l in test_labels if l == 0])
            num_ones = len([l for l in test_labels if l == 0])
            train_zeroes = [[f, l] for f, l in self.train_dataset if l[0] == 0]
            train_ones = [[f, l] for f, l in self.train_dataset if l[0] == 1]
            self.train_dataset = train_zeroes[num_zeroes*self.subsample_offset:num_zeroes*(self.subsample_offset + 1)] + train_ones[num_ones*self.subsample_offset:num_ones*(self.subsample_offset + 1)]
            random.shuffle(self.train_dataset)
        print('Finished preprocessing the CelebA dataset...')

    def get_labels(self):
        """Return all labels"""
        dataset = self.train_dataset if self.mode == "train" else self.test_dataset
        # Only use first attribute, which is assumed to be glasses.
        labels = np.asarray([y[0] for _, y in dataset], dtype=int)
        return labels

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        """Return the number of images."""
        return self.num_images


class BRATS_SYN(data.Dataset):
    """Dataset class for the BRATS dataset."""

    def __init__(self, image_dir, transform, mode):
        """Initialize and Load the BRATS dataset."""
        self.image_dir = image_dir
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.load_data()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def load_data(self):
        """Load BRATS dataset"""
        
        # Load test dataset
        test_neg = glob(os.path.join(self.image_dir, 'test', 'negative', '*jpg'))
        test_pos = glob(os.path.join(self.image_dir, 'test', 'positive', '*jpg'))

        for filename in test_neg:
            self.test_dataset.append([filename, [0]])

        for filename in test_pos:
            self.test_dataset.append([filename, [1]])


        # Load train dataset
        train_neg = glob(os.path.join(self.image_dir, 'train', 'negative', '*jpg'))
        train_pos = glob(os.path.join(self.image_dir, 'train', 'positive', '*jpg'))

        for filename in train_neg:
            self.train_dataset.append([filename, [0]])

        for filename in train_pos:
            self.train_dataset.append([filename, [1]])

        print('Finished loading the BRATS dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]
        image = Image.open(filename)
        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        """Return the number of images."""
        return self.num_images


class PCam(data.Dataset):
    """Dataset class for the PatchCamelyon dataset."""

    def __init__(self, image_dir, transform, mode, in_memory):
        """Initialize and Load the PCam dataset."""
        if mode not in ("train", "test", "val"):
            raise ValueError(f"Support modes are train, test and val, received: {mode}")
        self.image_dir = image_dir
        self.transform = transform
        self.mode = mode
        self.x = None
        self.y = None
        self.preloaded = in_memory
        self.num_images = None
        self.load_data()

    def load_data(self):
        """Load PCam dataset"""
        if self.mode == "train":
            x_file = "train_x.h5"
            y_file = "train_y.h5"
        elif self.mode == "val":
            x_file = "valid_x.h5"
            y_file = "valid_y.h5"
        elif self.mode == "test":
            x_file = "test_x.h5"
            y_file = "test_y.h5"
        else:
            raise ValueError(f"Support modes are train, test and val, received: {mode}")
        if self.preloaded:
            print(f"Loading {self.mode}_x")
            self.x = [self.transform(Image.fromarray(x.astype('uint8'), 'RGB')) for x in h5py.File(os.path.join(self.image_dir, x_file), 'r', swmr=True)['x']]
            print(f"Loading {self.mode}_y")
            self.y = [torch.from_numpy(y.flatten()).float() for y in h5py.File(os.path.join(self.image_dir, y_file), 'r', swmr=True)['y']]
            self.num_images = len(self.x)
        else:
            self.x = os.path.join(self.image_dir, x_file)
            x = h5py.File(os.path.join(self.image_dir, x_file), 'r', swmr=True)['x']
            self.num_images = len(x)
            self.y = os.path.join(self.image_dir, y_file)
        # TODO: Pre-loading the files here would be faster, but I have not been able
        #       to make it work as the h5py objects cannot be pickled and can
        #       therefore not be used in PyTorch. This can probably be fixed, but I
        #       have not been able to succeed with this despite an afternoon's worth
        #       of attempts...
        # self.x = h5py.File(os.path.join(self.image_dir, x_file), 'r', swmr=True)['x']
        # self.y = h5py.File(os.path.join(self.image_dir, y_file), 'r', swmr=True)['y']
        print('Finished loading the PCam dataset...')

    def get_labels(self):
        """Return all labels"""
        y = h5py.File(self.y, 'r', swmr=True)['y']
        return y.flatten()

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        if self.preloaded:
            x = self.x[index]
            y = self.y[index]
            return self.x[index], self.y[index]
        x = h5py.File(self.x, 'r', swmr=True)['x']
        y = h5py.File(self.y, 'r', swmr=True)['y']
        img = Image.fromarray(x[index, ...].astype('uint8'), 'RGB')
        return self.transform(img), torch.from_numpy(y[index, ...].flatten()).float()

    def __len__(self):
        """Return the number of images."""
        return self.num_images


def get_loader(image_dir, attr_path, selected_attrs, crop_size=178, image_size=128, 
               batch_size=16, dataset='CelebA', mode='train', num_workers=1, eval_dataset = None, in_memory=False, weighted=False, augment=None, match_distribution=False, subsample_offset=None):
    if eval_dataset is None:
        eval_dataset = mode
    """Build and return a data loader."""
    transform = []
    if augment is None:
        augment = mode == "train"
    if augment:
        transform.append(T.RandomHorizontalFlip())
    transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    if dataset == 'CelebA':
        dataset = CelebA(image_dir, attr_path, selected_attrs, transform, mode, eval_dataset, match_distribution, subsample_offset)
    elif dataset == 'BRATS':
        dataset = BRATS_SYN(image_dir, transform, mode)
    elif dataset == 'PCam':
        dataset = PCam(image_dir, transform, mode, in_memory)
    
    elif dataset == 'Directory':
        dataset = ImageFolder(image_dir, transform)
    if weighted:
        labels = dataset.get_labels()
        # invert prevalence to sample evenly
        class_weights = 1 / np.asarray([np.sum(labels == 0)/len(labels), np.sum(labels == 1)/len(labels)])
        sample_weights = class_weights[labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
    else:
        sampler = None
    data_loader = data.DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                shuffle=(mode=='train' and sampler is None),
                                num_workers=num_workers,
                                sampler=sampler)
    return data_loader

if __name__ == "__main__":
    PCam("data/pcam", [], "test")
