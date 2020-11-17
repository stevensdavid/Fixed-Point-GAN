from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random
from glob import glob
import h5py
import numpy as np
from torchvision.utils import save_image


class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, attr_path, selected_attrs, transform, mode, all_data, isGlasses=None):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.mode = mode
        self.all_data = all_data
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.label_attr2idx = {}
        self.idx2attr = {}
        self.isGlasses = isGlasses
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
        
        # very odd that they seem to shuffle the attribute file, 
        # now the filenames will not appear in order when you fetch them.
        # actually, since the filename is paired with the label, this is 
        # perfectly fine -- it just shuffles the input in a reproducible 
        # manner (see __getitem__, which returns the matching image for a 
        # label).
        random.seed(1234) 
        random.shuffle(lines)
        eyeglasses = 0.
        noneyeglasses = 0.
        total = 0.
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                self.label_attr2idx[attr_name] = len(label)
                label.append(values[idx] == '1')

                # if in test data and is eyeglasses and exists eyeglasses
                if (i+1) < 2000 and attr_name == "Eyeglasses":
                  total = total + 1
                  if values[idx] == '1':
                    eyeglasses = eyeglasses + 1
                  else:
                    noneyeglasses = noneyeglasses + 1
                  
            if (i+1) < 2000:
                # if eyeglasses exist, only choose them for generation
                if self.isGlasses is None:
                  self.test_dataset.append([filename, label])
                else:
                  if(label[self.label_attr2idx["Eyeglasses"]] == self.isGlasses):
                    self.test_dataset.append([filename, label])
            else:
                if self.all_data == True:
                  self.test_dataset.append([filename, label])
                self.train_dataset.append([filename, label])
        print("#Eyeglasses = " + str(eyeglasses))
        print("#Noneyeglasses = " + str(noneyeglasses))
        print("#total = " + str(total))
        print("(#eyeglasses+#noneyeglasses)/#total = " + str((eyeglasses + noneyeglasses) / total))
        print("#eyeglasses/#total = " + str(eyeglasses / total))
        # dataset has format [['pathtoimg', [attr1Boolean, attr2Boolean]], anotherinput, ...]
        print(self.isGlasses)
        if self.isGlasses is not None:
          data = np.array([(1 if x[1][self.label_attr2idx["Eyeglasses"]] == self.isGlasses else 0) for x in self.test_dataset])
          print(str(len(data[data == 1])))
        # print("eyeglass total calced from slicing: " + "hi")
        
        print('Finished preprocessing the CelebA dataset...')

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

    def __init__(self, image_dir, transform, mode):
        """Initialize and Load the PCam dataset."""
        if mode not in ("train", "test", "val"):
            raise ValueError(f"Support modes are train, test and val, received: {mode}")
        self.image_dir = image_dir
        self.transform = transform
        self.mode = mode
        self.x_file = None
        self.y_file = None
        self.num_images = None
        self.load_data()

    def load_data(self):
        """Load PCam dataset"""
        if self.mode == "train":
            x_file = "train_x.h5"
            y_file = "train_y.h5"
        elif self.mode == "val":
            x_file = "val_x.h5"
            y_file = "val_y.h5"
        elif self.mode == "test":
            x_file = "test_x.h5"
            y_file = "test_y.h5"
        else:
            raise ValueError(f"Support modes are train, test and val, received: {mode}")
        self.x_file = os.path.join(self.image_dir, x_file)
        x = h5py.File(os.path.join(self.image_dir, x_file), 'r', swmr=True)['x']
        self.num_images = len(x)
        self.y_file = os.path.join(self.image_dir, y_file)
        # TODO: Pre-loading the files here would be faster, but I have not been able
        #       to make it work as the h5py objects cannot be pickled and can
        #       therefore not be used in PyTorch. This can probably be fixed, but I
        #       have not been able to succeed with this despite an afternoon's worth
        #       of attempts...
        # self.x = h5py.File(os.path.join(self.image_dir, x_file), 'r', swmr=True)['x']
        # self.y = h5py.File(os.path.join(self.image_dir, y_file), 'r', swmr=True)['y']
        print('Finished loading the PCam dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        x = h5py.File(self.x_file, 'r', swmr=True)['x']
        y = h5py.File(self.y_file, 'r', swmr=True)['y']
        img = Image.fromarray(x[index, ...].astype('uint8'), 'RGB')
        return self.transform(img), torch.from_numpy(y[index, ...].flatten()).float()

    def __len__(self):
        """Return the number of images."""
        return self.num_images


def get_loader(image_dir, attr_path, selected_attrs, crop_size=178, image_size=128, 
               batch_size=16, dataset='CelebA', mode='train', all_data=False, num_workers=1,
               normalize=True, isGlasses=None):
    """Build and return a data loader."""
    transform = []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())
    transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    
    if normalize:
      transform.append(T.ToTensor())
      transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    else:
      class ToTensorWithoutScaling(object):
        """H x W x C -> C x H x W"""
        def __call__(self, image):
          return torch.ByteTensor(np.array(image)).permute(2, 0, 1).float()
      transform.append(ToTensorWithoutScaling())
    transform = T.Compose(transform)

    if dataset == 'CelebA':
        dataset = CelebA(image_dir, attr_path, selected_attrs, transform, mode, all_data, isGlasses)
    elif dataset == 'BRATS':
        dataset = BRATS_SYN(image_dir, transform, mode)
    elif dataset == 'PCam':
        dataset = PCam(image_dir, transform, mode)
    
    elif dataset == 'Directory':
        dataset = ImageFolder(image_dir, transform)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    return data_loader


def dump_images_to_dir(dl: data.DataLoader, dirname: str):
    data_iter = iter(dl)
    idx = 0
    for images, _ in data_iter:
        for image in images:
            save_image(image, fp=os.path.join(dirname, f"{idx}.jpg"), nrow=1, padding=0, normalize=True)
            idx += 1

if __name__ == "__main__":
    PCam("data/pcam", [], "test")
