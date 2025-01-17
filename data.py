import numpy as np
import pickle
import os
import torch

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, index):
        item = self.data[index]
        #item = item.unsqueeze(0)
        return item
    def __len__(self):
        return len(self.data)

def prepare_data(config):
    dataset_normalization_params = {
        'cifar10': {'shift': -120.63838, 'scale': 1. / 64.16736},
        'imagenet32': {'shift': -110.95837, 'scale': 1. / 64.82333},
        # mean and sd of custom dataset was calculated using preprocessing_custom_dataset.py
        'flower32' : {'shift': -100.11472258, 'scale': 1. / 66.41444},
    }
    loss_params = {'shift_loss': -127.5, 'scale_loss': 1. / 127.5}

    if config.dataset not in dataset_normalization_params:
        print("Error")
    if config.dataset == 'imagenet32':
        train_data, val_data, test_data = load_filtered_imagenet32(config.data_root)
        config.image_size, config.image_channels = 32, 3
    elif config.dataset == 'cifar10':
        train_data, val_data, test_data = load_cifar10(config.data_root)
        config.image_size, config.image_channels = 32, 3
    elif config.dataset == 'flower32':
        train_data, val_data, test_data = load_flower32(config.data_root)
        config.image_size, config.image_channels = 32, 3
    else:
        print("Error on dataset loading")

    if config.test_eval:
        print("test dataset is validation set")
        eval_data = test_data
    else:
        print("seperate test set")
        eval_data = val_data
    shift, scale = map_to_tensor(dataset_normalization_params[config.dataset])
    shift_loss, scale_loss = map_to_tensor(loss_params)
    train_dataset = CustomDataset(torch.as_tensor(train_data))
    eval_dataset = CustomDataset(torch.as_tensor(eval_data))
    transpose_required = False

    def preprocess_function(data_batch):
        nonlocal shift, scale, shift_loss, scale_loss, transpose_required
        inputs = data_batch[0].to(torch.float32).cuda(non_blocking=True)
        outputs = inputs.clone()

        if transpose_required:
            inputs = inputs.permute(0, 2, 3, 1)

        inputs.add_(shift).mul_(scale)
        outputs.add_(shift_loss).mul_(scale_loss)
        return inputs, outputs

    return config, train_dataset, eval_dataset, preprocess_function

def map_to_tensor(params):
    return_value = {value: torch.tensor([value]).cuda().view(1, 1, 1, 1) for key, value in params.items()}
    return return_value

def mkdir_from_path(path):
    os.makedirs(path, exist_ok=True)

def load_flower32(root, test_size=0.1, val_size=0.1, random_state=42):
    ## This is the custom dataset we used (not tested in original paper)
    # ========================================================================
    images = []

    for file_name in os.listdir(root + '102flowers/processed_png'):
        if file_name.endswith('.png'):
            # Load the image
            image_path = os.path.join(root + '102flowers/processed_png', file_name)
            image = Image.open(image_path).convert("RGB") 
            image_array = np.array(image, dtype=np.uint8)
            images.append(image_array)

    images = np.array(images)
    train_data, test_data = train_test_split(images, test_size=test_size, random_state=random_state)
    train_data, val_data = train_test_split(train_data, test_size=val_size, random_state=random_state)
    return train_data, val_data, test_data

def load_filtered_imagenet32(root, test_size=0.1, val_size=0.1, random_state=42):
    ## This is the filtered imagenet32 dataset (10,000 images)
    # ========================================================================
    images = []

    for file_name in os.listdir(root + 'imagenet_filtered/processed'):
        if file_name.endswith('.png'):
            # Load the image
            image_path = os.path.join(root + 'imagenet_filtered/processed', file_name)
            image = Image.open(image_path).convert("RGB") 
            image_array = np.array(image, dtype=np.uint8)
            images.append(image_array)

    images = np.array(images)
    train_data, test_data = train_test_split(images, test_size=test_size, random_state=random_state)
    train_data, val_data = train_test_split(train_data, test_size=val_size, random_state=random_state)
    return train_data, val_data, test_data

def load_cifar10(root):
    def unpickle(file):
        with open(file, 'rb') as f:
            return pickle.load(f, encoding='bytes')

    train_batches = [unpickle(os.path.join(root, 'cifar-10-batches-py', f'data_batch_{i}')) for i in range(1, 6)]
    train_data = np.vstack([batch[b'data'] for batch in train_batches])
    test_data = unpickle(os.path.join(root, 'cifar-10-batches-py', 'test_batch'))[b'data']

    train_data = train_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    test_data = test_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    train_data, val_data = train_test_split(train_data, test_size=5000, random_state=42)
    print(len(train_data), len(val_data), len(test_data))
    return train_data, val_data, test_data

