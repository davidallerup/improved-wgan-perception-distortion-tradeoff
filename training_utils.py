from pathlib import Path
from collections import OrderedDict

import gpustat
import torch
from torch import autograd
import torch.nn.init as init
from torchvision import transforms, datasets
from torch.utils.data import Dataset
from torch.autograd import grad
from functools import partial

from models.wgan import *

def mkdir_path(path):
    path.mkdir(parents=True, exist_ok=True)

def showMemoryUsage(device=1):
    gpu_stats = gpustat.GPUStatCollection.new_query()
    item = gpu_stats.jsonify()["gpus"][device]
    print('Used/total: ' + "{}/{}".format(item["memory.used"], item["memory.total"]))


def weights_init(m):
    if isinstance(m, MyConvo2d): 
        if m.conv.weight is not None:
            if m.he_init:
                init.kaiming_uniform_(m.conv.weight)
            else:
                init.xavier_uniform_(m.conv.weight)
        if m.conv.bias is not None:
            init.constant_(m.conv.bias, 0.0)
    if isinstance(m, nn.Linear):
        if m.weight is not None:
            init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0.0)


def remove_module_str_in_state_dict(state_dict):
    state_dict_rename = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "") # remove `module.`
        state_dict_rename[name] = v
    return state_dict_rename

class NoisyImageDataset(Dataset):
    def __init__(self, clean_dataset, noise_fn):
        self.clean_dataset = clean_dataset
        self.noise_fn = noise_fn

    def __len__(self):
        return len(self.clean_dataset)

    def __getitem__(self, idx):
        clean_img, *rest = self.clean_dataset[idx]
        noisy_img = self.noise_fn(clean_img)
        # If the dataset returns (img, label), keep label as well
        if rest:
            return noisy_img, clean_img, rest[0]
        else:
            return noisy_img, clean_img

# Example noise function (Gaussian noise)
def add_gaussian_noise(img, std=0.1):
    return img + torch.randn_like(img) * std

# Usage in your load_data function:
def load_data(image_data_type, path_to_folder, data_transform, batch_size, classes=None, num_workers=5, train=True, add_noise=False, noise_std=0.1):
    torch.set_num_threads(1)
    if image_data_type == 'lsun':
        dataset = datasets.LSUN(path_to_folder, classes=classes, transform=data_transform)
    elif image_data_type == "image_folder":
        dataset = datasets.ImageFolder(root=path_to_folder, transform=data_transform)
    elif image_data_type == "CIFAR10":
        dataset = datasets.CIFAR10(root=path_to_folder, train=train, download=False, transform=data_transform) # TODO: Set this to True on first run
    else:
        raise ValueError("Invalid image data type")
    if add_noise:
        dataset = NoisyImageDataset(dataset, partial(add_gaussian_noise, std=noise_std))
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=True)
    return dataset_loader


def generate_image(netG, dim, batch_size, noise=None):
    if noise is None:
        noise = gen_rand_noise()

    with torch.no_grad():
        noisev = noise 
    samples = netG(noisev)
    samples = samples.view(batch_size, 3, dim, dim)
    samples = samples * 0.5 + 0.5
    return samples

def gen_rand_noise(batch_size, ):
    noise = torch.randn(batch_size, 128)
    return noise


def calc_gradient_penalty(netD, real_data, fake_data, batch_size, dim, device, gp_lambda):
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolates = alpha * real_data.detach() + (1 - alpha) * fake_data.detach()
    interpolates.requires_grad_(True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(
        outputs=disc_interpolates, 
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gp_lambda
    return gradient_penalty




