import os, sys
sys.path.append(os.getcwd())
import click
import time
import functools
import pdb

import numpy as np
from pathlib import Path

import torch
import torchvision
import torch.nn as nn
mse_criterion = nn.MSELoss()

from torch import optim
from torchvision import transforms


from models.wgan import *
from training_utils import *
import libs as lib
import libs.plot
from tensorboardX import SummaryWriter

# to fix png loading
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from timeit import default_timer as timer


@click.command()
@click.option('--train_dir', default=None, help='Data path for training')
@click.option('--validation_dir', default=None, help='Data path for valication')
@click.option('--image_data_type', default="image_folder", type=click.Choice(["lsun", "image_folder", "CIFAR10"]), help='If you are using lsun images from lsun lmdb, use lsun. If you use your own data in a folder, then use "image_folder". If you use lmdb, you\'ll need to write the loader by yourself. Please check load_data function')
@click.option('--output_path', default=None, help='Output path where result (.e.g drawing images, cost, chart) will be stored')
@click.option('--dim', default=32, help='Model dimensionality or image resolution, tested with 32.')
@click.option('--critic_iters', default=5, help='How many iterations to train the critic/disciminator for')
@click.option('--gen_iters', default=1, help='How many iterations to train the gemerator for')
@click.option('--batch_size', default=64, help='Training batch size. Must be a multiple of number of gpus')
@click.option('--noisy_label_prob', default=0., help='Make the labels the noisy for the discriminator: occasionally flip the labels when training the discriminator')
@click.option('--epochs', default=35, help='Number of epochs to train for')
@click.option('--gp_lambda', default=10, help='Gradient penalty lambda hyperparameter')
@click.option('--num_workers', default=5, help='Number of workers to load data')
@click.option('--saving_step', default=200, help='Save model, sample every this saving step')
@click.option('--training_class', default=None, help='A list of classes, separated by comma ",". IGNORE this if you are NOT training on lsun, or if you want to train on other classes of lsun, then change it accordingly')
@click.option('--val_class', default=None, help='A list of classes, separated by comma ",". IGNORE this if you are NOT training on lsun, or if you want to train on other classes of lsun, then change it accordingly')
@click.option('--restore_mode/--no-restore_mode', default=False, help="If True, it will load saved model from OUT_PATH and continue to train")
@click.option('--lambda_adv', default=0.1)


def train(train_dir, validation_dir, image_data_type, output_path, dim, critic_iters, gen_iters, batch_size, noisy_label_prob, epochs, gp_lambda, num_workers, saving_step, training_class, val_class, restore_mode, lambda_adv):

    if train_dir is None or len(train_dir) == 0:
        raise Exception('Please specify path to data directory in gan.py!')

    output_path = Path(output_path)
    sample_path = output_path / "samples"
    mkdir_path(sample_path)
    if isinstance(training_class, str):
        training_class = training_class.split(",")
    if isinstance(val_class, str):
        val_class = val_class.split(",")

    data_transform = transforms.Compose([
        transforms.Resize(dim),
        transforms.RandomCrop(dim),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    one = torch.tensor(1, dtype=torch.float).to(device)
    mone = one * -1

    if restore_mode:
        aG = GoodGenerator(dim)
        aD = GoodDiscriminator(dim)
        g_state_dict = torch.load(str(output_path / "generator.pt"))
        aG.load_state_dict(remove_module_str_in_state_dict(g_state_dict))
        d_state_dict = torch.load(str(output_path / "discriminator.pt"))
        aD.load_state_dict(remove_module_str_in_state_dict(d_state_dict))
    else:
        aG = GoodGenerator(dim)
        aD = GoodDiscriminator(dim)
        aG.apply(weights_init)
        aD.apply(weights_init)

    optimizer_g = torch.optim.Adam(aG.parameters(), lr=1e-3, betas=(0.5,0.9))
    optimizer_d = torch.optim.Adam(aD.parameters(), lr=1e-4, betas=(0.5,0.9))

    scheduler_g = torch.optim.lr_scheduler.StepLR(optimizer_g, step_size=10, gamma=0.5)
    scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_d, step_size=10, gamma=0.5)

    aG = torch.nn.DataParallel(aG).to(device)
    aD = torch.nn.DataParallel(aD).to(device)

    writer = SummaryWriter()
    dataloader = load_data(image_data_type,
                           train_dir,
                           data_transform,
                           batch_size=batch_size,
                           classes=training_class,
                           num_workers=num_workers,
                           add_noise=True,
                           noise_std=0.1)

    total_batches = len(dataloader)
    global_iter = 0

    for epoch in range(epochs):
        start_time = time.time()
        for batch_idx, (noisy_img, clean_img, *_) in enumerate(dataloader):
            #---------------------TRAIN G------------------------
            for p in aD.parameters():
                p.requires_grad_(False)  # freeze D

            gen_cost = None
            for _ in range(gen_iters):
                aG.zero_grad()
                noisy_img = noisy_img.to(device)
                clean_img = clean_img.to(device)
                fake_data = aG(noisy_img)
                # Content loss (MSE)
                mse_loss = mse_criterion(fake_data, clean_img)
                # Adversarial loss (WGAN)
                adv_loss = -aD(fake_data).mean()
                # Total generator loss
                gen_cost = mse_loss + lambda_adv * adv_loss
                gen_cost.backward()
                optimizer_g.step()

            #---------------------TRAIN D------------------------
            for p in aD.parameters():
                p.requires_grad_(True)
            for i in range(critic_iters):
                aD.zero_grad()
                with torch.no_grad():
                    fake_data = aG(noisy_img).detach()
                    real_data = clean_img
                is_flipping = False
                if noisy_label_prob > 0 and noisy_label_prob < 1:
                    is_flipping = np.random.randint(1//noisy_label_prob, size=1)[0] == 1

                if not is_flipping:
                    disc_real = aD(real_data)
                    disc_real = disc_real.mean()
                    disc_fake = aD(fake_data)
                    disc_fake = disc_fake.mean()
                else:
                    disc_real = aD(fake_data)
                    disc_real = disc_real.mean()
                    disc_fake = aD(real_data)
                    disc_fake = disc_fake.mean()

                gradient_penalty = calc_gradient_penalty(aD, real_data, fake_data, batch_size, dim, device, gp_lambda)
                disc_cost = disc_fake - disc_real + gradient_penalty
                disc_cost.backward()
                w_dist = disc_fake - disc_real
                optimizer_d.step()

                # Logging for D
                if i == critic_iters-1:
                    writer.add_scalar('data/disc_cost', disc_cost, global_iter)
                    writer.add_scalar('data/gradient_pen', gradient_penalty, global_iter)

            # Logging for G
            writer.add_scalar('data/gen_cost', gen_cost, global_iter)
            writer.add_scalar('data/gen_mse_loss', mse_loss, global_iter)
            writer.add_scalar('data/gen_adv_loss', adv_loss, global_iter)

            lib.plot.plot(str(output_path / 'time'), time.time() - start_time)
            lib.plot.plot(str(output_path / 'train_disc_cost'), disc_cost.cpu().data.numpy())
            lib.plot.plot(str(output_path / 'train_gen_cost'), gen_cost.cpu().data.numpy())
            lib.plot.plot(str(output_path / 'wasserstein_distance'), w_dist.cpu().data.numpy())

            # Validation, sample, and save
            if global_iter > 0 and global_iter % saving_step == 0:
                val_loader = load_data(
                    image_data_type,
                    validation_dir,
                    data_transform,
                    batch_size=batch_size,
                    classes=val_class,
                    num_workers=num_workers
                )
                dev_disc_costs = []
                for _, images in enumerate(val_loader):
                    imgs = torch.Tensor(images[0])
                    imgs = imgs.to(device)
                    with torch.no_grad():
                        imgs_v = imgs

                    D = aD(imgs_v)
                    _dev_disc_cost = -D.mean().cpu().data.numpy()
                    dev_disc_costs.append(_dev_disc_cost)
                lib.plot.plot(str(output_path / 'dev_disc_cost.png'), np.mean(dev_disc_costs))
                lib.plot.flush()
                # Use a batch of real noisy images for visualization
                with torch.no_grad():
                    gen_images = aG(noisy_img)
                torchvision.utils.save_image(gen_images, str(sample_path / 'samples_{}.png').format(global_iter), nrow=8, padding=2)
                grid_images = torchvision.utils.make_grid(gen_images, nrow=8, padding=2)
                writer.add_image('images', grid_images, global_iter)
                torch.save(aG.state_dict(), str(output_path / "generator.pt"))
                torch.save(aD.state_dict(), str(output_path / "discriminator.pt"))
            lib.plot.tick()
            global_iter += 1

            # Print progress every 10% of the epoch
            if (batch_idx + 1) % max(1, (len(dataloader) // 10)) == 0:
                percent = int(100 * (batch_idx + 1) / len(dataloader))
                print(f"Epoch [{epoch+1}/{epochs}] - {percent}% complete ({batch_idx+1}/{len(dataloader)})")
        
        scheduler_g.step()
        scheduler_d.step()

        print(f"Epoch [{epoch+1}/{epochs}] completed in {time.time() - start_time:.2f}s")

if __name__ == '__main__':
    train()