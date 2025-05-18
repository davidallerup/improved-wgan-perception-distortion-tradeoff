# Initialize venv
poetry install --no-root

# Install packages
Install packages with/without GPU from either of the requirements files

# Run training on CIFAR10

py train.py --image_data_type="CIFAR10" --train_dir data/  --validation_dir data/ --output_path output/ --dim 32 --saving_step 300 --num_workers 8

Make sure that CIFAR10 is downloaded on the first run
dataset = datasets.CIFAR10(root=path_to_folder, train=train, download=False, transform=data_transform) # TODO: Set this to True on 

# Acknowledgements

* [igul222/improved_wgan_training](https://github.com/igul222/improved_wgan_training)
* [caogang/wgan-gp](https://github.com/caogang/wgan-gp)
* [ACGAN-PyTorch](https://github.com/clvrai/ACGAN-PyTorch)
* [LayerNorm](https://github.com/pytorch/pytorch/issues/1959)
