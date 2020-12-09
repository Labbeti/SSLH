#!/bin/sh

# This script is an alternative to the "environment.yml" installation.
# You must run this in the "SSL/" folder.

conda create -n env_ssl python=3.8.5
conda activate env_ssl

conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install librosa -c conda-forge
conda install scikit-image tensorboard pandas tqdm h5py

pip install git+https://github.com/leocances/augmentation_utils.git
pip install git+https://github.com/leocances/pytorch_metrics.git
pip install git+https://github.com/leocances/UrbanSound8K.git@new_data_management
pip install git+https://github.com/Labbeti/Torch-AudioSet.git

pip install -e .

mkdir -p results/models results/tensorboard
