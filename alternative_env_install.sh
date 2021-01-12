#!/bin/sh

# This script is an alternative to the "environment.yml" installation.
# You must run this in the "SSLH/" folder.

conda create -n env_sslh python=3.8.5
conda activate env_sslh

conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install librosa -c conda-forge
conda install scikit-image tensorboard pandas tqdm h5py

pip install git+https://github.com/leocances/augmentation_utils
pip install git+https://github.com/leocances/pytorch_metrics
pip install git+https://github.com/leocances/UrbanSound8K@new_data_management
pip install git+https://github.com/Labbeti/MLU@dev

pip install -e .

mkdir -p results/models results/tensorboard
