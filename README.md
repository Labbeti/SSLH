# Semi Supervised Learning with Holistic methods (SSLH)

Holistic methods for Semi-Supervised Learning (FixMatch, MixMatch, ReMixMatch and UDA) for AudioSet (ADS), CIFAR-10, ESC-10, GoogleSpeechCommands (GSC), Primate Vocalize Corpus (PVC) and UrbanSound8k (UBS8K) datasets.

## Installation
#### Prerequisites
- Make sure you have access to the following repositories : 
  - For UrbanSound8k dataset : https://github.com/leocances/UrbanSound8K (created by Léo Cances)
  - For utility functions and classes : https://github.com/Labbeti/MLU

#### Download & setup
- Clone the repository :
```bash
git clone https://github.com/Labbeti/SSLH
```
- Set up the package in your environment (passwords can be required during installation) :
```bash
cd SSLH
pip install -e .
```
- Create the results folders :
```bash
./build_directories.sh
```

The installation is now finished.

#### Alternatives
The project contains also a ```environment.yaml``` and ```requirements.txt``` for installing the packages respectively with conda or pip :
- With **conda** environment :
```bash
conda create -n env_sslh -f environment.yaml
conda activate env_sslh
pip install -e . --no-dependencies
```

- With **pip** environment :
```bash
pip install -r requirements.txt
pip install -e . --no-dependencies
```

## Datasets
CIFAR10, ESC10 and GoogleSpeechCommands are automatically downloaded and installed.
For UrbanSound8k, please read the [README of leocances](https://github.com/leocances/UrbanSound8K/blob/master/README.md#prepare-the-dataset), in section "Prepare the dataset". 
AudioSet (ADS) and Primate Vocalize Corpus (PVC) cannot be installed automatically by now.

[comment]: <> (TODO : For Audioset install !)
[comment]: <> (TODO : For PVC install !)

## Usage
The main scripts available are in directory ```standalone``` :
```
standalone
├── fixmatch.py
├── mixmatch.py
├── mixup.py
├── remixmatch.py
├── supervised.py
└── uda.py
```

The code use Hydra for parsing args. The syntax of an argument is "name=value" instead of "--name value".

Example 1 : MixMatch on ESC10
```bash
python mixmatch.py dataset=esc10
```

Example 2 : Supervised+Weak on GSC
```bash
python supervised.py dataset=gsc experiment.augm_train=weak bsize=256 epochs=300
```

Example 3 : FixMatch+MixUp on UBS8K
```bash
python fixmatch.py dataset=ubs8K dataset.root="../data/UBS8K" experiment=fixmatch_mixup bsize_s=128 bsize_u=128 epochs=300
```
(note: default folds used are in "config/dataset/ubs8k.yaml")

## Package overview
```
sslh
├── callbacks
├── datamodules
│     ├── fully_supervised
│     ├── partial_supervised
│     └── semi_supervised
├── datasets
├── experiments
│     ├── fixmatch
│     ├── mixmatch
│     ├── mixup
│     ├── remixmatch
│     ├── supervised
│     └── uda
├── metrics
├── models
├── transforms
│     ├── augments
│     ├── pools
│     └── self_transforms
└── utils
```

## Authors
This repository has been created by Etienne Labbé (Labbeti on Github).

It contains also some code from the following authors :
- Léo Cancès (leocances on github)
  - For AudioSet, ESC10, GSC and UBS8K datasets and samplers.
- Qiuqiang Kong (qiuqiangkong on Github)
  - For MobileNetV1 & V2 model implementation from [PANN](https://github.com/qiuqiangkong/audioset_tagging_cnn)

## Additional notes
- This project has been made with Ubuntu 20.04 and Python 3.8.5.

## Glossary
| Acronym | Description |
| --- | --- |
| activation | Activation function |
| ADS | AudioSet dataset |
| aug, augm, augment | Augmentation |
| ce | Cross-Entropy |
| exp | Experimental |
| fm | FixMatch |
| fn, func | Function |
| GSC | Google Speech Commands dataset (with 35 classes) |
| GSC12 | Google Speech Commands dataset (with 10 classes from GSC, 1 unknown class and 1 silence class) |
| hparams | Hyperparameters |
| js | Jensen-Shannon |
| kl | Kullback-Leibler |
| loc | Localisation |
| lr | Learning Rate |
| mm | MixMatch |
| mse | Mean Squared Error |
| pred | Prediction |
| PVC | Primate Vocalize Corpus dataset |
| rmm | ReMixMatch |
| _s | Supervised |
| sched | Scheduler |
| SSL | Semi-Supervised Learning |
| _u | Unsupervised |
| UBS8K | UrbanSound8K dataset |
