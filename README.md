# Deep Semi-Supervised Learning with Holistic methods (SSLH)

Unofficial PyTorch and PyTorch-Lightning implementations of Deep Semi-Supervised Learning methods for audio tagging.

There is 4 SSL methods :
- [FixMatch (FM)](https://arxiv.org/pdf/2001.07685.pdf)
- [MixMatch (MM)](https://arxiv.org/pdf/1905.02249.pdf)
- [ReMixMatch (RMM)](https://arxiv.org/pdf/1911.09785.pdf)
- [Unsupervised Data Augmentation (UDA)](https://arxiv.org/pdf/1904.12848.pdf)

For the following datasets :
- [CIFAR-10 (CIFAR10)](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)
- [ESC-10 (ESC10)](https://www.karolpiczak.com/papers/Piczak2015-ESC-Dataset.pdf)
- [Google Speech Commands (GSC)](https://arxiv.org/pdf/1804.03209.pdf)
- [Primate Vocalization Corpus (PVC)](https://arxiv.org/pdf/2101.10390.pdf)
- [UrbanSound8k (UBS8K)](http://www.justinsalamon.com/uploads/4/3/9/4/4394963/salamon_urbansound_acmmm14.pdf)

[comment]: <> (- [AudioSet &#40;ADS&#41;]&#40;https://static.googleusercontent.com/media/research.google.com/fr//pubs/archive/45857.pdf&#41;)
[comment]: <> (- [FSD50K]&#40;&#41;)

With 3 models :
- [WideResNet28 (WRN28)](https://arxiv.org/pdf/1605.07146.pdf)
- [MobileNetV1 (MNV1)](https://arxiv.org/pdf/1704.04861.pdf)
- [MobileNetV2 (MNV2)](https://arxiv.org/pdf/1801.04381.pdf)

The implementation of Mean Teacher (MT), Deep Co-Training (DCT) and Pseudo-Labeling (PL) are present in this repository but not fully tested.


## Installation
#### Download & setup
- Clone the repository :
```bash
git clone https://github.com/Labbeti/SSLH
```
- Set up the package in your environment :
```bash
cd SSLH
pip install -e .
```

The installation is now finished.

#### Alternatives
The project contains also a ```environment.yaml``` and ```requirements.txt``` for installing the packages respectively with conda or pip :
- With **conda** environment file :
```bash
conda env create -n env_sslh -f environment.yaml
conda activate env_sslh
pip install -e . --no-dependencies
```

- With **pip** requirements file :
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
The main scripts available are in ```standalone``` directory :
```
standalone
├── deep_co_training.py
├── fixmatch.py
├── mean_teacher.py
├── mixmatch.py
├── mixup.py
├── pseudo_labeling.py
├── remixmatch.py
├── supervised.py
└── uda.py
```

The code use Hydra for parsing args. The syntax of setting an argument is "name=value" instead of "--name value".

Example 1 : MixMatch on ESC10
```bash
python mixmatch.py data=esc10
```

Example 2 : Supervised+Weak on GSC
```bash
python supervised.py data=gsc expt.augm_train=weak bsize=256 epochs=300
```

Example 3 : FixMatch+MixUp on UBS8K
```bash
python fixmatch.py data=ubs8K expt=fixmatch_mixup bsize_s=128 bsize_u=128 epochs=300
```
(note: default folds used for UBS8K are in "config/data/ubs8k.yaml")

Example 4 : ReMixMatch on CIFAR-10
```bash
python remixmatch.py data=cifar10 model.n_input_channels=3
```

## List of main arguments

| Name | Description | Values | Default |
| --- | --- | --- | --- |
| data | Dataset used | ads, cifar10, esc10, fsd50k, gsc, pvc, ubs8k | esc10 |
| expt | Training method (experiment) used | *(depends of the python script, see the filenames in config/expt/ folder)* | *(depends of the python script)* |
| model | Pytorch model to use | mobilenetv1, mobilenetv2, vgg, wideresnet28 | wideresnet28 |
| optim | Optimizer used | adam, sgd | adam |
| sched | Learning rate scheduler | cosine, softcosine, none | softcosine |
| epochs | Number of training epochs | int | 1 |
| bsize | Batch size in SUP methods | int | 60 |
| ratio | Ratio of the training data used in SUP methods | float in [0, 1] | 1.0 |
| bsize_s | Batch size of supervised part in SSL methods | int | 30 |
| bsize_u | Batch size of unsupervised part in SSL methods | int | 30 |
| ratio_s | Ratio of the supervised training data used in SSL methods | float in [0, 1] | 0.1 |
| ratio_u | Ratio of the unsupervised training data used in SSL methods | float in [0, 1] | 0.9 |


## SSLH Package overview
```
sslh
├── callbacks
├── datamodules
│     ├── supervised
│     └── semi_supervised
├── datasets
├── expt
│     ├── deep_co_training
│     ├── fixmatch
│     ├── mean_teacher
│     ├── mixmatch
│     ├── mixup
│     ├── pseudo_labeling
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
  - For AudioSet, ESC10, GSC, PVC and UBS8K datasets base code.
- Qiuqiang Kong (qiuqiangkong on Github)
  - For MobileNetV1 & V2 model implementation from [PANN](https://github.com/qiuqiangkong/audioset_tagging_cnn).

## Additional notes
- This project has been made with Ubuntu 20.04 and Python 3.8.5.

## Glossary
| Acronym | Description |
| --- | --- |
| activation | Activation function |
| ADS | AudioSet |
| aug, augm, augment | Augmentation |
| ce | Cross-Entropy |
| expt | Experiment |
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
| SUP | Supervised Learning |
| _u | Unsupervised |
| UBS8K | UrbanSound8K dataset |
