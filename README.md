<!-- # -*- coding: utf-8 -*- -->

<div align="center">

# Deep Semi-Supervised Learning with Holistic methods (SSLH)

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.9+-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 1.7.1-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge&labelColor=gray"></a>

Unofficial PyTorch and PyTorch-Lightning implementations of Deep Semi-Supervised Learning methods for audio tagging.

</div>

There is 4 SSL methods :
- [FixMatch (FM)](https://arxiv.org/pdf/2001.07685.pdf) [1]
- [MixMatch (MM)](https://arxiv.org/pdf/1905.02249.pdf) [2]
- [ReMixMatch (RMM)](https://arxiv.org/pdf/1911.09785.pdf) [3]
- [Unsupervised Data Augmentation (UDA)](https://arxiv.org/pdf/1904.12848.pdf) [4]

For the following datasets :
- [CIFAR-10 (CIFAR10)](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)
- [ESC-10 (ESC10)](https://www.karolpiczak.com/papers/Piczak2015-ESC-Dataset.pdf)
- [Google Speech Commands (GSC)](https://arxiv.org/pdf/1804.03209.pdf)
- [Primate Vocalization Corpus (PVC)](https://arxiv.org/pdf/2101.10390.pdf)
- [UrbanSound8k (UBS8K)](http://www.justinsalamon.com/uploads/4/3/9/4/4394963/salamon_urbansound_acmmm14.pdf)

With 3 models :
- [WideResNet28 (WRN28)](https://arxiv.org/pdf/1605.07146.pdf)
- [MobileNetV1 (MNV1)](https://arxiv.org/pdf/1704.04861.pdf)
- [MobileNetV2 (MNV2)](https://arxiv.org/pdf/1801.04381.pdf)

**IMPORTANT NOTE: The implementation of Mean Teacher (MT), Deep Co-Training (DCT) and Pseudo-Labeling (PL) are present in this repository but not fully tested.**

You can find a more stable version of MT and DCT at https://github.com/lcances/semi-supervised.
The datasets AudioSet and FSD50K are in beta testing.
If you meet problems, you can contact me at 


## Installation
#### Download & setup
```bash
git clone https://github.com/Labbeti/SSLH
conda env create -n env_sslh -f environment.yaml
conda activate env_sslh
pip install -e SSLH --no-dependencies
```

#### Alternatives
- As python package :
```bash
pip install https://github.com/Labbeti/SSLH
```
The dependencies will be automatically installed with pip instead of conda, which means the the build versions can be slightly different.

The project contains also a ```environment.yaml``` and ```requirements.txt``` for installing the packages respectively with conda or pip.
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
CIFAR10, ESC10, GoogleSpeechCommands and FSD50K can be downloaded and installed.
For UrbanSound8k, please read the [README of leocances](https://github.com/leocances/UrbanSound8K/blob/master/README.md#prepare-the-dataset), in section "Prepare the dataset".
AudioSet (ADS) and Primate Vocalize Corpus (PVC) cannot be installed automatically by now.

To download a dataset, you can use the `data.download=true` option.

[comment]: <> (TODO : For Audioset install !)
[comment]: <> (TODO : For PVC install !)

## Usage
This code use Hydra for parsing args. The syntax of setting an argument is "name=value" instead of "--name value".

Example 1 : MixMatch on ESC10
```bash
python -m sslh.mixmatch data=ssl_esc10 data.dm.download=true
```

Example 2 : Supervised+Weak on GSC
```bash
python -m sslh.supervised data=sup_gsc aug@train_aug=weak data.dm.bsize=256 epochs=300 data.dm.download=true
```

Example 3 : FixMatch+MixUp on UBS8K
```bash
python -m sslh.fixmatch data=ssl_ubs8K pl=fixmatch_mixup data.dm.bsize_s=128 data.dm.bsize_u=128 epochs=300 data.dm.download=true
```

Example 4 : ReMixMatch on CIFAR-10
```bash
python -m sslh.remixmatch data=ssl_cifar10 model.n_input_channels=3 aug@weak_aug=img_weak aug@strong_aug=img_strong data.dm.download=true
```

## List of main arguments

| Name | Description | Values | Default |
| --- | --- | --- | --- |
| data | Dataset used | (sup|ssl)_(ads|cifar10|esc10|fsd50k|gsc|pvc|ubs8k) | (sup|ssl)_esc10 |
| pl | Pytorch Lightning training method (experiment) used | *(depends of the python script, see the filenames in config/pl/ folder)* | *(depends of the python script)* |
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
├── pl_modules
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
│     ├── get
│     ├── image
│     ├── other
│     ├── pools
│     ├── self_transforms
│     ├── spectrogram
│     └── waveform
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

## References

[1] K. Sohn, D. Berthelot, C.-L. Li, Z. Zhang, N. Carlini, E. D. Cubuk, A. Ku-
rakin, H. Zhang, and C. Raffel, “FixMatch: Simplifying Semi-Supervised
Learning with Consistency and Confidence,” p. 21.

[2] D. Berthelot, N. Carlini, I. Goodfellow, N. Papernot, A. Oliver, and
C. Raffel, “MixMatch: A Holistic Approach to Semi-Supervised Learning,”
Oct. 2019, number: arXiv:1905.02249 arXiv:1905.02249 [cs, stat]. [Online].
Available: http://arxiv.org/abs/1905.02249

[3] D. Berthelot, N. Carlini, E. D. Cubuk, A. Kurakin, K. Sohn,
H. Zhang, and C. Raffel, “ReMixMatch: Semi-Supervised Learning
with Distribution Alignment and Augmentation Anchoring,” Feb. 2020,
number: arXiv:1911.09785 arXiv:1911.09785 [cs, stat]. [Online]. Available:
http://arxiv.org/abs/1911.09785

[4] Q. Xie, Z. Dai, E. Hovy, M.-T. Luong, and Q. V. Le, “Unsu-
pervised Data Augmentation for Consistency Training,” Nov. 2020,
number: arXiv:1904.12848 arXiv:1904.12848 [cs, stat]. [Online]. Available:
http://arxiv.org/abs/1904.12848

<!-- Cances, L., Labbé, E. & Pellegrini, T. Comparison of semi-supervised deep learning algorithms for audio classification. J AUDIO SPEECH MUSIC PROC. 2022, 23 (2022). https://doi.org/10.1186/s13636-022-00255-6 -->

## Cite this repository
If you use this code, you can cite the following paper associated :
```
@article{cances_comparison_2022,
	title = {Comparison of semi-supervised deep learning algorithms for audio classification},
	volume = {2022},
	issn = {1687-4722},
	url = {https://doi.org/10.1186/s13636-022-00255-6},
	doi = {10.1186/s13636-022-00255-6},
	abstract = {In this article, we adapted five recent SSL methods to the task of audio classification. The first two methods, namely Deep Co-Training (DCT) and Mean Teacher (MT), involve two collaborative neural networks. The three other algorithms, called MixMatch (MM), ReMixMatch (RMM), and FixMatch (FM), are single-model methods that rely primarily on data augmentation strategies. Using the Wide-ResNet-28-2 architecture in all our experiments, 10\% of labeled data and the remaining 90\% as unlabeled data for training, we first compare the error rates of the five methods on three standard benchmark audio datasets: Environmental Sound Classification (ESC-10), UrbanSound8K (UBS8K), and Google Speech Commands (GSC). In all but one cases, MM, RMM, and FM outperformed MT and DCT significantly, MM and RMM being the best methods in most experiments. On UBS8K and GSC, MM achieved 18.02\% and 3.25\% error rate (ER), respectively, outperforming models trained with 100\% of the available labeled data, which reached 23.29\% and 4.94\%, respectively. RMM achieved the best results on ESC-10 (12.00\% ER), followed by FM which reached 13.33\%. Second, we explored adding the mixup augmentation, used in MM and RMM, to DCT, MT, and FM. In almost all cases, mixup brought consistent gains. For instance, on GSC, FM reached 4.44\% and 3.31\% ER without and with mixup. Our PyTorch code will be made available upon paper acceptance at https://github.com/Labbeti/SSLH.},
	number = {1},
	journal = {EURASIP Journal on Audio, Speech, and Music Processing},
	author = {Cances, Léo and Labbé, Etienne and Pellegrini, Thomas},
	month = sep,
	year = {2022},
	pages = {23},
}
```

## Contact
- Etienne Labbé "Labbeti" (maintainer) : labbeti.pub@gmail.com
