# Semi Supervised Learning with Holistic methods (SSLH)

Holistic methods for Semi-Supervised Learning : MixMatch, ReMixMatch and FixMatch for ESC-10, UrbanSound8k and GoogleSpeechCommands datasets.

[comment]: <>  (TODO : CIFAR-10 still need to be fixed)

## Installation
#### Prerequisites
- This project has been made with Ubuntu 20.04 and Python 3.8.5.
- This installation requires **Anaconda** to manage the Python environment. (version used: 4.8.5) 
- This project has been made for running with GPU but not with CPU.
- Make sure you have access to the following repositories : 
  - For UrbanSound8k dataset : https://github.com/leocances/UrbanSound8K (created by Léo Cances)
  - For AudioSet dataset : https://github.com/Labbeti/Torch-AudioSet
  - For utility functions and classes : https://github.com/Labbeti/MLU

#### Download & setup
- Clone the repository :
```bash
git clone https://github.com/Labbeti/SSL
```
- Create a conda environment with the YAML file (passwords can be required during installation) :
```bash
cd SSL
conda env create -f environment.yml
```
- Activate the new environment :
```bash
conda activate env_ssl
```
- Setup the main repository :
```bash
pip install -e .
```
- Create the results folders :
```bash
mkdir -p results/models results/tensorboard
```

## Datasets
CIFAR10, ESC10 and GoogleSpeechCommands are automatically download and installed.
For UrbanSound8k, please read the [README of leocances](https://github.com/leocances/UrbanSound8K/blob/master/README.md#prepare-the-dataset), in section "Prepare the dataset". 

## Usage
The main scripts available are in folder ```standalone/``` :
- ```supervised.py```
- ```mixmatch.py```
- ```remixmatch.py```
- ```fixmatch.py```

Example :
```
cd standalone/
python mixmatch.py --dataset ESC10 --dataset_path ../dataset --lr 3e-3
```

Every script contains all the initialisation needed for start training and the train code and losses is in ```sslh/``` folder.

The ```uda.py``` training is currently unstable.
The suffix "_exp" indicate that the file contains experimental code for running variants.

[comment]: <>  (Mettre uda.py quand il marchera)

## Notebook
You can find a mixmatch fast code example in a notebook [standalone/mixmatch.ipynb](https://github.com/Labbeti/SSL/blob/master/standalone/mixmatch.ipynb).

## Results

#### Categorical Accuracies (%)
[comment]: <> (TODO)
| Dataset | Supervised 10% | Supervised 100% | MixMatch | FixMatch | Supervised MixUp 10% | Supervised MixUp 100% | FixMatch + MixUp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ESC10 (cross-validation) | 62.78 | 92.33 | 59.44 | 64.44 | 63.56 | 92.67 | 63.11 |
| UBS8K (cross-validation) | 67.13 | 76.36 | 73.22 | 69.58 | 68.69 | 77.39 | 75.33 |
| GSC (35 classes, evaluation) | 89.68 | 94.43 | 92.69 | 83.88 | 92.29 | 96.97 | 93.21 |
| GSC10 (10 classes + unknown and silence classes, evaluation) | # | # | # | # | # | # | # |

## Code overview 
- ```sslh``` :
    - ```augments``` : Folder of augmentations added and utilities functions for building and managing augments pools.
    - ```datasets``` : Folder for which contains dataset classes and dataset interfaces for building datasets.
    - ```fixmatch``` : FixMatch training methods.
    - ```mixmatch``` : MixMatch training methods.
    - ```models``` : Models classes available and utilities functions for saving best models in file (Checkpoint).
    - ```remixmatch``` : ReMixMatch training methods.
    - ```supervised``` : Supervised training methods.
    - ```uda``` : UDA training methods.
    - ```utils``` : Main utilities functions for training methods.
    - ```validation``` : Validation and tests classes for inference.
- ```standalone``` :
    - Main scripts for running a method on a specific dataset.

## Glossary
| Acronym | Word |
| --- | --- |
| ABC | Abstract Class |
| activation | Activation Function |
| aug, augm, augment | Augmentation |
| ce | Cross-Entropy |
| ds | Dataset |
| exp | Experimental |
| fm | FixMatch |
| fn, func | Function |
| GSC | Google Speech Commands (with 35 classes) |
| GSC10 | Google Speech Commands (with 10 classes from GSC, 1 unknown class and 1 silence class) |
| hparams | Hyperparameters |
| js | Jensen-Shannon |
| kl | Kullback-Leibler |
| loc | Localisation |
| lr | Learning Rate |
| mm | MixMatch |
| mse | Mean Squared Error |
| optim | Optimizer |
| pred | Prediction |
| rmm | ReMixMatch |
| s | Supervised |
| sched | Scheduler |
| u | Unsupervised |
| UBS8K | UrbanSound8K |
