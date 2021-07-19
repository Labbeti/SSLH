#!/bin/sh

if [ ! "$1" = "run" ] && [ ! "$1" = "run_olympe" ] && [ ! "$1" = "run_osirim" ]; then
  echo "Usage: $0 (run|run_olympe|run_osirim)"
  exit 1
fi

dpath_scripts="."
run="$1"
dataset="ads"
train_subset="unbalanced"
n_train_steps=125000
sampler_s_balanced=true
val_check_interval=1000
epochs=1
activation="sigmoid"
gpus=2
cpus=10
accelerator="dp"
verbose=true

model="mobilenetv2"
model_acronym="mnv2"

# FM
threshold=0.75
threshold_guess=0.75
# MM
temperature=1.0

criterion="BCE"
bsize=256
bsize_s=128
bsize_u=128

rd="${RANDOM}"
tag_prefix="_taslp_${dataset}_${model_acronym}_${rd}"

dataset_params="dataset=${dataset} dataset.train_subset=${train_subset} dataset.n_train_steps=${n_train_steps} dataset.sampler_s_balanced=${sampler_s_balanced} dataset.val_check_interval=${val_check_interval}"
common_params="${run} ${dataset_params} epochs=${epochs} expt.activation=${activation} gpus=${gpus} cpus=${cpus} trainer.accelerator=${accelerator} verbose=${verbose}"

sup_params="expt.criterion=${criterion} bsize=${bsize} model=wideresnet28"
ssl_params="expt.criterion_s=${criterion} expt.criterion_u=${criterion} bsize_s=${bsize_s} bsize_u=${bsize_u}"

# Supervised 10%
#tag="${tag_prefix}_SUP10"
#$dpath_scripts/supervised.sh $common_params $sup_params expt=supervised ratio=0.1 tag=$tag
#tag="${tag_prefix}_SUP10m"
#$dpath_scripts/mixup.sh $common_params $sup_params expt=supervised ratio=0.1 tag=$tag
#
## Supervised 100%
#tag="${tag_prefix}_SUP100"
#$dpath_scripts/supervised.sh $common_params $sup_params expt=supervised ratio=1.0 tag=$tag
#tag="${tag_prefix}_SUP100m"
#$dpath_scripts/mixup.sh $common_params $sup_params expt=supervised ratio=1.0 tag=$tag
#
## Supervised 10% + Weak
#tag="${tag_prefix}_SUP10w"
#$dpath_scripts/supervised.sh $common_params $sup_params expt=supervised ratio=0.1 tag=$tag expt.augm_train=weak
#tag="${tag_prefix}_SUP10wm"
#$dpath_scripts/mixup.sh $common_params $sup_params expt=supervised ratio=0.1 tag=$tag expt.augm_train=weak
#
## Supervised 100% + Weak
#tag="${tag_prefix}_SUP100w"
#$dpath_scripts/supervised.sh $common_params $sup_params expt=supervised ratio=1.0 tag=$tag expt.augm_train=weak
#tag="${tag_prefix}_SUP100wm"
#$dpath_scripts/mixup.sh $common_params $sup_params expt=supervised ratio=1.0 tag=$tag expt.augm_train=weak
#
## Supervised 10% + Strong
#tag="${tag_prefix}_SUP10s"
#$dpath_scripts/supervised.sh $common_params $sup_params expt=supervised ratio=0.1 tag=$tag expt.augm_train=strong
#tag="${tag_prefix}_SUP10sm"
#$dpath_scripts/mixup.sh $common_params $sup_params expt=supervised ratio=0.1 tag=$tag expt.augm_train=strong
#
## Supervised 100% + Strong
#tag="${tag_prefix}_SUP100s"
#$dpath_scripts/supervised.sh $common_params $sup_params expt=supervised ratio=1.0 tag=$tag expt.augm_train=strong
#tag="${tag_prefix}_SUP100sm"
#$dpath_scripts/mixup.sh $common_params $sup_params expt=supervised ratio=1.0 tag=$tag expt.augm_train=strong

## FixMatch
#fm_params="model=wideresnet28 expt.threshold=${threshold} expt.threshold_guess=${threshold_guess}"
#tag="${tag_prefix}_FM"
#$dpath_scripts/fixmatch.sh $common_params $ssl_params $fm_params tag=$tag expt=fixmatch
#tag="${tag_prefix}_FMM"
#$dpath_scripts/fixmatch.sh $common_params $ssl_params $fm_params tag=$tag expt=fixmatch_mixup

# MixMatch
mm_params="model=wideresnet28 temperature=${temperature}"
tag="${tag_prefix}_MM"
$dpath_scripts/mixmatch.sh $common_params $ssl_params $mm_params tag=$tag expt=mixmatch
tag="${tag_prefix}_MMN"
$dpath_scripts/mixmatch.sh $common_params $ssl_params $mm_params tag=$tag expt=mixmatch_nomixup

## ReMixMatch
#rmm_params="model=wideresnet28_rot"
#tag="${tag_prefix}_RMM"
#$dpath_scripts/remixmatch.sh $common_params $ssl_params $rmm_params tag=$tag expt=remixmatch
#tag="${tag_prefix}_RMMN"
#$dpath_scripts/remixmatch.sh $common_params $ssl_params $rmm_params tag=$tag expt=remixmatch_nomixup
#
## UDA
#uda_params="model=${model}"
#tag="${tag_prefix}_UDA"
#$dpath_scripts/uda.sh $common_params $ssl_params $uda_params tag=$tag expt=uda
#tag="${tag_prefix}_UDAM"
#$dpath_scripts/uda.sh $common_params $ssl_params $uda_params tag=$tag expt=uda_mixup
