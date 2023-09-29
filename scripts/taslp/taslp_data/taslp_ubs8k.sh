#!/bin/sh

if [ ! "$1" = "run" ] && [ ! "$1" = "run_olympe" ] && [ ! "$1" = "run_osirim" ]; then
  echo "Usage: $0 (run|run_olympe|run_osirim)"
  exit 1
fi

dpath_scripts=`realpath $0 | xargs dirname | xargs dirname`
run="$1"

epochs=300
activation="softmax"
gpus=1
cpus=4
accelerator="dp"
verbose=true
debug=false

dataset="ubs8k"

criterion="CE"
bsize=256
bsize_s=128
bsize_u=128

model="wideresnet28"
model_acro="wrn28"

sup_params="expt.criterion=${criterion} bsize=${bsize} model=${model}"
ssl_params="expt.criterion_s=${criterion} expt.criterion_u=${criterion} bsize_s=${bsize_s} bsize_u=${bsize_u}"

for fold_val in 1 2 3 4 5 6 7 8 9 10
do

tag_prefix="_taslp_${dataset}_fold_${fold_val}_${model_acro}"
dataset_params="dataset=${dataset} dataset.val_folds=${fold_val}"
common_params="${run} ${dataset_params} epochs=${epochs} expt.activation=${activation} gpus=${gpus} cpus=${cpus} trainer.accelerator=${accelerator} verbose=${verbose} debug=${debug}"

# Supervised 10%
tag="${tag_prefix}_SUP10"
$dpath_scripts/supervised.sh $common_params $sup_params expt=supervised ratio=0.1 tag=$tag
tag="${tag_prefix}_SUP10m"
$dpath_scripts/mixup.sh $common_params $sup_params expt=supervised ratio=0.1 tag=$tag

# Supervised 100%
tag="${tag_prefix}_SUP100"
$dpath_scripts/supervised.sh $common_params $sup_params expt=supervised ratio=1.0 tag=$tag
tag="${tag_prefix}_SUP100m"
$dpath_scripts/mixup.sh $common_params $sup_params expt=supervised ratio=1.0 tag=$tag

# Supervised 10% + Weak
tag="${tag_prefix}_SUP10w"
$dpath_scripts/supervised.sh $common_params $sup_params expt=supervised ratio=0.1 tag=$tag expt.augm_train=weak
tag="${tag_prefix}_SUP10wm"
$dpath_scripts/mixup.sh $common_params $sup_params expt=supervised ratio=0.1 tag=$tag expt.augm_train=weak

# Supervised 100% + Weak
tag="${tag_prefix}_SUP100w"
$dpath_scripts/supervised.sh $common_params $sup_params expt=supervised ratio=1.0 tag=$tag expt.augm_train=weak
tag="${tag_prefix}_SUP100wm"
$dpath_scripts/mixup.sh $common_params $sup_params expt=supervised ratio=1.0 tag=$tag expt.augm_train=weak

# Supervised 10% + Strong
tag="${tag_prefix}_SUP10s"
$dpath_scripts/supervised.sh $common_params $sup_params expt=supervised ratio=0.1 tag=$tag expt.augm_train=strong
tag="${tag_prefix}_SUP10sm"
$dpath_scripts/mixup.sh $common_params $sup_params expt=supervised ratio=0.1 tag=$tag expt.augm_train=strong

# Supervised 100% + Strong
tag="${tag_prefix}_SUP100s"
$dpath_scripts/supervised.sh $common_params $sup_params expt=supervised ratio=1.0 tag=$tag expt.augm_train=strong
tag="${tag_prefix}_SUP100sm"
$dpath_scripts/mixup.sh $common_params $sup_params expt=supervised ratio=1.0 tag=$tag expt.augm_train=strong

# FixMatch
fm_params="model=${model}"
tag="${tag_prefix}_FM"
$dpath_scripts/fixmatch.sh $common_params $ssl_params $fm_params tag=$tag expt=fixmatch
tag="${tag_prefix}_FMM"
$dpath_scripts/fixmatch.sh $common_params $ssl_params $fm_params tag=$tag expt=fixmatch_mixup

# MixMatch
mm_params="model=${model}"
tag="${tag_prefix}_MM"
$dpath_scripts/mixmatch.sh $common_params $ssl_params $mm_params tag=$tag expt=mixmatch
tag="${tag_prefix}_MMN"
$dpath_scripts/mixmatch.sh $common_params $ssl_params $mm_params tag=$tag expt=mixmatch_nomixup

# ReMixMatch
rmm_params="model=${model}_rot"
tag="${tag_prefix}_RMM"
$dpath_scripts/remixmatch.sh $common_params $ssl_params $rmm_params tag=$tag expt=remixmatch
tag="${tag_prefix}_RMMN"
$dpath_scripts/remixmatch.sh $common_params $ssl_params $rmm_params tag=$tag expt=remixmatch_nomixup

# UDA
uda_params="model=${model}"
tag="${tag_prefix}_UDA"
$dpath_scripts/uda.sh $common_params $ssl_params $uda_params tag=$tag expt=uda
tag="${tag_prefix}_UDAM"
$dpath_scripts/uda.sh $common_params $ssl_params $uda_params tag=$tag expt=uda_mixup

done
