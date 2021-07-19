#!/bin/sh

if [ ! "$1" = "run" ] && [ ! "$1" = "run_olympe" ] && [ ! "$1" = "run_osirim" ]; then
  echo "Usage: $0 (run|run_olympe|run_osirim)"
  exit 1
fi

dpath_scripts="."
run="$1"
dataset="ads"
train_subset="balanced" # "unbalanced"
n_train_steps=10000 # null, 125000
sampler_s_balanced=true # false
pre_computed_specs=false
val_check_interval=500 # 0.25
epochs=1 # 10
activation="sigmoid"
gpus=1 # 2
cpus=4 # 10
accelerator=null # "dp"
verbose=true
debug=true

model="mobilenetv2"

# FM
threshold=0.75
threshold_guess=0.75
# MM
temperature=1.0
sharpen_threshold=0.5

criterion="BCE"
bsize=128
bsize_s=64
bsize_u=64

tag_prefix="_taslp_${dataset}_mnv2"

dataset_params="dataset=${dataset} dataset.train_subset=${train_subset} dataset.n_train_steps=${n_train_steps} dataset.sampler_s_balanced=${sampler_s_balanced} dataset.val_check_interval=${val_check_interval} dataset.pre_computed_specs=${pre_computed_specs}"
common_params="${run} ${dataset_params} debug=${debug} epochs=${epochs} model=${model} expt.activation=${activation} gpus=${gpus} cpus=${cpus} trainer.accelerator=${accelerator} verbose=${verbose}"
sup_params="expt.criterion=${criterion} bsize=${bsize}"
ssl_params="expt.criterion_s=${criterion} expt.criterion_u=${criterion} bsize_s=${bsize_s} bsize_u=${bsize_u}"

# Supervised 10%
# tag="${tag_prefix}_SUP10"
# $dpath_scripts/supervised.sh $common_params $sup_params expt=supervised ratio=0.1 tag=$tag
#tag="${tag_prefix}_SUP10m"
#$dpath_scripts/mixup.sh $common_params $sup_params expt=mixup ratio=0.1 tag=$tag

# Supervised 100%
tag="${tag_prefix}_SUP100"
$dpath_scripts/supervised.sh $common_params $sup_params expt=supervised ratio=1.0 tag=$tag
#tag="${tag_prefix}_SUP100m"
#$dpath_scripts/mixup.sh $common_params $sup_params expt=mixup ratio=1.0 tag=$tag

# FixMatch
#for threshold in 0.25 0.5 0.75
#do
#  for threshold_guess in 0.0 0.25 0.5 0.75
#  do
#    fm_params="expt.threshold=${threshold} expt.threshold_guess=${threshold_guess}"
#    tag="${tag_prefix}_FMTG"
##    $dpath_scripts/fixmatch.sh $common_params $ssl_params $fm_params tag=$tag expt=fixmatch_threshold_guess
#    tag="${tag_prefix}_FMTGM"
#    $dpath_scripts/fixmatch.sh $common_params $ssl_params $fm_params tag=$tag expt=fixmatch_threshold_guess_mixup
#  done
#done

# MixMatch
# mm_params=""
# tag="${tag_prefix}_MM"
# $dpath_scripts/mixmatch.sh $common_params $ssl_params $mm_params tag=$tag expt=mixmatch expt.temperature=${temperature}
# tag="${tag_prefix}_MMN"
# $dpath_scripts/mixmatch.sh $common_params $ssl_params $mm_params tag=$tag expt=mixmatch_nomixup expt.temperature=${temperature}
# tag="${tag_prefix}_MMM"
# $dpath_scripts/mixmatch.sh $common_params $ssl_params $mm_params tag=$tag expt=mixmatch_multi_sharp expt.sharpen_threshold=${sharpen_threshold}

