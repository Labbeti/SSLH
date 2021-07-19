#!/bin/sh

if [ ! "$1" = "run" ] && [ ! "$1" = "run_olympe" ] && [ ! "$1" = "run_osirim" ]; then
  echo "Usage: $0 (run|run_olympe|run_osirim)"
  exit 1
fi

dpath_scripts=`realpath $0 | xargs dirname | xargs dirname`
run="$1"

# ESC10
epochs=300
activation="softmax"
gpus=1
cpus=4
accelerator="dp"
verbose=true
debug=false

dataset="esc10"

criterion="CE"
bsize=60
bsize_s=30
bsize_u=30

model="wideresnet28"
model_acro="wrn28"

sup_params="expt.criterion=${criterion} bsize=${bsize} model=${model}"
ssl_params="expt.criterion_s=${criterion} expt.criterion_u=${criterion} bsize_s=${bsize_s} bsize_u=${bsize_u}"

for fold_val in 1 2 3 4 5
do

tag_prefix="_taslp_${dataset}_fold_${fold_val}_${model_acro}"
dataset_params="dataset=${dataset} dataset.folds_val=${fold_val}"
common_params="${run} ${dataset_params} epochs=${epochs} expt.activation=${activation} gpus=${gpus} cpus=${cpus} trainer.accelerator=${accelerator} verbose=${verbose} debug=${debug}"

# ReMixMatch
rmm_params="model=${model}_rot"
tag="${tag_prefix}_RMM"
$dpath_scripts/remixmatch.sh $common_params $ssl_params $rmm_params tag=$tag expt=remixmatch
tag="${tag_prefix}_RMMN"
$dpath_scripts/remixmatch.sh $common_params $ssl_params $rmm_params tag=$tag expt=remixmatch_nomixup

done

# UBS8K
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
dataset_params="dataset=${dataset} dataset.folds_val=${fold_val}"
common_params="${run} ${dataset_params} epochs=${epochs} expt.activation=${activation} gpus=${gpus} cpus=${cpus} trainer.accelerator=${accelerator} verbose=${verbose} debug=${debug}"

# ReMixMatch
rmm_params="model=${model}_rot"
tag="${tag_prefix}_RMM"
$dpath_scripts/remixmatch.sh $common_params $ssl_params $rmm_params tag=$tag expt=remixmatch
tag="${tag_prefix}_RMMN"
$dpath_scripts/remixmatch.sh $common_params $ssl_params $rmm_params tag=$tag expt=remixmatch_nomixup

done

# GSC
epochs=300
activation="softmax"
gpus=1
cpus=4
accelerator="dp"
verbose=true
debug=false

dataset="gsc"

criterion="CE"
bsize=256
bsize_s=128
bsize_u=128

model="wideresnet28"
model_acro="wrn28"

sup_params="expt.criterion=${criterion} bsize=${bsize} model=${model}"
ssl_params="expt.criterion_s=${criterion} expt.criterion_u=${criterion} bsize_s=${bsize_s} bsize_u=${bsize_u}"

tag_prefix="_taslp_${dataset}_${model_acro}"
dataset_params="dataset=${dataset}"
common_params="${run} ${dataset_params} epochs=${epochs} expt.activation=${activation} gpus=${gpus} cpus=${cpus} trainer.accelerator=${accelerator} verbose=${verbose} debug=${debug}"

# ReMixMatch
rmm_params="model=${model}_rot"
tag="${tag_prefix}_RMM"
$dpath_scripts/remixmatch.sh $common_params $ssl_params $rmm_params tag=$tag expt=remixmatch
tag="${tag_prefix}_RMMN"
$dpath_scripts/remixmatch.sh $common_params $ssl_params $rmm_params tag=$tag expt=remixmatch_nomixup
