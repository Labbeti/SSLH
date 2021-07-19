#!/bin/sh

if [ ! "$1" = "run" ] && [ ! "$1" = "run_olympe" ] && [ ! "$1" = "run_osirim" ]; then
  echo "Usage: $0 (run|run_olympe|run_osirim)"
  exit 1
fi

# - Common params
run="$1"
epochs=1
optim="adam"
lr=1e-3
sched="cosine"
tag_prefix="_interspeech_v3"

common_params="${run} epochs=${epochs} optim=${optim} optim.lr=${lr} sched=${sched}"

# Other non-common params
bsize=256
criterion="ce"

bsize_s=128
bsize_u=128
criterion_s="ce"
criterion_u="ce"
criterion_u1="ce"

sup_params="bsize=${bsize} criterion=${criterion}"
ssl_params="bsize_s=${bsize_s} bsize_u=${bsize_u} experiment.criterion_s=${criterion_s} experiment.criterion_u=${criterion_u}"

# -- PVC
dataset="pvc"
n_train_steps=50000
sched_on_epoch=false

dataset_params="dataset=${dataset} dataset.n_train_steps=${n_train_steps} sched.on_epoch=${sched_on_epoch}"


# - MobileNetV2 & MobileNetV2Rot
model="mnv2"
modelrot="mnv2rot"
tag="${tag_prefix}_${model}"

./mixup.sh $common_params $dataset_params $sup_params model=$model expt.augm_train=weak tag="${tag}_10%" ratio=0.1
./mixup.sh $common_params $dataset_params $sup_params model=$model expt.augm_train=weak tag="${tag}_100%"

./fixmatch.sh $common_params $dataset_params $ssl_params model=$model tag=$tag
./mixmatch.sh $common_params $dataset_params $ssl_params model=$model tag=$tag
./remixmatch.sh $common_params $dataset_params $ssl_params model=$modelrot tag=$tag expt.criterion_u1=${criterion_u1}
./uda.sh $common_params $dataset_params $ssl_params model=$model tag=$tag

./remixmatch.sh $common_params $dataset_params $ssl_params model=$modelrot tag=$tag expt.criterion_u1=${criterion_u1} expt=remixmatch_norot


# - WideResNet28 & WideResNet28Rot
model="wrn28"
modelrot="wrn28rot"
tag="${tag_prefix}_${model}"

./mixup.sh $common_params $dataset_params $sup_params model=$model expt.augm_train=weak tag="${tag}_10%" ratio=0.1
./mixup.sh $common_params $dataset_params $sup_params model=$model expt.augm_train=weak tag="${tag}_100%"

./fixmatch.sh $common_params $dataset_params $ssl_params model=$model tag=$tag
./mixmatch.sh $common_params $dataset_params $ssl_params model=$model tag=$tag
./remixmatch.sh $common_params $dataset_params $ssl_params model=$modelrot tag=$tag expt.criterion_u1=${criterion_u1}
./uda.sh $common_params $dataset_params $ssl_params model=$model tag=$tag

./remixmatch.sh $common_params $dataset_params $ssl_params model=$modelrot tag=$tag expt.criterion_u1=${criterion_u1} expt=remixmatch_norot
