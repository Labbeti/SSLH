#!/bin/sh

if [ ! "$1" = "run" ] && [ ! "$1" = "run_olympe" ] && [ ! "$1" = "run_osirim" ]; then
  echo "Usage: $0 (run|run_olympe|run_osirim)"
  exit 1
fi

# - Common params
run="$1"
epochs=300
optim="adam"
lr=1e-3
sched="cosine"
tag_prefix="_interspeech_v2"

common_params="${run} epochs=${epochs} optim=${optim} optim.lr=${lr} sched=${sched}"

# Other non-common params
bsize=256
bsize_s=128
bsize_u=128

# -- UBS8K
dataset="ubs8k"

for fold_val in 1 2 3 4 5 6 7 8 9 10
do

dataset_params="dataset=${dataset} dataset.folds_val=[${fold_val}] dataset.folds_train=null"

# - WideResNet28 & WideResNet28Rot
model="wrn28"
modelrot="wrn28rot"
tag="${tag_prefix}_${model}_fold_${fold_val}"

./mixup.sh $common_params $dataset_params model=$model bsize=$bsize experiment.augm_train=weak tag="${tag}_10%" ratio=0.1

./mixup.sh $common_params $dataset_params model=$model bsize=$bsize experiment.augm_train=weak tag="${tag}_100%"

./fixmatch.sh $common_params $dataset_params model=$model bsize_s=$bsize_s bsize_u=$bsize_u tag=$tag

./mixmatch.sh $common_params $dataset_params model=$model bsize_s=$bsize_s bsize_u=$bsize_u tag=$tag

./remixmatch.sh $common_params $dataset_params model=$modelrot bsize_s=$bsize_s bsize_u=$bsize_u tag=$tag
./remixmatch.sh $common_params $dataset_params model=$modelrot bsize_s=$bsize_s bsize_u=$bsize_u tag=$tag experiment=remixmatch_norot

./uda.sh $common_params $dataset_params model=$model bsize_s=$bsize_s bsize_u=$bsize_u tag=$tag


# - MobileNetV2 & MobileNetV2Rot
model="mnv2"
modelrot="mnv2rot"
tag="${tag_prefix}_${model}_fold_${fold_val}"

./mixup.sh $common_params $dataset_params model=$model bsize=$bsize experiment.augm_train=weak tag="${tag}_10%" ratio=0.1

./mixup.sh $common_params $dataset_params model=$model bsize=$bsize experiment.augm_train=weak tag="${tag}_100%"

./fixmatch.sh $common_params $dataset_params model=$model bsize_s=$bsize_s bsize_u=$bsize_u tag=$tag

./mixmatch.sh $common_params $dataset_params model=$model bsize_s=$bsize_s bsize_u=$bsize_u tag=$tag

./remixmatch.sh $common_params $dataset_params model=$modelrot bsize_s=$bsize_s bsize_u=$bsize_u tag=$tag
./remixmatch.sh $common_params $dataset_params model=$modelrot bsize_s=$bsize_s bsize_u=$bsize_u tag=$tag experiment=remixmatch_norot

./uda.sh $common_params $dataset_params model=$model bsize_s=$bsize_s bsize_u=$bsize_u tag=$tag

done