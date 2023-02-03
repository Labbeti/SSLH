#!/bin/bash
# -*- coding: utf-8 -*-

dn0=`dirname $0`
bn0=`basename $0`

cache_dpath="${dn0}/cache"
mkdir -p "${cache_dpath}"
round_fpath="${cache_dpath}/round.txt"

if [ -f "${round_fpath}" ]; then
    rd=`cat "${round_fpath}"`
else
    rd=-1
fi
rd=$(expr $rd + 1)
echo $rd > "${round_fpath}"

# rd=`seq -f "%04g" ${rd} ${rd}`
rd="R${rd}"

echo "Launch round seed: ${rd} (dn0=${dn0})"

# # ------------------------------------------------------
# # PL, UDA without and with Mixup
# # ------------------------------------------------------
# launcher="osi"
# epochs=300
# seed=1234

# common="launcher=${launcher} epochs=${epochs} seed=${seed}"

# config_name_lst=(
#     "pseudo_labeling"
#     "pseudo_labeling"
#     "uda"
#     "uda"
# )
# pl_lst=(
#     "pseudo_labeling"
#     "pseudo_labeling_mixup"
#     "uda"
#     "uda_mixup"
# )

# for i in $(seq 0 $(expr ${#pl_lst[@]} - 1))
# do
# config_name="${config_name_lst[$i]}"
# pl=${pl_lst[$i]}

# # --- ESC10
# data="ssl_esc10"
# bsize_s=30
# bsize_u=30

# for val_fold in {1..5}
# do
#     tag="${rd}-pl_${pl}-data_${data}"

#     data_args="data=${data} data.dm.bsize_s=${bsize_s} data.dm.bsize_u=${bsize_u} data.dm.train_folds=null data.dm.val_folds=[${val_fold}]"
#     other_args="tag=${tag}"

#     ${dn0}/run.sh --config-name ${config_name} ${common} ${data_args} ${other_args}
# done

# # --- UBS8K
# data="ssl_ubs8k"
# bsize_s=128
# bsize_u=128

# for val_fold in {1..10}
# do
#     tag="${rd}-pl_${pl}-data_${data}"

#     data_args="data=${data} data.dm.bsize_s=${bsize_s} data.dm.bsize_u=${bsize_u} data.dm.train_folds=null data.dm.val_folds=[${val_fold}]"
#     other_args="tag=${tag}"

#     ${dn0}/run.sh --config-name ${config_name} ${common} ${data_args} ${other_args}
# done

# # --- GSC
# data="ssl_gsc"
# bsize_s=128
# bsize_u=128

# tag="${rd}-pl_${pl}-data_${data}"

# data_args="data=${data} data.dm.bsize_s=${bsize_s} data.dm.bsize_u=${bsize_u}"
# other_args="tag=${tag}"

# ${dn0}/run.sh --config-name ${config_name} ${common} ${data_args} ${other_args}

# done

# # ------------------------------------------------------
# # Mean Teacher ?
# # ------------------------------------------------------
# launcher="osi"
# epochs=300
# seed=1234

# common="launcher=${launcher} epochs=${epochs} seed=${seed}"
# config_name="mean_teacher"
# pl="mean_teacher_aug"

# # --- GSC
# data="ssl_gsc"
# bsize_s=128
# bsize_u=128

# tag="${rd}-pl_${pl}-data_${data}"

# data_args="data=${data} data.dm.bsize_s=${bsize_s} data.dm.bsize_u=${bsize_u}"
# other_args="tag=${tag}"

# ${dn0}/run.sh --config-name ${config_name} ${common} ${data_args} ${other_args}

# ------------------------------------------------------
# Search better hp
# ------------------------------------------------------
launcher="osi"
epochs=300
seed=1234

common="launcher=${launcher} epochs=${epochs} seed=${seed}"

config_name_lst=(
    "pseudo_labeling"
    "pseudo_labeling"
    "mean_teacher"
    "mean_teacher"
    "uda"
    "uda"
)
pl_lst=(
    "pseudo_labeling"
    "pseudo_labeling_mixup"
    "mean_teacher"
    "mean_teacher_mixup"
    "uda"
    "uda_mixup"
)
added_args_lst=(
    ""
    ""
    ""
    ""
    ""
    ""
)
posttag_lst=(
    ""
    ""
    ""
    ""
    ""
    ""
)

# --- ESC10
for i in $(seq 0 $(expr ${#config_name_lst[@]} - 1))
do
config_name="${config_name_lst[$i]}"
pl=${pl_lst[$i]}
added_args=${added_args_lst[$i]}
posttag=${posttag_lst[$i]}

data="ssl_esc10"
bsize_s=30
bsize_u=30

for val_fold in {1..5}
do
    tag="${rd}-data_${data}-pl_${pl}-bsizes_${bsize_s}_${bsize_u}-${posttag}-val_fold_${val_fold}"

    data_args="data=${data} data.dm.bsize_s=${bsize_s} data.dm.bsize_u=${bsize_u} data.dm.train_folds=null data.dm.val_folds=[${val_fold}]"
    pl_args="pl=${pl}"
    other_args="tag=${tag}"

    ${dn0}/run.sh --config-name ${config_name} ${common} ${data_args} ${other_args} ${pl_args} ${added_args}
done
done

# --- GSC
for i in $(seq 0 $(expr ${#config_name_lst[@]} - 1))
do
config_name="${config_name_lst[$i]}"
pl=${pl_lst[$i]}
added_args=${added_args_lst[$i]}
posttag=${posttag_lst[$i]}

data="ssl_gsc"
bsize_s=128
bsize_u=128

tag="${rd}-data_${data}-pl_${pl}-bsizes_${bsize_s}_${bsize_u}-${posttag}"

data_args="data=${data} data.dm.bsize_s=${bsize_s} data.dm.bsize_u=${bsize_u}"
pl_args="pl=${pl}"
other_args="tag=${tag}"

${dn0}/run.sh --config-name ${config_name} ${common} ${data_args} ${other_args} ${pl_args} ${added_args}
done


# --- UBS8K
for i in $(seq 0 $(expr ${#config_name_lst[@]} - 1))
do
config_name="${config_name_lst[$i]}"
pl=${pl_lst[$i]}
added_args=${added_args_lst[$i]}
posttag=${posttag_lst[$i]}

data="ssl_ubs8k"
bsize_s=128
bsize_u=128

for val_fold in {1..10}
do
    tag="${rd}-data_${data}-pl_${pl}-bsizes_${bsize_s}_${bsize_u}-${posttag}-val_fold_${val_fold}"

    data_args="data=${data} data.dm.bsize_s=${bsize_s} data.dm.bsize_u=${bsize_u} data.dm.train_folds=null data.dm.val_folds=[${val_fold}]"
    pl_args="pl=${pl}"
    other_args="tag=${tag}"

    ${dn0}/run.sh --config-name ${config_name} ${common} ${data_args} ${other_args} ${pl_args} ${added_args}
done
done
