#! /bin/bash

rd="${RANDOM}"
path="osirim"

# ------------------------------------------------------
# Test augments for supervised
# ------------------------------------------------------
# bsize=256
# data="sup_gsc"
# epochs=200
# pl="supervised"
# ratio=1.0

# common="path=${path} data=${data} epochs=${epochs} data.dm.bsize=${bsize} pl=${pl} data.dm.ratio=${ratio}"

# for seed in 1234 1235 1236
# do
#     train_aug="ident"
#     tag="${rd}__data_${data}__pl_${pl}__epochs_${epochs}__ratio_${ratio}__train_aug_${train_aug}__seed_${seed}"

#     ./run.sh supervised ${common} tag=${tag} aug@train_aug=${train_aug} seed=${seed}

#     train_aug="spec_aug"

#     time_drop_width=64
#     time_stripes_num=2
#     freq_drop_width=2
#     freq_stripes_num=1

#     for p in 0.5 1.0
#     do
#         tag="${rd}__data_${data}__pl_${pl}__epochs_${epochs}__ratio_${ratio}__train_aug_${train_aug}__time_drop_width_${time_drop_width}__time_stripes_num_${time_stripes_num}__freq_drop_width_${freq_drop_width}__freq_stripes_num_${freq_stripes_num}__p_${p}__seed_${seed}"

#         ./run.sh supervised ${common} tag=${tag} aug@train_aug=${train_aug} seed=${seed} train_aug.0.aug.time_drop_width=${time_drop_width} train_aug.0.aug.time_stripes_num=${time_stripes_num} train_aug.0.aug.freq_drop_width=${freq_drop_width} train_aug.0.aug.freq_stripes_num=${freq_stripes_num} train_aug.0.aug.p=${p}
#     done
# done

# ------------------------------------------------------
# Test remixmatch rotation replace
# ------------------------------------------------------
# epochs=200
# common="path=${path} epochs=${epochs}"

# # GSC
# data="ssl_gsc"
# bsize_s=128
# bsize_u=128
# data_args="data=${data} data.dm.bsize_s=${bsize_s} data.dm.bsize_u=${bsize_u}"

# pl="remixmatch"
# rot_size=4
# self_transform_mode="hvflips"
# tag="${rd}__data_${data}__pl_${pl}__epochs_${epochs}__rot_size_${rot_size}__self_transform_mode_${self_transform_mode}"
# ./run.sh remixmatch ${common} pl=${pl} model.rot_size=${rot_size} pl.self_transform_mode=${self_transform_mode} tag=${tag} ${data_args}

# pl="remixmatch"
# rot_size=2
# self_transform_mode="hflips"
# tag="${rd}__data_${data}__pl_${pl}__epochs_${epochs}__rot_size_${rot_size}__self_transform_mode_${self_transform_mode}"
# ./run.sh remixmatch ${common} pl=${pl} model.rot_size=${rot_size} pl.self_transform_mode=${self_transform_mode} tag=${tag} ${data_args}

# pl="remixmatch_norot"
# model="wideresnet28"
# tag="${rd}__data_${data}__pl_${pl}__epochs_${epochs}__model_${model}"
# ./run.sh remixmatch ${common} pl=${pl} model=${model} tag=${tag} ${data_args}

# pl="remixmatch"
# rot_size=2
# self_transform_mode="vflips"
# tag="${rd}__data_${data}__pl_${pl}__epochs_${epochs}__rot_size_${rot_size}__self_transform_mode_${self_transform_mode}"
# ./run.sh remixmatch ${common} pl=${pl} model.rot_size=${rot_size} pl.self_transform_mode=${self_transform_mode} tag=${tag} ${data_args}

# # ESC10
# data="ssl_esc10"
# bsize_s=30
# bsize_u=30
# data_args="data=${data} data.dm.bsize_s=${bsize_s} data.dm.bsize_u=${bsize_u} data.dm.train_folds=null"

# for val_fold in 1 2 3 4 5
# do
#     pl="remixmatch"
#     rot_size=4
#     self_transform_mode="hvflips"
#     tag="${rd}__data_${data}__pl_${pl}__epochs_${epochs}__rot_size_${rot_size}__self_transform_mode_${self_transform_mode}__val_fold_${val_fold}"
#     ./run.sh remixmatch ${common} pl=${pl} model.rot_size=${rot_size} pl.self_transform_mode=${self_transform_mode} tag=${tag} ${data_args} data.dm.val_folds=[${val_fold}]

#     pl="remixmatch"
#     rot_size=2
#     self_transform_mode="hflips"
#     tag="${rd}__data_${data}__pl_${pl}__epochs_${epochs}__rot_size_${rot_size}__self_transform_mode_${self_transform_mode}__val_fold_${val_fold}"
#     ./run.sh remixmatch ${common} pl=${pl} model.rot_size=${rot_size} pl.self_transform_mode=${self_transform_mode} tag=${tag} ${data_args} data.dm.val_folds=[${val_fold}]

#     pl="remixmatch_norot"
#     model="wideresnet28"
#     tag="${rd}__data_${data}__pl_${pl}__epochs_${epochs}__model_${model}__val_fold_${val_fold}"
#     ./run.sh remixmatch ${common} pl=${pl} model=${model} tag=${tag} ${data_args} data.dm.val_folds=[${val_fold}]

#     pl="remixmatch"
#     rot_size=2
#     self_transform_mode="vflips"
#     tag="${rd}__data_${data}__pl_${pl}__epochs_${epochs}__rot_size_${rot_size}__self_transform_mode_${self_transform_mode}__val_fold_${val_fold}"
#     ./run.sh remixmatch ${common} pl=${pl} model.rot_size=${rot_size} pl.self_transform_mode=${self_transform_mode} tag=${tag} ${data_args} data.dm.val_folds=[${val_fold}]
# done

# # UBS8K
# epochs=200
# data="ssl_ubs8k"
# bsize_s=128
# bsize_u=128
# common="path=${path} epochs=${epochs} data=${data} data.dm.bsize_s=${bsize_s} data.dm.bsize_u=${bsize_u} data.dm.train_folds=null"

# for val_fold in 1 2 3 4 5 6 7 8 9 10
# do
    # pl="remixmatch"
    # rot_size=4
    # self_transform_mode="hvflips"
    # tag="${rd}__data_${data}__pl_${pl}__epochs_${epochs}__rot_size_${rot_size}__self_transform_mode_${self_transform_mode}__val_fold_${val_fold}"
    # ./run.sh remixmatch ${common} pl=${pl} model.rot_size=${rot_size} pl.self_transform_mode=${self_transform_mode} tag=${tag} data.dm.val_folds=[${val_fold}]

    # pl="remixmatch"
    # rot_size=2
    # self_transform_mode="hflips"
    # tag="${rd}__data_${data}__pl_${pl}__epochs_${epochs}__rot_size_${rot_size}__self_transform_mode_${self_transform_mode}__val_fold_${val_fold}"
    # ./run.sh remixmatch ${common} pl=${pl} model.rot_size=${rot_size} pl.self_transform_mode=${self_transform_mode} tag=${tag} data.dm.val_folds=[${val_fold}]

#     pl="remixmatch_norot"
#     model="wideresnet28"
#     tag="${rd}__data_${data}__pl_${pl}__epochs_${epochs}__model_${model}__val_fold_${val_fold}"
#     ./run.sh remixmatch ${common} pl=${pl} model=${model} tag=${tag} data.dm.val_folds=[${val_fold}]

#     pl="remixmatch"
#     rot_size=2
#     self_transform_mode="vflips"
#     tag="${rd}__data_${data}__pl_${pl}__epochs_${epochs}__rot_size_${rot_size}__self_transform_mode_${self_transform_mode}__val_fold_${val_fold}"
#     ./run.sh remixmatch ${common} pl=${pl} model.rot_size=${rot_size} pl.self_transform_mode=${self_transform_mode} tag=${tag} data.dm.val_folds=[${val_fold}]
# done


# ------------------------------------------------------
# Test augments for mixup
# ------------------------------------------------------
# bsize=256
# data="sup_gsc"
# epochs=200
# pl="mixup_mix_label"
# ratio=1.0

# common="path=${path} data=${data} epochs=${epochs} data.dm.bsize=${bsize} pl=${pl} data.dm.ratio=${ratio}"
# seed=1234

# train_aug="ident"
# tag="${rd}__data_${data}__pl_${pl}__epochs_${epochs}__ratio_${ratio}__train_aug_${train_aug}__seed_${seed}"

# ./run.sh mixup ${common} tag=${tag} aug@train_aug=${train_aug} seed=${seed}


# train_aug="weak"
# tag="${rd}__data_${data}__pl_${pl}__epochs_${epochs}__ratio_${ratio}__train_aug_${train_aug}__seed_${seed}"

# ./run.sh mixup ${common} tag=${tag} aug@train_aug=${train_aug} seed=${seed}


# train_aug="strong"
# tag="${rd}__data_${data}__pl_${pl}__epochs_${epochs}__ratio_${ratio}__train_aug_${train_aug}__seed_${seed}"

# ./run.sh mixup ${common} tag=${tag} aug@train_aug=${train_aug} seed=${seed}


# train_aug="spec_aug"

# freq_drop_width=8
# freq_stripes_num=2

# for time_drop_width in 3 4 8
# do
#     for time_stripes_num in 1 2
#     do
#         for p in 0.5 1.0
#         do
#             tag="${rd}__data_${data}__pl_${pl}__epochs_${epochs}__ratio_${ratio}__train_aug_${train_aug}__time_drop_width_${time_drop_width}__time_stripes_num_${time_stripes_num}__freq_drop_width_${freq_drop_width}__freq_stripes_num_${freq_stripes_num}__p_${p}__seed_${seed}"

#             ./run.sh mixup ${common} tag=${tag} aug@train_aug=${train_aug} seed=${seed} train_aug.0.aug.time_drop_width=${time_drop_width} train_aug.0.aug.time_stripes_num=${time_stripes_num} train_aug.0.aug.freq_drop_width=${freq_drop_width} train_aug.0.aug.freq_stripes_num=${freq_stripes_num} train_aug.0.aug.p=${p}
#         done
#     done
# done

# ------------------------------------------------------
# Update RMM results on olympe
# ------------------------------------------------------
# path="olympe"
# epochs=300
# common="path=${path} epochs=${epochs}"

# # ESC10
# data="ssl_esc10"
# bsize_s=30
# bsize_u=30
# data_args="data=${data} data.dm.bsize_s=${bsize_s} data.dm.bsize_u=${bsize_u} data.dm.train_folds=null"

# for val_fold in 1 2 3 4 5
# do
#     pl="remixmatch"
#     tag="${rd}__data_${data}__pl_${pl}__epochs_${epochs}__val_fold_${val_fold}"
#     ./run.sh remixmatch ${common} pl=${pl} tag=${tag} ${data_args} data.dm.val_folds=[${val_fold}]

#     pl="remixmatch_nomixup"
#     tag="${rd}__data_${data}__pl_${pl}__epochs_${epochs}__val_fold_${val_fold}"
#     ./run.sh remixmatch ${common} pl=${pl} tag=${tag} ${data_args} data.dm.val_folds=[${val_fold}]

#     pl="remixmatch_norot"
#     model="wideresnet28"
#     tag="${rd}__data_${data}__pl_${pl}__epochs_${epochs}__model_${model}__val_fold_${val_fold}"
#     ./run.sh remixmatch ${common} pl=${pl} tag=${tag} ${data_args} data.dm.val_folds=[${val_fold}] model=${model}

#     pl="remixmatch_nomixup_norot"
#     model="wideresnet28"
#     tag="${rd}__data_${data}__pl_${pl}__epochs_${epochs}__model_${model}__val_fold_${val_fold}"
#     ./run.sh remixmatch ${common} pl=${pl} tag=${tag} ${data_args} data.dm.val_folds=[${val_fold}] model=${model}
# done

# # GSC
# data="ssl_gsc"
# bsize_s=128
# bsize_u=128
# data_args="data=${data} data.dm.bsize_s=${bsize_s} data.dm.bsize_u=${bsize_u}"

# pl="remixmatch"
# tag="${rd}__data_${data}__pl_${pl}__epochs_${epochs}"
# ./run.sh remixmatch ${common} pl=${pl} tag=${tag} ${data_args}

# pl="remixmatch_nomixup"
# tag="${rd}__data_${data}__pl_${pl}__epochs_${epochs}"
# ./run.sh remixmatch ${common} pl=${pl} tag=${tag} ${data_args}

# pl="remixmatch_norot"
# model="wideresnet28"
# tag="${rd}__data_${data}__pl_${pl}__epochs_${epochs}__model_${model}"
# ./run.sh remixmatch ${common} pl=${pl} tag=${tag} ${data_args} model=${model}

# pl="remixmatch_nomixup_norot"
# model="wideresnet28"
# tag="${rd}__data_${data}__pl_${pl}__epochs_${epochs}__model_${model}"
# ./run.sh remixmatch ${common} pl=${pl} tag=${tag} ${data_args} model=${model}

# # UBS8K
# data="ssl_ubs8k"
# bsize_s=128
# bsize_u=128
# data_args="data=${data} data.dm.bsize_s=${bsize_s} data.dm.bsize_u=${bsize_u} data.dm.train_folds=null"

# for val_fold in 1 2 3 4 5 6 7 8 9 10
# do
#     pl="remixmatch"
#     tag="${rd}__data_${data}__pl_${pl}__epochs_${epochs}__val_fold_${val_fold}"
#     ./run.sh remixmatch ${common} pl=${pl} tag=${tag} ${data_args} data.dm.val_folds=[${val_fold}]

#     pl="remixmatch_nomixup"
#     tag="${rd}__data_${data}__pl_${pl}__epochs_${epochs}__val_fold_${val_fold}"
#     ./run.sh remixmatch ${common} pl=${pl} tag=${tag} ${data_args} data.dm.val_folds=[${val_fold}]

#     pl="remixmatch_norot"
#     model="wideresnet28"
#     tag="${rd}__data_${data}__pl_${pl}__epochs_${epochs}__model_${model}__val_fold_${val_fold}"
#     ./run.sh remixmatch ${common} pl=${pl} tag=${tag} ${data_args} data.dm.val_folds=[${val_fold}] model=${model}

#     pl="remixmatch_nomixup_norot"
#     model="wideresnet28"
#     tag="${rd}__data_${data}__pl_${pl}__epochs_${epochs}__model_${model}__val_fold_${val_fold}"
#     ./run.sh remixmatch ${common} pl=${pl} tag=${tag} ${data_args} data.dm.val_folds=[${val_fold}] model=${model}
# done

# ------------------------------------------------------
# Test MT with augments
# ------------------------------------------------------
path="olympe"
epochs=200
ckpt_monitor="val/acc_tea"
ckpt_mode="max"

common="path=${path} epochs=${epochs} ckpt.monitor=${ckpt_monitor} ckpt.mode=${ckpt_mode}"

# # -- Base MT
# noise="gaussian_noise"
# train_aug_stu_s="ident"
# train_aug_stu_u="ident"
# train_aug_tea_s="ident"
# train_aug_tea_u="ident"
# aug_args="trans@pl.noise=${noise} aug@train_aug_stu_s=${train_aug_stu_s} aug@train_aug_stu_u=${train_aug_stu_u} aug@train_aug_tea_s=${train_aug_tea_s} aug@train_aug_tea_u=${train_aug_tea_u}"

# # ESC10
# data="ssl_esc10"
# bsize_s=6
# bsize_u=58
# drop_last="false"
# data_args="data=${data} data.dm.bsize_s=${bsize_s} data.dm.bsize_u=${bsize_u} data.dm.drop_last=${drop_last} data.dm.train_folds=null"

# for val_fold in 1 2 3 4 5
# do
#     pl="mean_teacher"
#     tag="${rd}__data_${data}__pl_${pl}__epochs_${epochs}__drop_last_${drop_last}__noise_${noise}__train_aug_stu_s_${train_aug_stu_s}__train_aug_stu_u_${train_aug_stu_u}__train_aug_tea_s_${train_aug_tea_s}__train_aug_tea_u_${train_aug_tea_u}__val_fold_${val_fold}"
#     ./run.sh mean_teacher ${common} pl=${pl} tag=${tag} ${data_args} ${aug_args} data.dm.val_folds=[${val_fold}]
# done

# # GSC
# data="ssl_gsc"
# bsize_s=6
# bsize_u=58
# drop_last="false"
# data_args="data=${data} data.dm.bsize_s=${bsize_s} data.dm.bsize_u=${bsize_u} data.dm.drop_last=${drop_last}"

# pl="mean_teacher"
# tag="${rd}__data_${data}__pl_${pl}__epochs_${epochs}__drop_last_${drop_last}__noise_${noise}__train_aug_stu_s_${train_aug_stu_s}__train_aug_stu_u_${train_aug_stu_u}__train_aug_tea_s_${train_aug_tea_s}__train_aug_tea_u_${train_aug_tea_u}__val_fold_${val_fold}"
# ./run.sh mean_teacher ${common} pl=${pl} tag=${tag} ${data_args} ${aug_args}


# --- Replace noise by weak
noise="ident"
train_aug_stu_s="ident"
train_aug_stu_u="ident"
train_aug_tea_s="weak"
train_aug_tea_u="weak"
aug_args="trans@pl.noise=${noise} aug@train_aug_stu_s=${train_aug_stu_s} aug@train_aug_stu_u=${train_aug_stu_u} aug@train_aug_tea_s=${train_aug_tea_s} aug@train_aug_tea_u=${train_aug_tea_u}"

# # ESC10
# data="ssl_esc10"
# bsize_s=6
# bsize_u=58
# drop_last="false"
# data_args="data=${data} data.dm.bsize_s=${bsize_s} data.dm.bsize_u=${bsize_u} data.dm.drop_last=${drop_last} data.dm.train_folds=null"

# for val_fold in 1 2 3 4 5
# do
#     pl="mean_teacher_aug"
#     tag="${rd}__data_${data}__pl_${pl}__epochs_${epochs}__drop_last_${drop_last}__noise_${noise}__train_aug_stu_s_${train_aug_stu_s}__train_aug_stu_u_${train_aug_stu_u}__train_aug_tea_s_${train_aug_tea_s}__train_aug_tea_u_${train_aug_tea_u}__val_fold_${val_fold}"
#     ./run.sh mean_teacher ${common} pl=${pl} tag=${tag} ${data_args} ${aug_args} data.dm.val_folds=[${val_fold}]
# done

# GSC
data="ssl_gsc"
bsize_s=6
bsize_u=58
drop_last="false"
data_args="data=${data} data.dm.bsize_s=${bsize_s} data.dm.bsize_u=${bsize_u} data.dm.drop_last=${drop_last}"

pl="mean_teacher_aug"
tag="${rd}__data_${data}__pl_${pl}__epochs_${epochs}__drop_last_${drop_last}__noise_${noise}__train_aug_stu_s_${train_aug_stu_s}__train_aug_stu_u_${train_aug_stu_u}__train_aug_tea_s_${train_aug_tea_s}__train_aug_tea_u_${train_aug_tea_u}__val_fold_${val_fold}"
./run.sh mean_teacher ${common} pl=${pl} tag=${tag} ${data_args} ${aug_args}



# --- Add weak everywhere
noise="ident"
train_aug_stu_s="weak"
train_aug_stu_u="weak"
train_aug_tea_s="weak"
train_aug_tea_u="weak"
aug_args="trans@pl.noise=${noise} aug@train_aug_stu_s=${train_aug_stu_s} aug@train_aug_stu_u=${train_aug_stu_u} aug@train_aug_tea_s=${train_aug_tea_s} aug@train_aug_tea_u=${train_aug_tea_u}"

# # ESC10
# data="ssl_esc10"
# bsize_s=6
# bsize_u=58
# drop_last="false"
# data_args="data=${data} data.dm.bsize_s=${bsize_s} data.dm.bsize_u=${bsize_u} data.dm.drop_last=${drop_last} data.dm.train_folds=null"

# for val_fold in 1 2 3 4 5
# do
#     pl="mean_teacher_aug"
#     tag="${rd}__data_${data}__pl_${pl}__epochs_${epochs}__drop_last_${drop_last}__noise_${noise}__train_aug_stu_s_${train_aug_stu_s}__train_aug_stu_u_${train_aug_stu_u}__train_aug_tea_s_${train_aug_tea_s}__train_aug_tea_u_${train_aug_tea_u}__val_fold_${val_fold}"
#     ./run.sh mean_teacher ${common} pl=${pl} tag=${tag} ${data_args} ${aug_args} data.dm.val_folds=[${val_fold}]
# done

# GSC
data="ssl_gsc"
bsize_s=6
bsize_u=58
drop_last="false"
data_args="data=${data} data.dm.bsize_s=${bsize_s} data.dm.bsize_u=${bsize_u} data.dm.drop_last=${drop_last}"

pl="mean_teacher_aug"
tag="${rd}__data_${data}__pl_${pl}__epochs_${epochs}__drop_last_${drop_last}__noise_${noise}__train_aug_stu_s_${train_aug_stu_s}__train_aug_stu_u_${train_aug_stu_u}__train_aug_tea_s_${train_aug_tea_s}__train_aug_tea_u_${train_aug_tea_u}__val_fold_${val_fold}"
./run.sh mean_teacher ${common} pl=${pl} tag=${tag} ${data_args} ${aug_args}


# ------------------------------------------------------
# Test augment specaugment + resample (weak4 & strong4)
# ------------------------------------------------------
# bsize=256
# data="sup_gsc"
# epochs=200
# pl="mixup_mix_label"
# ratio=1.0

# common="path=${path} data=${data} epochs=${epochs} data.dm.bsize=${bsize} pl=${pl} data.dm.ratio=${ratio}"
# seed=1234

# freq_drop_width=8
# freq_stripes_num=2

# for time_drop_width in 8 16
# do
#     for time_stripes_num in 1 2
#     do
#         for p in 0.5 1.0
#         do
#             train_aug="weak4"
#             tag="${rd}__data_${data}__pl_${pl}__epochs_${epochs}__train_aug_${train_aug}__freq_drop_width_${freq_drop_width}__freq_stripes_num_${freq_stripes_num}__time_drop_width_${time_drop_width}__time_stripes_num_${time_stripes_num}__p_${p}__seed_${seed}"
#             ./run.sh mixup ${common} tag=${tag} aug@train_aug=${train_aug} seed=${seed} train_aug.0.aug.time_drop_width=${time_drop_width} train_aug.0.aug.time_stripes_num=${time_stripes_num} train_aug.0.aug.freq_drop_width=${freq_drop_width} train_aug.0.aug.freq_stripes_num=${freq_stripes_num} train_aug.0.aug.p=${p}
        
#             train_aug="strong4"
#             tag="${rd}__data_${data}__pl_${pl}__epochs_${epochs}__train_aug_${train_aug}__freq_drop_width_${freq_drop_width}__freq_stripes_num_${freq_stripes_num}__time_drop_width_${time_drop_width}__time_stripes_num_${time_stripes_num}__p_${p}__seed_${seed}"
#             ./run.sh mixup ${common} tag=${tag} aug@train_aug=${train_aug} seed=${seed} train_aug.0.aug.time_drop_width=${time_drop_width} train_aug.0.aug.time_stripes_num=${time_stripes_num} train_aug.0.aug.freq_drop_width=${freq_drop_width} train_aug.0.aug.freq_stripes_num=${freq_stripes_num} train_aug.0.aug.p=${p}
#         done
#     done
# done
