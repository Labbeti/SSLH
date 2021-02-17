
run_esc10=1
run_ubs8k=1
run_gsc=1

# -----------------------------------
# Global (common parameters)
# -----------------------------------
su_ratio=0.1
full_rat=1.0
tag_prefix="paper_$RANDOM"
seed=1234
nb_epochs=300
lr=0.001
su_alpha=0.4
ss_alpha=0.75
sched="none"
use_warmup_by_epoch=0 # 1
warmup_nb_steps=16000 # 50

# -----------------------------------
# ESC10
# -----------------------------------
if [ ${run_esc10} = 1 ]
then

dataset="ESC10"
cv=1
bsize_s=30
bsize_u=30
logdir="/users/samova/elabbe/root_sslh/tensorboard/ESC10/paper/"
threshold=0.8

# Supervised trainings
# - base 
./su_exp.sh \
	--dataset ${dataset} --logdir ${logdir} --bsize_s ${bsize_s} --cv ${cv} --seed ${seed} --nb_epochs ${nb_epochs} --lr ${lr} --sched ${sched} \
	--su_ratio ${su_ratio} --tag ${tag_prefix}_su
./su_exp.sh \
	--dataset ${dataset} --logdir ${logdir} --bsize_s ${bsize_s} --cv ${cv} --seed ${seed} --nb_epochs ${nb_epochs} --lr ${lr} --sched ${sched} \
	--su_ratio ${full_rat} --tag ${tag_prefix}_su

# - mixup
./su_exp.sh \
	--dataset ${dataset} --logdir ${logdir} --bsize_s ${bsize_s} --cv ${cv} --seed ${seed} --nb_epochs ${nb_epochs} --lr ${lr} --sched ${sched} \
	--su_ratio ${su_ratio} --tag ${tag_prefix}_su_mix --use_mixup 1 --alpha ${su_alpha}
./su_exp.sh \
	--dataset ${dataset} --logdir ${logdir} --bsize_s ${bsize_s} --cv ${cv} --seed ${seed} --nb_epochs ${nb_epochs} --lr ${lr} --sched ${sched} \
	--su_ratio ${full_rat} --tag ${tag_prefix}_su_mix --use_mixup 1 --alpha ${su_alpha}

# - weak
./su_exp.sh \
	--dataset ${dataset} --logdir ${logdir} --bsize_s ${bsize_s} --cv ${cv} --seed ${seed} --nb_epochs ${nb_epochs} --lr ${lr} --sched ${sched} \
	--su_ratio ${su_ratio} --tag ${tag_prefix}_su_w2 --augm_none weak
./su_exp.sh \
	--dataset ${dataset} --logdir ${logdir} --bsize_s ${bsize_s} --cv ${cv} --seed ${seed} --nb_epochs ${nb_epochs} --lr ${lr} --sched ${sched} \
	--su_ratio ${full_rat} --tag ${tag_prefix}_su_w2 --augm_none weak

# - weak + mixup
./su_exp.sh \
	--dataset ${dataset} --logdir ${logdir} --bsize_s ${bsize_s} --cv ${cv} --seed ${seed} --nb_epochs ${nb_epochs} --lr ${lr} --sched ${sched} \
	--su_ratio ${su_ratio} --tag ${tag_prefix}_su_w2_mix --augm_none weak --use_mixup 1 --alpha ${su_alpha}
./su_exp.sh \
	--dataset ${dataset} --logdir ${logdir} --bsize_s ${bsize_s} --cv ${cv} --seed ${seed} --nb_epochs ${nb_epochs} --lr ${lr} --sched ${sched} \
	--su_ratio ${full_rat} --tag ${tag_prefix}_su_w2_mix --augm_none weak --use_mixup 1 --alpha ${su_alpha}

# - strong
./su_exp.sh \
	--dataset ${dataset} --logdir ${logdir} --bsize_s ${bsize_s} --cv ${cv} --seed ${seed} --nb_epochs ${nb_epochs} --lr ${lr} --sched ${sched} \
	--su_ratio ${su_ratio} --tag ${tag_prefix}_su_s2 --augm_none strong
./su_exp.sh \
	--dataset ${dataset} --logdir ${logdir} --bsize_s ${bsize_s} --cv ${cv} --seed ${seed} --nb_epochs ${nb_epochs} --lr ${lr} --sched ${sched} \
	--su_ratio ${full_rat} --tag ${tag_prefix}_su_s2 --augm_none strong

# - strong + mixup
./su_exp.sh \
	--dataset ${dataset} --logdir ${logdir} --bsize_s ${bsize_s} --cv ${cv} --seed ${seed} --nb_epochs ${nb_epochs} --lr ${lr} --sched ${sched} \
	--su_ratio ${su_ratio} --tag ${tag_prefix}_su_s2_mix --augm_none strong --use_mixup 1 --alpha ${su_alpha}
./su_exp.sh \
	--dataset ${dataset} --logdir ${logdir} --bsize_s ${bsize_s} --cv ${cv} --seed ${seed} --nb_epochs ${nb_epochs} --lr ${lr} --sched ${sched} \
	--su_ratio ${full_rat} --tag ${tag_prefix}_su_s2_mix --augm_none strong --use_mixup 1 --alpha ${su_alpha}

# Semi-Supervised trainings
# - mixmatch
./mm_exp.sh \
	--dataset ${dataset} --logdir ${logdir} --bsize_s ${bsize_s} --cv ${cv} --seed ${seed} --nb_epochs ${nb_epochs} --lr ${lr} --sched ${sched} \
	--su_ratio ${su_ratio} --tag ${tag_prefix}_mm --bsize_u ${bsize_u} --alpha ${ss_alpha} --use_warmup_by_epoch ${use_warmup_by_epoch} --warmup_nb_steps ${warmup_nb_steps}

# - mixmatch + no mixup
./mm_exp.sh \
	--dataset ${dataset} --logdir ${logdir} --bsize_s ${bsize_s} --cv ${cv} --seed ${seed} --nb_epochs ${nb_epochs} --lr ${lr} --sched ${sched} \
	--su_ratio ${su_ratio} --tag ${tag_prefix}_mm_no_mix --bsize_u ${bsize_u} --use_no_mixup 1 --use_warmup_by_epoch ${use_warmup_by_epoch} --warmup_nb_steps ${warmup_nb_steps}

# - fixmatch
./fm_exp.sh \
	--dataset ${dataset} --logdir ${logdir} --bsize_s ${bsize_s} --cv ${cv} --seed ${seed} --nb_epochs ${nb_epochs} --lr ${lr} --sched ${sched} \
	--su_ratio ${su_ratio} --tag ${tag_prefix}_fm --bsize_u ${bsize_u} --threshold ${threshold}

# - fixmatch + mixup
./fm_exp.sh \
	--dataset ${dataset} --logdir ${logdir} --bsize_s ${bsize_s} --cv ${cv} --seed ${seed} --nb_epochs ${nb_epochs} --lr ${lr} --sched ${sched} \
	--su_ratio ${su_ratio} --tag ${tag_prefix}_fm_mix --bsize_u ${bsize_u} --threshold ${threshold} --use_mixup 1 --alpha ${ss_alpha}

fi

# -----------------------------------
# UBS8K
# -----------------------------------
if [ ${run_ubs8k} = 1 ]
then

dataset="UBS8K"
cv=1
bsize_s=128
bsize_u=128
logdir="/users/samova/elabbe/root_sslh/tensorboard/UBS8K/paper/"
threshold=0.95

# Supervised trainings
# - base 
./su_exp.sh \
	--dataset ${dataset} --logdir ${logdir} --bsize_s ${bsize_s} --cv ${cv} --seed ${seed} --nb_epochs ${nb_epochs} --lr ${lr} --sched ${sched} \
	--su_ratio ${su_ratio} --tag ${tag_prefix}_su
./su_exp.sh \
	--dataset ${dataset} --logdir ${logdir} --bsize_s ${bsize_s} --cv ${cv} --seed ${seed} --nb_epochs ${nb_epochs} --lr ${lr} --sched ${sched} \
	--su_ratio ${full_rat} --tag ${tag_prefix}_su

# - mixup
./su_exp.sh \
	--dataset ${dataset} --logdir ${logdir} --bsize_s ${bsize_s} --cv ${cv} --seed ${seed} --nb_epochs ${nb_epochs} --lr ${lr} --sched ${sched} \
	--su_ratio ${su_ratio} --tag ${tag_prefix}_su_mix --use_mixup 1 --alpha ${su_alpha}
./su_exp.sh \
	--dataset ${dataset} --logdir ${logdir} --bsize_s ${bsize_s} --cv ${cv} --seed ${seed} --nb_epochs ${nb_epochs} --lr ${lr} --sched ${sched} \
	--su_ratio ${full_rat} --tag ${tag_prefix}_su_mix --use_mixup 1 --alpha ${su_alpha}

# - weak
./su_exp.sh \
	--dataset ${dataset} --logdir ${logdir} --bsize_s ${bsize_s} --cv ${cv} --seed ${seed} --nb_epochs ${nb_epochs} --lr ${lr} --sched ${sched} \
	--su_ratio ${su_ratio} --tag ${tag_prefix}_su_w2 --augm_none weak
./su_exp.sh \
	--dataset ${dataset} --logdir ${logdir} --bsize_s ${bsize_s} --cv ${cv} --seed ${seed} --nb_epochs ${nb_epochs} --lr ${lr} --sched ${sched} \
	--su_ratio ${full_rat} --tag ${tag_prefix}_su_w2 --augm_none weak

# - weak + mixup
./su_exp.sh \
	--dataset ${dataset} --logdir ${logdir} --bsize_s ${bsize_s} --cv ${cv} --seed ${seed} --nb_epochs ${nb_epochs} --lr ${lr} --sched ${sched} \
	--su_ratio ${su_ratio} --tag ${tag_prefix}_su_w2_mix --augm_none weak --use_mixup 1 --alpha ${su_alpha}
./su_exp.sh \
	--dataset ${dataset} --logdir ${logdir} --bsize_s ${bsize_s} --cv ${cv} --seed ${seed} --nb_epochs ${nb_epochs} --lr ${lr} --sched ${sched} \
	--su_ratio ${full_rat} --tag ${tag_prefix}_su_w2_mix --augm_none weak --use_mixup 1 --alpha ${su_alpha}

# - strong
./su_exp.sh \
	--dataset ${dataset} --logdir ${logdir} --bsize_s ${bsize_s} --cv ${cv} --seed ${seed} --nb_epochs ${nb_epochs} --lr ${lr} --sched ${sched} \
	--su_ratio ${su_ratio} --tag ${tag_prefix}_su_s2 --augm_none strong
./su_exp.sh \
	--dataset ${dataset} --logdir ${logdir} --bsize_s ${bsize_s} --cv ${cv} --seed ${seed} --nb_epochs ${nb_epochs} --lr ${lr} --sched ${sched} \
	--su_ratio ${full_rat} --tag ${tag_prefix}_su_s2 --augm_none strong

# - strong + mixup
./su_exp.sh \
	--dataset ${dataset} --logdir ${logdir} --bsize_s ${bsize_s} --cv ${cv} --seed ${seed} --nb_epochs ${nb_epochs} --lr ${lr} --sched ${sched} \
	--su_ratio ${su_ratio} --tag ${tag_prefix}_su_s2_mix --augm_none strong --use_mixup 1 --alpha ${su_alpha}
./su_exp.sh \
	--dataset ${dataset} --logdir ${logdir} --bsize_s ${bsize_s} --cv ${cv} --seed ${seed} --nb_epochs ${nb_epochs} --lr ${lr} --sched ${sched} \
	--su_ratio ${full_rat} --tag ${tag_prefix}_su_s2_mix --augm_none strong --use_mixup 1 --alpha ${su_alpha}

# Semi-Supervised trainings
# - mixmatch
./mm_exp.sh \
	--dataset ${dataset} --logdir ${logdir} --bsize_s ${bsize_s} --cv ${cv} --seed ${seed} --nb_epochs ${nb_epochs} --lr ${lr} --sched ${sched} \
	--su_ratio ${su_ratio} --tag ${tag_prefix}_mm --bsize_u ${bsize_u} --alpha ${ss_alpha} --use_warmup_by_epoch ${use_warmup_by_epoch} --warmup_nb_steps ${warmup_nb_steps}

# - mixmatch + no mixup
./mm_exp.sh \
	--dataset ${dataset} --logdir ${logdir} --bsize_s ${bsize_s} --cv ${cv} --seed ${seed} --nb_epochs ${nb_epochs} --lr ${lr} --sched ${sched} \
	--su_ratio ${su_ratio} --tag ${tag_prefix}_mm_no_mix --bsize_u ${bsize_u} --use_no_mixup 1 --use_warmup_by_epoch ${use_warmup_by_epoch} --warmup_nb_steps ${warmup_nb_steps}

# - fixmatch
./fm_exp.sh \
	--dataset ${dataset} --logdir ${logdir} --bsize_s ${bsize_s} --cv ${cv} --seed ${seed} --nb_epochs ${nb_epochs} --lr ${lr} --sched ${sched} \
	--su_ratio ${su_ratio} --tag ${tag_prefix}_fm --bsize_u ${bsize_u} --threshold ${threshold}

# - fixmatch + mixup
./fm_exp.sh \
	--dataset ${dataset} --logdir ${logdir} --bsize_s ${bsize_s} --cv ${cv} --seed ${seed} --nb_epochs ${nb_epochs} --lr ${lr} --sched ${sched} \
	--su_ratio ${su_ratio} --tag ${tag_prefix}_fm_mix --bsize_u ${bsize_u} --threshold ${threshold} --use_mixup 1 --alpha ${ss_alpha}

fi

# -----------------------------------
# GSC
# -----------------------------------
if [ ${run_gsc} = 1 ]
then

dataset="GSC"
cv=0
bsize_s=128
bsize_u=128
logdir="/users/samova/elabbe/root_sslh/tensorboard/GSC/paper/"
threshold=0.8

# Supervised trainings
# - base 
./su_exp.sh \
	--dataset ${dataset} --logdir ${logdir} --bsize_s ${bsize_s} --cv ${cv} --seed ${seed} --nb_epochs ${nb_epochs} --lr ${lr} --sched ${sched} \
	--su_ratio ${su_ratio} --tag ${tag_prefix}_su
./su_exp.sh \
	--dataset ${dataset} --logdir ${logdir} --bsize_s ${bsize_s} --cv ${cv} --seed ${seed} --nb_epochs ${nb_epochs} --lr ${lr} --sched ${sched} \
	--su_ratio ${full_rat} --tag ${tag_prefix}_su

# - mixup
./su_exp.sh \
	--dataset ${dataset} --logdir ${logdir} --bsize_s ${bsize_s} --cv ${cv} --seed ${seed} --nb_epochs ${nb_epochs} --lr ${lr} --sched ${sched} \
	--su_ratio ${su_ratio} --tag ${tag_prefix}_su_mix --use_mixup 1 --alpha ${su_alpha}
./su_exp.sh \
	--dataset ${dataset} --logdir ${logdir} --bsize_s ${bsize_s} --cv ${cv} --seed ${seed} --nb_epochs ${nb_epochs} --lr ${lr} --sched ${sched} \
	--su_ratio ${full_rat} --tag ${tag_prefix}_su_mix --use_mixup 1 --alpha ${su_alpha}

# - weak
./su_exp.sh \
	--dataset ${dataset} --logdir ${logdir} --bsize_s ${bsize_s} --cv ${cv} --seed ${seed} --nb_epochs ${nb_epochs} --lr ${lr} --sched ${sched} \
	--su_ratio ${su_ratio} --tag ${tag_prefix}_su_w2 --augm_none weak
./su_exp.sh \
	--dataset ${dataset} --logdir ${logdir} --bsize_s ${bsize_s} --cv ${cv} --seed ${seed} --nb_epochs ${nb_epochs} --lr ${lr} --sched ${sched} \
	--su_ratio ${full_rat} --tag ${tag_prefix}_su_w2 --augm_none weak

# - weak + mixup
./su_exp.sh \
	--dataset ${dataset} --logdir ${logdir} --bsize_s ${bsize_s} --cv ${cv} --seed ${seed} --nb_epochs ${nb_epochs} --lr ${lr} --sched ${sched} \
	--su_ratio ${su_ratio} --tag ${tag_prefix}_su_w2_mix --augm_none weak --use_mixup 1 --alpha ${su_alpha}
./su_exp.sh \
	--dataset ${dataset} --logdir ${logdir} --bsize_s ${bsize_s} --cv ${cv} --seed ${seed} --nb_epochs ${nb_epochs} --lr ${lr} --sched ${sched} \
	--su_ratio ${full_rat} --tag ${tag_prefix}_su_w2_mix --augm_none weak --use_mixup 1 --alpha ${su_alpha}

# - strong
./su_exp.sh \
	--dataset ${dataset} --logdir ${logdir} --bsize_s ${bsize_s} --cv ${cv} --seed ${seed} --nb_epochs ${nb_epochs} --lr ${lr} --sched ${sched} \
	--su_ratio ${su_ratio} --tag ${tag_prefix}_su_s2 --augm_none strong
./su_exp.sh \
	--dataset ${dataset} --logdir ${logdir} --bsize_s ${bsize_s} --cv ${cv} --seed ${seed} --nb_epochs ${nb_epochs} --lr ${lr} --sched ${sched} \
	--su_ratio ${full_rat} --tag ${tag_prefix}_su_s2 --augm_none strong

# - strong + mixup
./su_exp.sh \
	--dataset ${dataset} --logdir ${logdir} --bsize_s ${bsize_s} --cv ${cv} --seed ${seed} --nb_epochs ${nb_epochs} --lr ${lr} --sched ${sched} \
	--su_ratio ${su_ratio} --tag ${tag_prefix}_su_s2_mix --augm_none strong --use_mixup 1 --alpha ${su_alpha}
./su_exp.sh \
	--dataset ${dataset} --logdir ${logdir} --bsize_s ${bsize_s} --cv ${cv} --seed ${seed} --nb_epochs ${nb_epochs} --lr ${lr} --sched ${sched} \
	--su_ratio ${full_rat} --tag ${tag_prefix}_su_s2_mix --augm_none strong --use_mixup 1 --alpha ${su_alpha}

# Semi-Supervised trainings
# - mixmatch
./mm_exp.sh \
	--dataset ${dataset} --logdir ${logdir} --bsize_s ${bsize_s} --cv ${cv} --seed ${seed} --nb_epochs ${nb_epochs} --lr ${lr} --sched ${sched} \
	--su_ratio ${su_ratio} --tag ${tag_prefix}_mm --bsize_u ${bsize_u} --alpha ${ss_alpha} --use_warmup_by_epoch ${use_warmup_by_epoch} --warmup_nb_steps ${warmup_nb_steps}

# - mixmatch + no mixup
./mm_exp.sh \
	--dataset ${dataset} --logdir ${logdir} --bsize_s ${bsize_s} --cv ${cv} --seed ${seed} --nb_epochs ${nb_epochs} --lr ${lr} --sched ${sched} \
	--su_ratio ${su_ratio} --tag ${tag_prefix}_mm_no_mix --bsize_u ${bsize_u} --use_no_mixup 1 --use_warmup_by_epoch ${use_warmup_by_epoch} --warmup_nb_steps ${warmup_nb_steps}

# - fixmatch
./fm_exp.sh \
	--dataset ${dataset} --logdir ${logdir} --bsize_s ${bsize_s} --cv ${cv} --seed ${seed} --nb_epochs ${nb_epochs} --lr ${lr} --sched ${sched} \
	--su_ratio ${su_ratio} --tag ${tag_prefix}_fm --bsize_u ${bsize_u} --threshold ${threshold}

# - fixmatch + mixup
./fm_exp.sh \
	--dataset ${dataset} --logdir ${logdir} --bsize_s ${bsize_s} --cv ${cv} --seed ${seed} --nb_epochs ${nb_epochs} --lr ${lr} --sched ${sched} \
	--su_ratio ${su_ratio} --tag ${tag_prefix}_fm_mix --bsize_u ${bsize_u} --threshold ${threshold} --use_mixup 1 --alpha ${ss_alpha}

fi
