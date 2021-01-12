
dataset="ESC10"
cross_val=1
bsize_s=30
bsize_u=30

su_ratio=0.1
full_rat=1.0

./su_exp.sh --dataset ${dataset} --bsize_s ${bsize_s} --cross_val ${cross_val} --su_ratio ${su_ratio} --tag su_paper
./su_exp.sh --dataset ${dataset} --bsize_s ${bsize_s} --cross_val ${cross_val} --su_ratio ${full_rat} --tag su_paper

./su_exp.sh --dataset ${dataset} --bsize_s ${bsize_s} --cross_val ${cross_val} --su_ratio ${su_ratio} --tag su_mix_paper --use_mixup 1
./su_exp.sh --dataset ${dataset} --bsize_s ${bsize_s} --cross_val ${cross_val} --su_ratio ${full_rat} --tag su_mix_paper --use_mixup 1

./su_exp.sh --dataset ${dataset} --bsize_s ${bsize_s} --cross_val ${cross_val} --su_ratio ${su_ratio} --tag su_w2_paper --augm_none weak
./su_exp.sh --dataset ${dataset} --bsize_s ${bsize_s} --cross_val ${cross_val} --su_ratio ${full_rat} --tag su_w2_paper --augm_none weak

./su_exp.sh --dataset ${dataset} --bsize_s ${bsize_s} --cross_val ${cross_val} --su_ratio ${su_ratio} --tag su_mix_w2_paper --use_mixup 1 --augm_none weak
./su_exp.sh --dataset ${dataset} --bsize_s ${bsize_s} --cross_val ${cross_val} --su_ratio ${full_rat} --tag su_mix_w2_paper --use_mixup 1 --augm_none weak

./mm_exp.sh --dataset ${dataset} --bsize_s ${bsize_s} --cross_val ${cross_val} --su_ratio ${su_ratio} --tag mm_paper --bsize_u ${bsize_u}

./fm_exp.sh --dataset ${dataset} --bsize_s ${bsize_s} --cross_val ${cross_val} --su_ratio ${su_ratio} --tag fm_paper --bsize_u ${bsize_u}

./fm_exp.sh --dataset ${dataset} --bsize_s ${bsize_s} --cross_val ${cross_val} --su_ratio ${su_ratio} --tag fm_mix_paper --use_mixup 1 --bsize_u ${bsize_u}
