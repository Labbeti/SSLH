
# sched_coef = 7/16 = 0.4375

./fm_exp.sh \
	--dataset CIFAR10 \
	--lambda_u 1.0 \
	--threshold 0.95 \
	--bsize_s 64 \
	--bsize_u 448 \
	--optim SGD \
	--lr 0.001 \
	--weight_decay 0.0005 \
	--momentum 0.9 \
	--nesterov True \
	--sched CosineLRScheduler \
	--su_ratio 0.08 \
	--augm_weak "weak4" \
	--augm_strong "strong4" \
	--standardize True \
	--epochs 1000 \
	--zip_policy "max" \
	--use_soft_reduce_u False \
	--sched_coef 0.4375 \
	--sched_nb_steps None \
	--tag "CLASSIC_FM_lr_0.001" \
