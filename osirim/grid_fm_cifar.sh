
dataset="CIFAR10"
su_ratio=0.08
seed=1234

nb_epochs=200
bsize_s=64
bsize_u=448

optim="SGD"
lr=0.1
wd=0.0005
momentum=0.9
nesterov=0
sched="MultiStepLR"

lambda_u=1.0
threshold=0.95

for threhsold in 0.95
do
	# 128 256 448
	for bsize_u in 448
	do
		for lr in 0.1 0.03 0.001
		do
			for wd in 0.0 0.0005
			do
				for sched in MultiStepLR CosineLRScheduler
				do
					optim="SGD"
					nesterov=0
					./fm.sh \
						--dataset ${dataset} \
						--su_ratio ${su_ratio} \
						--seed ${seed} \
						--nb_epochs ${nb_epochs} \
						--bsize_s ${bsize_s} \
						--bsize_u ${bsize_u} \
						--optim ${optim} \
						--lr ${lr} \
						--wd ${wd} \
						--momentum ${momentum} \
						--nesterov ${nesterov} \
						--sched ${sched} \
						--lambda_u ${lambda_u} \
						--threshold ${threshold} \
						--tag "GRID_FM_threshold_${threshold}_bsize_u_${bsize_u}_lr_${lr}_wd_${wd}_sched_${sched}_optim_${optim}_nesterov_${nesterov}"
						
					optim="SGD"
					nesterov=1
					./fm.sh \
						--dataset ${dataset} \
						--su_ratio ${su_ratio} \
						--seed ${seed} \
						--nb_epochs ${nb_epochs} \
						--bsize_s ${bsize_s} \
						--bsize_u ${bsize_u} \
						--optim ${optim} \
						--lr ${lr} \
						--wd ${wd} \
						--momentum ${momentum} \
						--nesterov ${nesterov} \
						--sched ${sched} \
						--lambda_u ${lambda_u} \
						--threshold ${threshold} \
						--tag "GRID_FM_threshold_${threshold}_bsize_u_${bsize_u}_lr_${lr}_wd_${wd}_sched_${sched}_optim_${optim}_nesterov_${nesterov}"
						
					optim="Adam"
					nesterov=0
					./fm.sh \
						--dataset ${dataset} \
						--su_ratio ${su_ratio} \
						--seed ${seed} \
						--nb_epochs ${nb_epochs} \
						--bsize_s ${bsize_s} \
						--bsize_u ${bsize_u} \
						--optim ${optim} \
						--lr ${lr} \
						--wd ${wd} \
						--momentum ${momentum} \
						--nesterov ${nesterov} \
						--sched ${sched} \
						--lambda_u ${lambda_u} \
						--threshold ${threshold} \
						--tag "GRID_FM_threshold_${threshold}_bsize_u_${bsize_u}_lr_${lr}_wd_${wd}_sched_${sched}_optim_${optim}_nesterov_${nesterov}"
				done
			done
		done
	done
done

