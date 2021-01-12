
for threshold in 0.5 0.65 0.8 0.95
do
	for alpha in 0.0 0.25 0.5 0.75 1.0 1.5 2.0
	do
		./fm_exp.sh --dataset GSC --use_mixup 1 --augm_weak weak2 --augm_strong strong2 --epochs 300 --alpha ${alpha} --threshold ${threshold} --tag GRID_FM_ALPHA_alpha_${alpha}_threshold_${threshold}
	done
done
