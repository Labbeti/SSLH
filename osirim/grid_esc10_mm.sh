
for lr in 1e-3 1e-4
do
	for augm_weak in weak2
	do
		for alpha in 0.0 0.1 0.2 0.3
		do
			for nb_augms in 2 1
			do
				cross_val=1
				tag="GRID_SEARCH_ESC10_MM_lr_${lr}_augm_weak_${augm_weak}_alpha_${alpha}_nb_augms_${nb_augms}_cross_val_${cross_val}"
				./mm_exp.sh --dataset ESC10 --lr $lr --augm_weak $augm_weak --alpha $alpha --nb_augms $nb_augms --cross_val $cross_val --bsize_s 30 --bsize_u 30 --tag $tag
			done
		done
	done
done
