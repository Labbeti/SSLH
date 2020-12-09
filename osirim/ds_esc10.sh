
dataset="ESC10"

./su.sh --dataset "$dataset" --supervised_ratio 0.1 --bsize_s 30 $@
./su.sh --dataset "$dataset" --supervised_ratio 1.0 --bsize_s 30 $@
./mm.sh --dataset "$dataset" --supervised_ratio 0.1 --bsize_s 30 --bsize_u 30 $@
./fm.sh --dataset "$dataset" --supervised_ratio 0.1 --bsize_s 30 --bsize_u 30 $@
./uda.sh --dataset "$dataset" --supervised_ratio 0.1 --bsize_s 30 --bsize_u 30 $@

./su_exp.sh --dataset "$dataset" --supervised_ratio 0.1 --use_mixup 1 $@
./su_exp.sh --dataset "$dataset" --supervised_ratio 1.0 --use_mixup 1 $@
./fm_exp.sh --dataset "$dataset" --supervised_ratio 0.1 --use_mixup 1 $@
./uda_exp.sh --dataset "$dataset" --supervised_ratio 0.1 --use_mixup 1 $@

./fm_exp.sh --dataset "$dataset" --supervised_ratio 0.1 --use_uniloss 1 $@