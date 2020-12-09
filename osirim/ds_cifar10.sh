
dataset="CIFAR10"

./su.sh --dataset "$dataset" --supervised_ratio 0.08 $@
./su.sh --dataset "$dataset" --supervised_ratio 1.00 $@
./mm.sh --dataset "$dataset" --supervised_ratio 0.08 $@
./fm.sh --dataset "$dataset" --supervised_ratio 0.08 $@
./uda.sh --dataset "$dataset" --supervised_ratio 0.08 $@

./su_exp.sh --dataset "$dataset" --supervised_ratio 0.08 --use_mixup 1 $@
./su_exp.sh --dataset "$dataset" --supervised_ratio 1.00 --use_mixup 1 $@
./fm_exp.sh --dataset "$dataset" --supervised_ratio 0.08 --use_mixup 1 $@
./uda_exp.sh --dataset "$dataset" --supervised_ratio 0.08 --use_mixup 1 $@

./fm_exp.sh --dataset "$dataset" --supervised_ratio 0.08 --use_uniloss 1 $@
