
dataset="UBS8K"

./su.sh --dataset "$dataset" --supervised_ratio 0.1 $@
./su.sh --dataset "$dataset" --supervised_ratio 1.0 $@
./mm.sh --dataset "$dataset" --supervised_ratio 0.1 $@
./fm.sh --dataset "$dataset" --supervised_ratio 0.1 $@
./uda.sh --dataset "$dataset" --supervised_ratio 0.1 $@

./su_exp.sh --dataset "$dataset" --supervised_ratio 0.1 --use_mixup 1 $@
./su_exp.sh --dataset "$dataset" --supervised_ratio 1.0 --use_mixup 1 $@
./fm_exp.sh --dataset "$dataset" --supervised_ratio 0.1 --use_mixup 1 $@
./uda_exp.sh --dataset "$dataset" --supervised_ratio 0.1 --use_mixup 1 $@

./fm_exp.sh --dataset "$dataset" --supervised_ratio 0.1 --use_uniloss 1 $@