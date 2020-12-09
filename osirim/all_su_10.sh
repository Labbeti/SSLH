
./su.sh --dataset "CIFAR10" --supervised_ratio 0.08 $@
./su.sh --dataset "UBS8K" --supervised_ratio 0.1 $@
./su.sh --dataset "ESC10" --supervised_ratio 0.1 --bsize_s 30 $@
./su.sh --dataset "GSC" --supervised_ratio 0.1 $@
