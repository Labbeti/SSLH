
./fm.sh --dataset "CIFAR10" --supervised_ratio 0.08 $@
./fm.sh --dataset "UBS8K" --supervised_ratio 0.1 $@
./fm.sh --dataset "ESC10" --supervised_ratio 0.1 --bsize_s 30 --bsize_u 30 $@
./fm.sh --dataset "GSC" --supervised_ratio 0.1 $@
