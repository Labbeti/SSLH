
./uda.sh --dataset "CIFAR10" --supervised_ratio 0.08 $@
./uda.sh --dataset "UBS8K" --supervised_ratio 0.1 $@
./uda.sh --dataset "ESC10" --supervised_ratio 0.1 --bsize_s 30 --bsize_u 30 $@
./uda.sh --dataset "GSC" --supervised_ratio 0.1 $@
