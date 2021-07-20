#! /bin/sh

root=`pwd`

for dir in logs boards
do
  for dataset in ADS CIFAR10 ESC10 FSD50K GSC PVC UBS8K
  do
    mkdir -p $root/$dir/$dataset
  done
done

mkdir -p $root/logs/sbatch

echo "Script '$0' done."
exit 0
