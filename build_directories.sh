#! /bin/sh

root=`pwd`

for dir in logs boards
do
  for dataset in ADS CIFAR10 ESC10 GSC PVC UBS8K
  do
    echo "Building directory '$root/$dir/$dataset' ..."
    mkdir -p $root/$dir/$dataset
  done
done

echo "Script '$0' done."
exit 0
