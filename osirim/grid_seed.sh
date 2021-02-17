
suffix="_RAdam"

start_seed=1234
i=0
end=5

ds="ESC10"
logdir="/users/samova/elabbe/root_sslh/tensorboard/ESC10/seed_test/"
ds_path="`./get_ds_path.sh $ds`"

while [ $i -lt $end ]; do
	seed=$(($i+$start_seed))
	
	./su.sh --dataset $ds --dataset_path $ds_path --su_ratio 0.1 --lr 3e-3 --seed $seed --tag seed_script_$seed$suffix --logdir $logdir $@
	
	i=$(($i+1))
done
