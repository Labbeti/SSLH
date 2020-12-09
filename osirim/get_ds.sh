
ds_name="None"
found=false

for arg in $@
do
	if [ $found = true ]
	then
		ds_name="$arg"
		found=false
	fi
	
	case $arg in
		"--dataset_name" )
			found=true;;
		"--dataset" )
			found=true;;
	esac
done

echo "$ds_name"

exit 0
