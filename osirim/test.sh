echo "START"
echo "$@"
# cat $@

was_suffix=false
was_run=false

for arg in $@
do
	if [ $was_suffix = true ]
	then
		suffix="$arg"
		was_suffix=false
	elif [ "$was_run" = true ]
	then
		run="$arg"
		was_run=false
	fi
	case $arg in
		"--run" )
			was_run=true;;
		"--suffix" )
			was_suffix=true;;
	esac
done

echo "$run"
echo "$suffix"
echo "END"

exit $run $suffix
