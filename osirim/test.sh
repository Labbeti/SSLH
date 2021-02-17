echo "START"
echo "$@"

was_suffix=false
was_run=false

run=1
if [ $run = 1 ]
then
echo "COUCOU"
fi

exit 0


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
