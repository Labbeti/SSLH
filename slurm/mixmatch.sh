#!/bin/sh

usage="Usage: $0 (run|run_olympe|run_osirim) [SCRIPT_PARAMS]"

dpath_parent=`realpath $0 | xargs dirname`
script_name="`basename $0`"
script_name=`echo ${script_name} | sed "s/.sh/.py/g"`
run="$1"
script_params=`$dpath_parent/get_skip_params.sh 1 $@`

if [ "$1" = "run" ]; then
  $dpath_parent/run.sh ${script_name} ${script_params}
elif [ "$1" = "run_olympe" ]; then
  $dpath_parent/run_olympe.sh ${script_name} ${script_params}
elif [ "$1" = "run_osirim" ]; then
  $dpath_parent/run_osirim.sh ${script_name} ${script_params}
else
  echo "$usage"
fi

exit 0
