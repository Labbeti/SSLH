#!/bin/sh

usage="Usage: $0 (run|run_olympe|run_osirim) [SCRIPT_PARAMS]"

script_name="`basename $0`"
script_name=`echo ${script_name} | sed "s/.sh/.py/g"`
run="$1"
script_params=`./get_skip_params.sh 1 $@`

if [ "$1" = "run" ]; then
  ./run.sh ${script_name} ${script_params}
elif [ "$1" = "run_olympe" ]; then
  ./run_olympe.sh ${script_name} ${script_params}
elif [ "$1" = "run_osirim" ]; then
  ./run_osirim.sh ${script_name} ${script_params}
else
  echo "$usage"
fi

exit 0
