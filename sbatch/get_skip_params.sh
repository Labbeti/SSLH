#!/bin/sh

if [ "$1" = "usage" ]; then
  echo "Usage: $0 NB_SKIP [PARAMS]"
  exit 0
fi

nb_skip_params=$(expr $1 + 1)
it=0
script_params=""

for arg in "$@"
do
	if [ $it -ge ${nb_skip_params} ]; then
		script_params="${script_params} ${arg}"
	fi
  it=$(expr $it + 1)
done

echo ${script_params}
