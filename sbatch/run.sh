#!/bin/sh

fname_script="$1"
script_params=`./get_skip_params.sh 1 $@`

path=`./get_param.sh "path" "NOT_FOUND" $@`
if [ "$path" = "NOT_FOUND" ]; then
  path="default"
  script_params="$script_params path=$path"
fi

dpath_project=`pwd`
dpath_project=`dirname $dpath_project`

fpath_python="python"
dpath_standalone="${dpath_project}/standalone"
fpath_script="${dpath_standalone}/${fname_script}"

echo "Run script '${fname_script}'"
${fpath_python} ${fpath_script} ${script_params}

exit 0
