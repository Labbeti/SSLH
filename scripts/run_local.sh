#!/bin/sh

dpath_parent=`realpath $0 | xargs dirname`
fname_script="$1.py"
script_params=`$dpath_parent/get_skip_params.sh 1 $@`

path=`$dpath_parent/get_param.sh "path" "NOT_FOUND" $@`
if [ "$path" = "NOT_FOUND" ]; then
  path="default"
  script_params="$script_params path=$path"
fi

tag=`$dpath_parent/get_param.sh "tag" "NOTAG" $@`

dpath_project=`echo $dpath_parent | xargs dirname`

fpath_python="python"
dpath_standalone="${dpath_project}/sslh"
fpath_script="${dpath_standalone}/${fname_script}"

echo "Run script '${fname_script}' with tag '${tag}'"
${fpath_python} ${fpath_script} ${script_params}

exit 0
