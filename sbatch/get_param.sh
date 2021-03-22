#!/bin/sh

name="$1"
default_value="$2"
pattern="^${name}=.*"

value="${default_value}"
found=false

nb_skip_params=2
it=0

for arg in $@
do
	if [ $it -ge ${nb_skip_params} ]; then
	  result=`echo $arg | grep $pattern`
    if [ ! -z "$result" ]; then
      value=`echo $arg | cut -d "=" -f2`
    fi
  fi
  it=$(expr $it + 1)
done

echo "${value}"
exit 0
