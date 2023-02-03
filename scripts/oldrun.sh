#!/bin/sh

# --- PARAMS
function get_hydra_value() {
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
            result=`echo $arg | grep "$pattern"`
            if [ ! -z "$result" ]; then
                value=`echo "$arg" | cut -d "=" -f2`
            fi
        fi
        it=$(expr $it + 1)
    done

    echo "${value}"
    return 0
}

usage="$0 METHOD [path=(local|osi|oly|arai)] PARAMS... \nExample: \n\t$0 supervised path=local bsize=128 epochs=10\n\n"

dpath_sbatch=`realpath $0 | xargs dirname`
path=`get_hydra_value "path" "NOT_FOUND" $@`

# --- RUN
if [ "${path}" = "local" ] || [ "${path}" = "arai" ]; then
	${dpath_sbatch}/run_local.sh $@
elif [ "${path}" = "osi" ]; then
	${dpath_sbatch}/run_osirim.sh $@
elif [ "${path}" = "oly" ]; then
	${dpath_sbatch}/run_olympe.sh $@
else
	printf "${usage}"
fi

exit 0
