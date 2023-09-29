#!/bin/bash
# -*- coding: utf-8 -*-

# SBATCH MAKER MAIN SCRIPT.

if [ "${HOSTNAME}" = "co2-slurm-client1" ]; then
    NAME="OSIRIM"
    DEFAULT_LAUNCHER="osi"
    PYTHON_FPATH="${HOME}/miniconda3/envs/env_sslh/bin/python"

elif [[ "${HOSTNAME}" = jean-zay[0-9] ]]; then
    NAME="JEAN-ZAY"
    DEFAULT_LAUNCHER="jz"
    PYTHON_FPATH="${HOME}/.conda/envs/env_sslh/bin/python"

elif [ "${HOSTNAME}" = "araigne" ]; then
    NAME="ARAIGNE"
    DEFAULT_LAUNCHER="araigne"
    PYTHON_FPATH="${HOME}/miniconda3/envs/env_sslh/bin/python"

elif [ "${HOSTNAME}" = "olympelogin1.bullx" ]; then
    NAME="OLYMPE"
    DEFAULT_LAUNCHER="oly"
    PYTHON_FPATH="/tmpdir/${USER}/miniconda3/envs/env_sslh/bin/python"

else
    echo "Cannot detect hostname '${HOSTNAME}'."
    exit 1
fi

# --- Prepare params
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

function get_bash_value() {
    name="$1"
    default_value="$2"

    value="${default_value}"
    found="false"
    i=0

    for arg in $@
    do
        if [ $i -lt 2 ]; then
            true
        elif [ "${found}" = "true" ]; then
            found="false"
            value="$arg"
            break
        elif [ "${arg}" = "${name}" ]; then
            found=true
        fi
        i=$(expr $i + 1)
    done

    echo "${value}"
    return 0
}

function get_hydra_args() {
    skip_next=false
    hydra_args=""

    for arg in $@
    do
        if [[ "$arg" = --* ]]; then
            skip_next=true
        elif [ "$skip_next" = "true" ]; then
            skip_next=false
        else
            hydra_args="${hydra_args} ${arg}"
        fi
    done
    echo "${hydra_args}"
    exit 0
}

function get_bash_args() {
    pattern="^.*=.*"
    bash_args=""

    for arg in $@
    do
        is_hydra_arg=`echo ${arg} | grep "${pattern}"`
        if [ "${is_hydra_arg}" ]; then
            true
        else
            bash_args="${bash_args} ${arg}"
        fi
    done
    echo "${bash_args}"
    exit 0
}

function is_clean_repository() {
    curpath=`pwd -P`
    if [ "$1" != "" ]; then
        cd "$1"
    fi

    # Detecte les fichiers modifies mais pas add
    git diff --quiet --exit-code
    is_clean_modified=$?

    # Detecte les fichiers add mais pas commit
    git diff --quiet --exit-code --cached 
    is_clean_added=$?

    if [ ${is_clean_modified} -eq 0 ] && [ ${is_clean_added} -eq 0 ]
    then
        code=0
    else
        code=1
    fi

    cd "${curpath}"
    return ${code}
}

function yaml_get() {
    vari="$1"
    key="$2"
    value=`echo "${vari}" | grep "^${key}: "`
    echo "${value#*: }"  # note: print everything after ': ' pattern
    exit 0
}

dn0=`dirname $0`
bn0=`basename $0`
proj_dpath=`dirname ${dn0}`
git_hash=`cd "${proj_dpath}" && git describe --always`

hydra_args=`get_hydra_args $@`
bash_args=`get_bash_args $@`

hydra_args="launcher=${DEFAULT_LAUNCHER} git_hash=${git_hash} ${hydra_args}"

check_dirty_git=`get_hydra_value "+check_dirty_git" "true" ${hydra_args}`
config_name=`get_bash_value "--config-name" "supervised" ${bash_args}`

if [ "${check_dirty_git}" = "true" ]; then
    is_clean_repository "${proj_dpath}"
    is_clean=$?
    # note: code 0 means no error
    if [ ${is_clean} -ne 0 ]; then
        echo "[${bn0}] GIT CHECK ERROR: Cannot submit job with dirty git status. (path=${tgt_repo})"
        exit 1
    fi
fi

fpath_print_cfg="${proj_dpath}/src/sslh/print_cfg.py"
fpath_script="${proj_dpath}/src/sslh/${config_name}.py"

if [ ! -f "${fpath_script}" ]; then
    echo "Python script file does not exist. (${fpath_script})"
    exit 1
fi

# see https://stackoverflow.com/questions/65104134/disable-file-output-of-hydra
disabled_hydra_params="hydra.output_subdir=null hydra.run.dir=. hydra.sweep.dir=. hydra.sweep.subdir=."
output=`${PYTHON_FPATH} -O "${fpath_print_cfg}" ${bash_args} ${hydra_args} ${disabled_hydra_params} 2>&1`
exitcode=$?

if [ ${exitcode} -eq 0 ]; then
    cfg="${output}"
else
    echo "[${bn0}] Invalid arguments given to `basename ${fpath_print_cfg}`."
    echo "[${bn0}] Output:"
    echo "${output}"
    exit ${exitcode}
fi

# SBATCH config params
error=`yaml_get "${cfg}" "slurm.error"`
job_name=`yaml_get "${cfg}" "slurm.job_name"`
output=`yaml_get "${cfg}" "slurm.output"`

# SBATCH other params
gpus=`yaml_get "${cfg}" "slurm.gpus"`
module_cmds=`yaml_get "${cfg}" "slurm.module_cmds"`
sbatch=`yaml_get "${cfg}" "slurm.sbatch"`
srun=`yaml_get "${cfg}" "slurm.srun"`
test_only=`yaml_get "${cfg}" "slurm.test_only"`

# Other params
datetime=`yaml_get "${cfg}" "datetime"`
verbose=`yaml_get "${cfg}" "verbose"`

# Build log directories
dpath_log_out=`dirname ${output}`
dpath_log_err=`dirname ${error}`

mkdir -p "${dn0}/cache" "${dpath_log_out}" "${dpath_log_err}"

# Add the datetime of the print_cfg script
hydra_args="datetime=${datetime} ${hydra_args}"

# Read sbatch config values
# note: values are expected in config group "slurm" with the same name (except for '-' which is replaced by '_')
# example: "cpus-per-task" sbatch config value is in "slurm.cpus_per_task" hydra cfg value.
sbatch_config=(
    "constraint"
    "cpus-per-task"
    "distribution"
    "error"
    "gres"
    "gres-flags"
    "hint"
    "job-name"
    "mem",
    "mem-per-cpu"
    "nodes"
    "ntasks-per-node"
    "output"
    "partition"
    "qos"
    "time"
)

lines=()
i=0
while [ $i -lt ${#sbatch_config[@]} ]; do
    sbatch_key="${sbatch_config[$i]}"
    cfg_key="slurm.${sbatch_key//-/_}"  # note: replace '-' by "_"
    sbatch_value=`yaml_get "${cfg}" "${cfg_key}"`

    line=`[ "${sbatch_value}" != "null" ] && [ "${sbatch_value}" != "" ] && echo "\n#SBATCH --${sbatch_key}=${sbatch_value}"`
    lines+=("$line")

    i=`expr $i + 1`
done

if [ "${test_only}" = "true" ]; then
    lines+=("\n#SBATCH --test-only")
fi

# --- BUILD SBATCH FILE
fpath_sbatch="${dn0}/cache/${bn0}.sbatch"

cat << EOT > ${fpath_sbatch}
#!/bin/sh
# -*- coding: utf-8 -*-
`echo -e ${lines[@]}`

${module_cmds}

${srun} ${PYTHON_FPATH} -u -O ${fpath_script} ${bash_args} ${hydra_args}

EOT

if [ "${verbose}" -ge 2 ]; then
    echo "[$bn0] ${NAME} SBATCH:"
    cat -n ${fpath_sbatch}
fi

# --- RUN
echo "[$bn0] Submit job '${job_name}'." | tee -a "${dn0}/cache/run_${NAME}_logs.txt"
${sbatch} ${fpath_sbatch}

exit 0
