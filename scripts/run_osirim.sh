#!/bin/sh
# -*- coding: utf-8 -*-

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

function get_skip_params() {
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
    return 0
}

dn0=`dirname $0`
dpath_project=`realpath $0 | xargs dirname | xargs dirname`
fpath_python="${HOME}/miniconda3/envs/env_sslh/bin/python"
fpath_singularity="/logiciels/containerCollections/CUDA11/pytorch-NGC-21-03-py3.sif"
srun="srun singularity exec ${fpath_singularity}"
default_cpus="5"
default_path="osirim"
module_load="module load singularity/3.0.3"
dpath_log="${dpath_project}/logs/slurm"

fname_script="$1.py"
script_params=`get_skip_params 1 $@`

cpus=`get_hydra_value "cpus" "NOT_FOUND" $@`
data=`get_hydra_value "data" "NOT_FOUND" $@`
datetime=`get_hydra_value "datetime" "NOT_FOUND" $@`
gpus=`get_hydra_value "gpus" "1" $@`
partition=`get_hydra_value "+partition" "NOT_FOUND" $@`
path=`get_hydra_value "path" "NOT_FOUND" $@`
tag=`get_hydra_value "tag" "NOT_FOUND" $@`

job_name="%j-${tag}"

if [ "${cpus}" = "NOT_FOUND" ]; then
    cpus="${default_cpus}"
    script_params="${script_params} cpus=${cpus}"
fi
if [ "${datetime}" = "NOT_FOUND" ]; then
    datetime=`date +"%F_%H-%M-%S"`
    script_params="${script_params} datetime=${datetime}"
fi
if [ "${path}" = "NOT_FOUND" ]; then
    path="${default_path}"
    script_params="$script_params path=$path"
fi
if [ "${tag}" = "NOT_FOUND" ]; then
    tag="NOTAG"
    script_params="${script_params} tag=${tag}"
fi

fpath_script="${dpath_project}/src/sslh/${fname_script}"

fpath_out="${dpath_log}/${job_name}.out"
fpath_err="${fpath_out}"

mkdir -p ${dpath_log}

if [ "${partition}" = "NOT_FOUND" ]; then
    if [ "${gpus}" -eq "0" ]; then
        partition="48CPUNodes"
    else
        partition="GPUNodes" # GPUNodes, RTX6000Node
    fi
fi

if [ "${partition}" = "24CPUNodes" ]; then
    mem_per_cpu="7500M"
elif [ "${partition}" = "48CPUNodes" ]; then
    mem_per_cpu="10000M"
elif [ "${partition}" = "64CPUNodes" ]; then
    mem_per_cpu="8000M"
elif [ "${partition}" = "GPUNodes" ]; then
    mem_per_cpu="9000M"
elif [ "${partition}" = "RTX6000Node" ]; then
    mem_per_cpu="4500M"
else
    echo "Invalid partition ${partition} for ${path}. (expected 24CPUNodes, 48CPUNodes, 64CPUNodes, GPUNodes or RTX6000Node)"
    exit 1
fi

# Build sbatch file ----------------------------------------------------------------------------------------------------
# Memory format : number[K|M|G|T]. If 0, no memory limit, use all of node.
mem=""
# Time format : days-hours:minutes:seconds. If 0, no time limit.
time="0"

fpath_sbatch=".tmp_${job_name}.sbatch"

slurm_params="+slurm.output=${fpath_out} +slurm.error=${fpath_err} +slurm.partition=${partition} +slurm.mem_per_cpu=${mem_per_cpu} +slurm.mem=${mem} +slurm.time=${time}"
extended_script_params="${script_params} ${slurm_params}"

cat << EOT > ${fpath_sbatch}
#!/bin/sh

# Minimal number of nodes (equiv: -N)
#SBATCH --nodes=1

# Number of tasks (equiv: -n)
#SBATCH --ntasks=1

# Job name (equiv: -J)
#SBATCH --job-name=${job_name}

# Log output file
#SBATCH --output=${fpath_out}

# Log err file
#SBATCH --error=${fpath_err}

# Number of CPU per task
#SBATCH --cpus-per-task=${cpus}

# Memory limit (0 means no limit) -- DISABLED
## #SBATCH --mem=${mem}

# Memory per cpu
#SBATCH --mem-per-cpu=${mem_per_cpu}

# Duration limit (0 means no limit)
#SBATCH --time=${time}

# Mail for optional auto-sends -- DISABLED
## #SBATCH --mail-user=""

# Select partition
#SBATCH --partition=${partition}

# For GPU nodes, select the number of GPUs
#SBATCH --gres=gpu:${gpus}

# For GPU nodes, force job to start only when CPU and GPU are all available
#SBATCH --gres-flags=enforce-binding


# For testing the sbatch file -- DISABLED
## #SBATCH --test-only

# Specify a node list -- DISABLED
## #SBATCH --nodelist=gpu-nc04

# Others -- DISABLED
## #SBATCH --ntasks-per-node=4
## #SBATCH --ntasks-per-core=1
## #SBATCH --mail-type=END


module purge
${module_load}

${srun} ${fpath_python} ${fpath_script} ${extended_script_params}

EOT

# --- RUN
mkdir -p "${dn0}/cache/start_logs"
echo "Sbatch job '${job_name}' with tag '${tag}'" | tee -a "${dn0}/cache/run_osirim_logs.txt"
sbatch ${fpath_sbatch}

exit 0
