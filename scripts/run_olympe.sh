#!/bin/sh

fpath_python="/tmpdir/labbe/miniconda3/envs/env_sslh/bin/python"
fpath_singularity=""
srun="srun"
default_cpus="10"
default_path="olympe"
module_load="module load cuda/10.1.105"
dpath_log="/tmpdir/labbe/sslh_logs/slurm"

job_name="$1"
fname_script="$1.py"
script_params=`./get_skip_params.sh 1 $@`

cpus=`./get_param.sh "cpus" "NOT_FOUND" $@`
data=`./get_param.sh "data" "NOT_FOUND" $@`
datetime=`./get_param.sh "datetime" "NOT_FOUND" $@`
gpus=`./get_param.sh "gpus" "1" $@`
partition=`./get_param.sh "+partition" "NOT_FOUND" $@`
path=`./get_param.sh "path" "NOT_FOUND" $@`
tag=`./get_param.sh "tag" "NOT_FOUND" $@`

if [ "${cpus}" = "NOT_FOUND" ]; then
  cpus="${default_cpus}"
  script_params="${script_params} cpus=${cpus}"
fi
if [ "${datetime}" = "NOT_FOUND" ]; then
  datetime=`date +"%F_%H-%M-%S"`
  script_params="${script_params} datetime=${datetime}"
fi
if [ "$path" = "NOT_FOUND" ]; then
  path="${default_path}"
  script_params="$script_params path=$path"
fi
if [ "${tag}" = "NOT_FOUND" ]; then
  tag="NOTAG"
  script_params="${script_params} tag=${tag}"
fi

dpath_project=`realpath $0 | xargs dirname | xargs dirname`
fpath_script="${dpath_project}/sslh/${fname_script}"

fpath_out="${dpath_log}/${data}_${job_name}_%j_${tag}.out"
fpath_err="${dpath_log}/${data}_${job_name}_%j_${tag}.err"

mkdir -p ${dpath_log}

if [ "${partition}" = "NOT_FOUND" ]; then
    partition=""
fi

mem_per_cpu=""

# Build sbatch file ----------------------------------------------------------------------------------------------------
# Memory format : number[K|M|G|T]. If 0, no memory limit, use all of node.
# note on olympe : https://www.calmip.univ-toulouse.fr/spip.php?article738
mem="64G"
# Time format : days-hours:minutes:seconds. If 0, no time limit.
time="2-00:00:00"

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

# Memory limit (0 means no limit)
#SBATCH --mem=${mem}

# Memory per cpu -- DISABLED
## #SBATCH --mem-per-cpu=${mem_per_cpu}

# Duration limit (0 means no limit)
#SBATCH --time=${time}

# Mail for optional auto-sends
#SBATCH --mail-user="etienne.labbe@irit.fr"

# Select partition -- DISABLED
## #SBATCH --partition=${partition}

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
mkdir -p "${dpath_log}/start_logs"
echo "Sbatch job '${job_name}' with tag '${tag}'" | tee -a "${dpath_log}/start_logs/run_osirim_logs.txt"
sbatch ${fpath_sbatch}

exit 0
