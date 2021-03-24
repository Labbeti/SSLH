#!/bin/sh

fname_script="$1"
script_params=`./get_skip_params.sh 1 $@`

path=`./get_param.sh "path" "NOT_FOUND" $@`
if [ "$path" = "NOT_FOUND" ]; then
  path="olympe"
  script_params="$script_params path=$path"
fi

cpus=`./get_param.sh "cpus" "10" $@`
gpus=`./get_param.sh "gpus" "1" $@`
tag=`./get_param.sh "tag" "" $@`
dataset=`./get_param.sh "dataset" "esc10" $@`
dataset=`echo $dataset | tr a-z A-Z`

dpath_project=`pwd`
dpath_project=`dirname $dpath_project`
dpath_conda="/tmpdir/labbe/miniconda3"
conda_env="env_sslh"

fpath_python="${dpath_conda}/envs/${conda_env}/bin/python"
dpath_standalone="${dpath_project}/standalone"
fpath_script="${dpath_standalone}/${fname_script}"

job_name="`basename ${fname_script} .py`"
dpath_log="${dpath_project}/logs/${dataset}"
fpath_out="${dpath_log}/${dataset}_${job_name}_${tag}_%j.out"
fpath_err="${dpath_log}/${dataset}_${job_name}_${tag}_%j.err"
fpath_singularity=""
srun="srun"

# Build sbatch file ----------------------------------------------------------------------------------------------------
partition=""
# Memory format : number[K|M|G|T]. If 0, no memory limit, use all of node.
mem="64G"
# Time format : days-hours:minutes:seconds. If 0, no time limit.
time="3-00:00:00"

module_load="module load cuda/10.1.105"

fpath_sbatch=".tmp_${job_name}.sbatch"
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

# Duration limit (0 means no limit)
#SBATCH --time=${time}

# Mail for optional auto-sends
#SBATCH --mail-user=etienne.labbe@irit.fr


# For GPU nodes, select partition
#SBATCH --partition=${partition}

# For GPU nodes, select the number of GPUs
#SBATCH --gres=gpu:${gpus}

# For GPU nodes, force job to start only when CPU and GPU are all available
#SBATCH --gres-flags=enforce-binding


# For testing the sbatch file
## #SBATCH --test-only

# Others
## #SBATCH --ntasks-per-node=4
## #SBATCH --ntasks-per-core=1


module purge
${module_load}

${srun} ${fpath_python} ${fpath_script} ${script_params}

EOT

# Run & exit --------------------------------------------------------------------------------------------------------------
echo "Sbatch job '${job_name}' for script '${fname_script}'"
sbatch ${fpath_sbatch}

exit 0
