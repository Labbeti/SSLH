#!/bin/sh

fname_script="$1"
script_params=`./get_skip_params.sh 1 $@`

path=`./get_param.sh "path" "NOT_FOUND" $@`
if [ "$path" = "NOT_FOUND" ]; then
  path="osirim"
  script_params="$script_params path=$path"
fi

cpus=`./get_param.sh "cpus" "10" $@`
gpus=`./get_param.sh "gpus" "2" $@`
dataset=`./get_param.sh "dataset" "esc10" $@`
dataset=`echo $dataset | tr a-z A-Z`

dpath_project=`pwd`
dpath_project=`dirname $dpath_project`
dpath_conda="$HOME/.conda"
conda_env="env_sslh"

fpath_python="${dpath_conda}/envs/${conda_env}/bin/python"
dpath_standalone="${dpath_project}/standalone"
fpath_script="${dpath_standalone}/${fname_script}"

job_name="${fname_script}"
dpath_log="${dpath_project}/logs/${dataset}"
fpath_out="${dpath_log}/${dataset}_${job_name}_%j.out"
fpath_err="${dpath_log}/${dataset}_${job_name}_%j.err"
fpath_singularity="/logiciels/containerCollections/CUDA10/pytorch.sif"
srun="srun singularity exec ${fpath_singularity}"

# Build sbatch file ----------------------------------------------------------------------------------------------------
partition="GPUNodes"
# Memory format : number[K|M|G|T]. If 0, no memory limit, use all of node.
mem="0"
# Time format : days-hours:minutes:seconds. If 0, no time limit.
time="0"

module_load="module load singularity/3.0.3"

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