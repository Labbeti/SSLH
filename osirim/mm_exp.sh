#!/bin/sh

run="mixmatch_exp"

ds_name=`./get_ds.sh $@`
logdir=`./get_board_path.sh $ds_name`
dataset_path=`./get_ds_path.sh $ds_name`

path_script="$HOME/root/SSL/standalone/$run.py"
path_torch="/logiciels/containerCollections/CUDA10/pytorch.sif"
path_py="$HOME/miniconda3/envs/env_ssl/bin/python"

name="$run"
out_file="$HOME/logs/${ds_name}_${run}_%j.out"
err_file="$HOME/logs/${ds_name}_${run}_%j.err"
tmp_file=".sbatch_$run.sh"


cat << EOT > $tmp_file
#!/bin/sh

#SBATCH --job-name=$name
#SBATCH --output=$out_file
#SBATCH --error=$err_file
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
# For GPU nodes
#SBATCH --partition="GPUNodes"
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding

module purge
module load singularity/3.0.3

srun singularity exec $path_torch $path_py $path_script --logdir "$logdir" --dataset_path "$dataset_path" $@

EOT

sbatch $tmp_file
