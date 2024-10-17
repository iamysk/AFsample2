#!/bin/bash
#SBATCH --gpus 1
#SBATCH -t 4320
#SBATCH -A berzelius-2022-218

# Set default log file names, and overwrite them using the first argument ($1) passed to the script
LOG_NAME=${1:-"default_log"}  # If $1 is not provided, use 'default_log'

# Dynamically set log file names using the provided name
LOG_OUTPUT="${LOG_NAME}_slurm.out"
LOG_ERROR="${LOG_NAME}_slurm.err"

# Redirect stdout and stderr to log files
exec > "$LOG_OUTPUT" 2> "$LOG_ERROR"

AF_PATH='/proj/wallner-b/users/x_yogka/github_repositories/AFsample2/AF_multitemplate'

#module load Anaconda/2021.05-nsc1
module load Anaconda
#module load buildenv-gcccuda/11.2-8.3.1-bare
module load buildenv-gcccuda/11.4-8.3.1-bare
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/software/sse/manual/CUDA/11.4.2_470.57.02/lib64/

# env
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_PATH
export TF_FORCE_UNIFIED_MEMORY=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=10.0

echo $XLA_FLAGS
conda activate /proj/wallner-b/users/x_yogka/miniconda3/envs/af2_yk

which python
nvidia-smi
date
echo Running CMD: python $AF_PATH/run_alphafold.py $@
python $AF_PATH/run_alphafold.py "${@:2}"
date