#!/bin/bash

# #######################################################################
# Slurm script to run jobs on Bowie Supercomputer Cluster               #
# Info on Bowie: https://open-atmos-krk.github.io/projects/hpc-diy.html #
# Expected arguments:                                                   # 
#   1: Python version (passed to pyenv)                                 #
#   2: command to execute (passed to mpiexec)                           #
#   3: file path to store exit code                                     #
# #######################################################################

#SBATCH --gres=gpu:1

source $HOME/.setup-pyenv
pyenv shell $1
python --version
python -m venv ./.venv
source ./.venv/bin/activate

pip install uv
uv pip install \
  --find-links "/mnt/cluster-workspace/shared/python/wheels" \
  --index-strategy first-index \
  -e '.[unit-tests]' 2>&1

# COMMAND to execute
bash -c "$2"

exit_code=$?
echo "Tests completed with exit code $exit_code"
echo $exit_code > "$3"
