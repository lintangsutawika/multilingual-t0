#!/bin/bash

# Request half an hour of runtime:
#SBATCH --time=6-23:59:00

# Ask for the CPU partition and 16 cores
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

# Request for RAM memory (16GB) (CPU RAM):
#SBATCH --mem-per-cpu=10G 

# Specify a job name:
#SBATCH -J exp-001-prepare_dataset

# Specify an output file
#SBATCH -o /users/zyong2/data/zyong2/mt0/logs/log-001/prepare_dataset.out
#SBATCH -e /users/zyong2/data/zyong2/mt0/logs/log-001/prepare_dataset.err


# load python
module load python/3.7.4

# set up current directory path
CURRENT_DIR="/users/zyong2/data/zyong2/mt0/data/external/mt0/mC4_download"

# set up virtual environment and install necessary pip packages
python3 -m venv "${CURRENT_DIR}/env_prepare_dataset"
source "${CURRENT_DIR}/env_prepare_dataset/bin/activate"
pip3 install --upgrade pip
pip3 install -r "${CURRENT_DIR}/requirements.txt"

# download data
DATA_DIR="${CURRENT_DIR}/data"
mkdir -p $DATA_DIR
python3 prepare_dataset.py --base_dir $DATA_DIR --num_process 16
