#!/bin/bash
#SBATCH -A bii_dsc_community
#SBATCH -p gpu          # 1
#SBATCH --gres=gpu:1    # 1
#SBATCH -c 1
#SBATCH -t 00:01:00
#SBATCH -J sactest
#SBATCH -o sactest-%A.out
#SBATCH -e sactest-%A.err

module purge
module load cuda cudnn
module load anaconda
conda deactivate
conda activate ml_practice

python main.py