#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100_4g.20gb:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=00-72:00
#SBATCH --output=out/%N-%j.txt
#SBATCH --job-name=DDPM1
#SBATCH --account=def-plato

module load python/3.10
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install -q --no-index --upgrade pip
pip install -q --no-index -r requirements.txt

python sample.py
                