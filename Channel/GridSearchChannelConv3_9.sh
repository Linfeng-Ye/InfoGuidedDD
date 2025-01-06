#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100_4g.20gb:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=00-56:00
#SBATCH --output=out/%N-%j.txt
#SBATCH --job-name=DDPM1
#SBATCH --account=def-plato

module load python/3.10
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install -q --no-index --upgrade pip
pip install -q --no-index -r requirements.txt

python train_channel.py --batch_size 256 --max_epoch 400 --NRepeat 8 \
                        --temperature 0.01 --CMItemperature 0.01 --lr 0.003 --CMIFactor 0.01 \
                        --dataset CIFAR100 --channel_model conv3

python train_channel.py --batch_size 256 --max_epoch 400 --NRepeat 8 \
                        --temperature 0.07 --CMItemperature 0.01 --lr 0.003 --CMIFactor 0.01 \
                        --dataset CIFAR100 --channel_model conv3

python train_channel.py --batch_size 256 --max_epoch 400 --NRepeat 8 \
                        --temperature 0.3 --CMItemperature 0.01 --lr 0.003 --CMIFactor 0.01 \
                        --dataset CIFAR100 --channel_model conv3                        

python train_channel.py --batch_size 256 --max_epoch 400 --NRepeat 8 \
                        --temperature 0.01 --CMItemperature 0.07 --lr 0.003 --CMIFactor 0.01 \
                        --dataset CIFAR100 --channel_model conv3

python train_channel.py --batch_size 256 --max_epoch 400 --NRepeat 8 \
                        --temperature 0.07 --CMItemperature 0.07 --lr 0.003 --CMIFactor 0.01 \
                        --dataset CIFAR100 --channel_model conv3

python train_channel.py --batch_size 256 --max_epoch 400 --NRepeat 8 \
                        --temperature 0.3 --CMItemperature 0.07 --lr 0.003 --CMIFactor 0.01 \
                        --dataset CIFAR100 --channel_model conv3                        

python train_channel.py --batch_size 256 --max_epoch 400 --NRepeat 8 \
                        --temperature 0.01 --CMItemperature 0.3 --lr 0.003 --CMIFactor 0.01 \
                        --dataset CIFAR100 --channel_model conv3

python train_channel.py --batch_size 256 --max_epoch 400 --NRepeat 8 \
                        --temperature 0.07 --CMItemperature 0.3 --lr 0.003 --CMIFactor 0.01 \
                        --dataset CIFAR100 --channel_model conv3

python train_channel.py --batch_size 256 --max_epoch 400 --NRepeat 8 \
                        --temperature 0.3 --CMItemperature 0.3 --lr 0.003 --CMIFactor 0.01 \
                        --dataset CIFAR100 --channel_model conv3     

