python train_channel.py --batch_size 256 --max_epoch 400 --NRepeat 8 \
                        --temperature 0.07 --CMItemperature 0.07 --lr 0.03 --CMIFactor 1 \
                        --dataset_root ~/Documents/datasets/imagenette/imagenette2 \
                        --dataset ImageNette --channel_model resnet18 