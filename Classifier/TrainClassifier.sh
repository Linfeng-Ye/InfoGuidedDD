python train_classifier.py --dataset CIFAR100 --channel_ckpt_root ChannelCkpt --load_pretrained_epoch 399 \
                           --feature_dim 128 --temperature 0.01 --CMItemperature 0.07 --lr 0.01 --CMIFactor 1.0 \
                           --classifier_input_dim 128 --classifier_latent_dim 256 --classifier_depth 3 \
                           --classifier_lr 0.1
