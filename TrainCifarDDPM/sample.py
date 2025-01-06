from Diffusion.Train import train, eval


def main(model_config = None):
    modelConfig = {
        "state": "train", # or eval
        "epoch": 2000,
        "batch_size": 256,
        "T": 1000,
        "channel": 128,
        "channel_mult": [1, 2, 3, 4],
        "attn": [2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "img_size": 32,
        "grad_clip": 1.,
        "device": "cuda:0", ### MAKE SURE YOU HAVE A GPU !!!
        "training_load_weight": None,
        "save_weight_dir": "./CheckpointsCifar10/", # or /CheckpointsCifar100/
        "test_load_weight": "ckpt_1999_.pt",
        "sampled_dir": "./SampledImgsCifar10/",
        "sampledNoisyImgName": "NoisyNoGuidenceImgs.png",
        "sampledImgName": "SampledNoGuidenceImgs.png",
        "nrow": 8,
        "Dataset": "Cifar10" # or Cifar100
        }
        
    if model_config is not None:
        modelConfig = model_config
    if modelConfig["state"] == "train":
        train(modelConfig)
    else:
        eval(modelConfig)


if __name__ == '__main__':
    main()
