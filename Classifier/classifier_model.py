import torch
import torch.nn as nn
from collections import OrderedDict
# class MLP(torch.nn.Module):
#     def __init__(self, config):
#         super(MLP, self).__init__()
#         self.linear1 = torch.nn.Linear(2048, 512)
#         self.BN1 = nn.BatchNorm1d(512)
#         self.linear2 = torch.nn.Linear(512, 256)
#         self.BN2 = nn.BatchNorm1d(256)
#         self.linear3 = torch.nn.Linear(256, 128)
#         self.BN3 = nn.BatchNorm1d(128)
#         self.linear4 = torch.nn.Linear(128, 10)
#         self.activation = torch.nn.ReLU()
        
#     def forward(self, x):
#         x = self.linear1(x)
#         x = self.activation(x)
#         x = self.BN1(x)
#         x = self.linear2(x)
#         x = self.activation(x)
#         x = self.BN2(x)
#         x = self.linear3(x)
#         x = self.activation(x)
#         x = self.BN3(x)
#         x = self.linear4(x)
#         return x

# class MLP(torch.nn.Module):
#     def __init__(self, config, num_classes):
#         super(MLP, self).__init__()
#         Nlayers = config.classifier_depth
#         dim_input = config.classifier_input_dim
#         dim_latent = config.classifier_latent_dim
#         Norm = config.classifier_Norm

#         layers = []
#         for layeridx in range(Nlayers):
#             if layeridx == 0:
#                 layer = ("fc0",torch.nn.Linear(dim_input, dim_latent))
#                 layers.append(layer)
#                 if Norm == "BatchNorm":
#                     layers.append(("bn0",nn.BatchNorm1d(dim_latent)))
#                 layers.append(("ac0",torch.nn.ReLU()))
#             elif layeridx ==Nlayers-1:
#                 layer = ("fc{}".format(layeridx),
#                          torch.nn.Linear(dim_latent, num_classes))
#                 layers.append(layer)
#             else:
#                 layer = torch.nn.Linear(dim_latent, dim_latent)
#                 layers.append(layer)
#                 if Norm == "BatchNorm":
#                     layers.append(("bn{}".format(layeridx),
#                                    nn.BatchNorm1d(dim_latent)))
#                 layers.append(("ac{}".format(layeridx),
#                                torch.nn.ReLU()))
#         breakpoint()
#         self.model = nn.Sequential(OrderedDict(layers))

#     def forward(self, x):
#         return self.model(x)


class MLP(torch.nn.Module):
    def __init__(self, config, num_classes):
        super(MLP, self).__init__()
        Nlayers = config.classifier_depth
        dim_input = config.classifier_input_dim
        dim_latent = config.classifier_latent_dim
        Norm = config.classifier_Norm

        layers = []
        for layeridx in range(Nlayers):
            if layeridx == 0:
                layer = torch.nn.Linear(dim_input, dim_latent)
                layers.append(layer)
                if Norm == "BatchNorm":
                    layers.append(nn.BatchNorm1d(dim_latent))
                layers.append(torch.nn.ReLU())
            elif layeridx ==Nlayers-1:
                layer = torch.nn.Linear(dim_latent, num_classes)
                layers.append(layer)
            else:
                layer = torch.nn.Linear(dim_latent, dim_latent)
                layers.append(layer)
                if Norm == "BatchNorm":
                    layers.append(nn.BatchNorm1d(dim_latent))
                layers.append(torch.nn.ReLU())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)