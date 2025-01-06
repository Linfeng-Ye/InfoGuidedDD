
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from torchvision import transforms

def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

class InfoGuidedGaussianDiffusionSampler(nn.Module):
    def __init__(self, DDPMmodel, model, classifier, beta_1, beta_T, T, Scmi=0.5, Sce=0.5, KernelSize=32, UnfoldStride=1):
        """
        Sce: Cross entropy gradient scaling factor
        Scmi: Conditional mutual information gradient scaling factor
        model: Channel model/ backbone model
        classifier: classification head
        """
        super().__init__()
        self.DDPMmodel = DDPMmodel
        self.classifier = classifier
        self.model = model
        self.T = T
        self.Sce = Sce
        self.Scmi = Scmi
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))
        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

        self.unfold = torch.nn.Unfold(kernel_size=KernelSize, dilation=1, 
                            padding=0, stride=UnfoldStride)
        self.kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
        
    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.coeff1, t, x_t.shape) * x_t -
            extract(self.coeff2, t, x_t.shape) * eps
        )

    def p_mean_variance(self, x_t, t):
        # below: only log_variance is used in the KL computations
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)

        eps = self.DDPMmodel(x_t, t)
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)

        return xt_prev_mean, var
    def CMI(self, Probfeatures):
        rtval = 0.
        Centrods = torch.mean(Probfeatures, 1).detach()
        for idx, Probfeature in enumerate(Probfeatures):
            rtval += self.kl_loss(Centrods[idx].log(), Probfeature.log())
        return rtval
    
    def forward(self, x_T):
        ChannelNorm = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        invDDPMNorm = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                        std = [ 1/0.5, 1/0.5, 1/0.5 ]),
                                                transforms.Normalize(mean = [ -0.5, -0.5, -0.5],
                                                                    std = [ 1., 1., 1. ]),
                                            ])
        
        x_t = x_T
        cross_entropy = nn.CrossEntropyLoss()

        for time_step in reversed(range(self.T)):
            with torch.no_grad():
                print(time_step)
                t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
                mean, var= self.p_mean_variance(x_t=x_t, t=t)
                # no noise when t == 0
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            input = x_t.detach()
            unfoldInput = self.unfold(ChannelNorm(invDDPMNorm(input)))
            unfoldInput = unfoldInput.transpose(1,2)
            N, NP, _ = unfoldInput.shape
            unfoldInput = unfoldInput.reshape(-1,3,self.KernelSize, self.KernelSize)

            feature = self.model(unfoldInput)
            feature = F.normalize(feature, dim=1)
            logit = self.classifier(feature)
            Probfeature = torch.softmax(feature/0.1, -1)
            ProbPredict = logit/0.1
            Probfeature = Probfeature.reshape(10, IPC*NP, Probfeature.shape[-1])
            ProbPredict = ProbPredict.reshape(10, IPC* NP, ProbPredict.shape[-1])
            loss = - self.Scmi.CMI(Probfeature)
            for idx, cent in enumerate(range(10)):
                centu = torch.eye(10)[idx].cuda()#*0.5+0.5*cent
                loss += self.Sce*cross_entropy(ProbPredict[idx], centu[None,:].expand(ProbPredict.shape[1], 10))
            loss.backward()
            Guided = input.grad
            x_t = mean + torch.sqrt(var) * noise + self.S*Guided
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
            self.classifier.zero_grad()
        x_0 = x_t
        return torch.clip(x_0, -1, 1)
    