import torch
import torch.nn as nn
from basicsr.utils.registry import ARCH_REGISTRY

class ResMLP(nn.Module):
    def __init__(self,n_feats = 512):
        super(ResMLP, self).__init__()
        self.resmlp = nn.Sequential(
            nn.Linear(n_feats , n_feats ),
            nn.LeakyReLU(0.1, True),
        )
    def forward(self, x):
        res=self.resmlp(x)
        return res

@ARCH_REGISTRY.register()
class simple_denoise(nn.Module):
    def __init__(self,
                 n_feats = 64, 
                 n_denoise_res = 5,
                 timesteps=5):
        super(simple_denoise, self).__init__()
        self.max_period=timesteps*10
        n_featsx4=4*n_feats
        resmlp = [
            nn.Linear(n_featsx4*2+1, n_featsx4),
            nn.LeakyReLU(0.1, True),
        ]
        for _ in range(n_denoise_res):
            resmlp.append(ResMLP(n_featsx4))
        self.resmlp=nn.Sequential(*resmlp)

    def forward(self,x, c, t):
        b, n, _ = x.shape
        t=t.float()
        t =t/self.max_period
        t=t.view(b, n, 1)
        c = torch.cat([c,t,x],dim=-1)

        fea = self.resmlp(c)

        return fea

