import math
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction
from basicsr.utils.registry import ARCH_REGISTRY

class MLP(nn.Module):
    def __init__(self,
                 num_patches,
                 embed_dims,
                 patch_expansion,
                 channel_expansion,
                 **kwargs):

        super(MLP, self).__init__()

        patch_mix_dims = int(patch_expansion * num_patches)
        channel_mix_dims = int(channel_expansion * embed_dims)

        self.patch_mixer = nn.Sequential(
            nn.Linear(num_patches, patch_mix_dims),
            nn.GELU(),
            nn.Linear(patch_mix_dims, num_patches),
        )

        self.channel_mixer = nn.Sequential(
            nn.Linear(embed_dims, channel_mix_dims),
            nn.GELU(),
            nn.Linear(channel_mix_dims, embed_dims),
        )
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)

    def forward(self, x):
        x = x + self.patch_mixer(self.norm1(x).transpose(1,2)).transpose(1,2)
        x = x + self.channel_mixer(self.norm2(x))

        return x


@ARCH_REGISTRY.register()
class denoising(nn.Module):
    def __init__(
        self,
        in_channel=6,
        out_channel=3,
        inner_channel=32,
        block_num=3,
        layerscale_value=1e-3,
        group=4, 
        patch_expansion=0.5, 
        channel_expansion=4
    ):
        super().__init__()

        self.time_mlp = nn.Parameter(layerscale_value * torch.ones(1), requires_grad=True)

        self.first_layer = nn.Sequential(
                nn.Linear(in_channel*3, inner_channel),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),)

        self.blocks = nn.ModuleList()
        for i in range(block_num-2):
            block = nn.Sequential(
                MLP(num_patches=group*group, embed_dims=inner_channel, patch_expansion=patch_expansion, channel_expansion=channel_expansion),
                nn.GELU(),)
            self.blocks.append(block)

        self.final_layer = nn.Sequential(
                nn.Linear(inner_channel, out_channel),
                nn.GELU(),)

    def forward(self, x, c, time):
        if len(time.shape) == 2:
            time = time.unsqueeze(-1).repeat(1, 1, x.shape[-1])
        t = time * self.time_mlp
        x = torch.cat([x, t, c], dim=-1)
        x = self.first_layer(x)
        for block in self.blocks:
            x = block(x)
        return self.final_layer(x)
