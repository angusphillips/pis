import torch as th
from torch import nn
import numpy as np
from einops import rearrange

from typing import  Iterable
    
def check_shape(cur_shape):
    if isinstance(cur_shape, Iterable):
        return tuple(cur_shape)
    elif isinstance(cur_shape, int):
        return tuple(
            [
                cur_shape,
            ]
        )
    else:
        raise NotImplementedError(f"Type {type(cur_shape)} not support")

class FourierMLP(nn.Module):
    def __init__(self, in_shape, out_shape, num_layers=2, channels=128, zero_init=True):
        super().__init__()
        self.out_shape = check_shape(out_shape)
        self.in_shape = check_shape(in_shape)

        self.register_buffer(
            "timestep_coeff", th.linspace(start=0.1, end=100, steps=channels)[None]
        )
        self.timestep_phase = nn.Parameter(th.randn(channels)[None])
        # self.input_embed = nn.Linear(int(np.prod(in_shape)), channels)
        self.input_embed = nn.Sequential(
            nn.Linear(int(np.prod(in_shape)), channels),
            nn.GELU(),
            *[
                nn.Sequential(nn.Linear(channels, channels), nn.GELU())
                for _ in range(num_layers)
            ],
        )
        self.timestep_embed = nn.Sequential(
            nn.Linear(2 * channels, channels),
            nn.GELU(),
            *[
                nn.Sequential(nn.Linear(channels, channels), nn.GELU())
                for _ in range(num_layers)
            ],
        )
        self.layers = nn.Sequential(
            nn.GELU(),
            nn.Linear(2 * channels, channels),
            nn.GELU(),
            *[
                nn.Sequential(nn.Linear(channels, channels), nn.GELU())
                for _ in range(num_layers-1)
            ],
            nn.Linear(channels, int(np.prod(self.out_shape))),
        )
        if zero_init:
            self.layers[-1].weight.data.fill_(0.0)
            self.layers[-1].bias.data.fill_(0.0)

    def forward(self, cond, inputs):
        cond = cond.view(-1, 1).expand((inputs.shape[0], 1))
        sin_embed_cond = th.sin(
            (self.timestep_coeff * cond.float()) + self.timestep_phase
        )
        cos_embed_cond = th.cos(
            (self.timestep_coeff * cond.float()) + self.timestep_phase
        )
        embed_cond = self.timestep_embed(
            rearrange([sin_embed_cond, cos_embed_cond], "d b w -> b (d w)")
        )
        embed_ins = self.input_embed(inputs.view(inputs.shape[0], -1))
        out = self.layers(th.cat([embed_ins, embed_cond], axis=-1))
        return out.view(-1, *self.out_shape)