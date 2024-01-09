import torch as th
from torch import nn
import math

from typing import Callable, Union, Iterable

def get_timestep_embedding_and_rescale(
    timesteps, embedding_dim, rescale=1000.0, device='cpu'
):
    """
    Rescales the input because we want higher frequencies in the embedding and t is only between [0,1].
    """
    assert embedding_dim % 2 == 0
    MAGIC = 10_000  # comes from Transformers paper

    half_dim = embedding_dim // 2
    emb = math.log(MAGIC) / (half_dim - 1)
    emb = th.exp(th.arange(half_dim, dtype=th.float32, device=device) * -emb)
    emb = rescale * timesteps * emb.unsqueeze(0) 
    emb = th.cat([th.sin(emb), th.cos(emb)], -1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = th.nn.functional.pad(emb, [0, 1, 0, 1]) #not tested
    return emb

class Sin(nn.Module):
  def __init__(self):
    super().__init__()
  def forward(self, x):
    return th.sin(x)

class MLP(nn.Module):
    def __init__(
        self,
        input_shape: int,
        hidden_shapes: list,
        output_shape: Union[Iterable[int], int],
        zero_init: bool
    ):
        super().__init__()
        self.input_shape = input_shape
        self.hidden_shapes = hidden_shapes
        self.output_shape = output_shape

        self.layers = nn.Sequential(
            nn.Linear(input_shape, hidden_shapes[0]),
            Sin(),
            *[nn.Sequential(nn.Linear(hidden_shapes[i], hidden_shapes[i+1]), Sin()) for i in range(len(hidden_shapes)-1)],
            nn.Linear(hidden_shapes[-1], output_shape)
        )
        if zero_init:
            self.layers[-1].weight.data.fill_(0.0)
            self.layers[-1].bias.data.fill_(0.0)

    def __call__(self, x):
        return self.layers(x)

class MLPEmbedConcat(nn.Module):
    def __init__(
        self,
        hidden_shapes: list,
        t_hidden_shapes: list,
        x_hidden_shapes: list,
        x_emb_dim: int,
        t_emb_dim: int,
        dim: int,
        device,
        t_dim: int = 128,
    ):
        super().__init__()

        self.device=device

        self.t_dim = t_dim

        self.t_encoder = MLP(
            input_shape = t_dim, hidden_shapes=t_hidden_shapes, output_shape=t_emb_dim, zero_init=False
        )
        self.x_encoder = MLP(
            input_shape = dim, hidden_shapes=x_hidden_shapes, output_shape=x_emb_dim, zero_init=False
        )

        self.nn = MLP(input_shape = t_emb_dim + x_emb_dim, hidden_shapes=hidden_shapes, output_shape=dim, zero_init=True)

    def __call__(self, cond, x):
        cond = cond.view(-1, 1).expand((x.shape[0], 1))
        lbd_emb = get_timestep_embedding_and_rescale(cond, self.t_dim, device=self.device)
        x_emb = self.x_encoder(x)
        lbd_emb = self.t_encoder(lbd_emb)
        return (
            self.nn(th.cat([x_emb, lbd_emb], axis=-1))
        )