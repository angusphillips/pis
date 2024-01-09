import torch as th
from einops import rearrange
from torch import nn

from src.networks.mlp import MLP, get_timestep_embedding_and_rescale


class TimeMLP(nn.Module):
    def __init__(
        self, hidden_shapes: list, t_dim: int = 128
    ):
        super().__init__()
        self.t_dim = t_dim
        self.nn = MLP(input_shape=t_dim, hidden_shapes=hidden_shapes, output_shape=1, zero_init=False)
        self.nn.layers[-1].weight.data.fill_(0.0)
        self.nn.layers[-1].bias.data.fill_(0.01)

    def __call__(self, t):
        t_emb = get_timestep_embedding_and_rescale(t.unsqueeze(-1), self.t_dim, device='cuda')
        return self.nn(t_emb)


class TimeConder(nn.Module):
    def __init__(self, channel, out_dim, num_layers):
        super().__init__()
        self.register_buffer(
            "timestep_coeff", th.linspace(start=0.1, end=100, steps=channel)[None]
        )
        self.timestep_phase = nn.Parameter(th.randn(channel)[None])
        self.layers = nn.Sequential(
            nn.Linear(2 * channel, channel),
            *[
                nn.Sequential(
                    nn.GELU(),
                    nn.Linear(channel, channel),
                )
                for _ in range(num_layers - 1)
            ],
            nn.GELU(),
            nn.Linear(channel, out_dim)
        )

        self.layers[-1].weight.data.fill_(0.0)
        self.layers[-1].bias.data.fill_(0.01)

    def forward(self, t):
        sin_cond = th.sin((self.timestep_coeff * t.float()) + self.timestep_phase)
        cos_cond = th.cos((self.timestep_coeff * t.float()) + self.timestep_phase)
        cond = rearrange([sin_cond, cos_cond], "d b w -> b (d w)")
        return self.layers(cond)


if __name__ == "__main__":
    from torchinfo import summary

    net = TimeConder(64, 1, 3)
    batch_size = 10
    summary(net, input_size=(1,))
