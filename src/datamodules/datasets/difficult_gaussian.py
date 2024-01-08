import torch as th
import torch.distributions as D
from src.utils.jamtorch_utils import as_numpy
import matplotlib.pyplot as plt

from .base_set import BaseSet, counter

class DifficultGaussian(BaseSet):
    def __init__(self, mean, scale, len_data, dim, is_linear=True, device='cpu'):
        super().__init__(len_data, device=device, is_linear=is_linear)
        self.data = th.ones(dim, dtype=float).cuda()  # pylint: disable= not-callable
        self.data_ndim = dim

        self.target = D.Normal(th.tensor(mean).cuda(), th.tensor(scale).cuda())

    def cal_gt_big_z(self):  # pylint: disable=no-self-use
        return 1

    def get_gt_disc(self, x):
        return -self.gaussian_log_pdf(x)

    def viz_pdf(self, fsave="difficult_gaussian_density.png"):  # pylint: disable=no-self-use
        x = th.linspace(-2, 8, 100).cuda()
        density = self.unnorm_pdf(x)
        x, pdf = as_numpy([x, density])
        fig, axs = plt.subplots(1, 1, figsize=(1 * 7, 1 * 7))
        axs.plot(x, pdf)
        fig.savefig(fsave)
        plt.close(fig)
        pass

    @counter
    def gaussian_log_pdf(self, x):
        return self.target.log_prob(x[:, 0])  # (B, )

    def sample(self, batch_size):
        x = self.target.sample((batch_size,))  # (B,1)
        return x
