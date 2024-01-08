import torch as th
import torch.distributions as D

from .base_set import BaseSet, counter


class FunnelSet(BaseSet):
    def __init__(self, len_data, dim, is_linear=True, device='cpu'):
        super().__init__(len_data, device=device, is_linear=is_linear)
        self.data = th.ones(dim, dtype=float).to(device)  # pylint: disable= not-callable
        self.data_ndim = dim

    def cal_gt_big_z(self):  # pylint: disable=no-self-use
        return 1

    def get_gt_disc(self, x):
        return -self.evaluate_log_density(x)

    def viz_pdf(self, fsave="density.png", lim=3):  # pylint: disable=no-self-use
        pass

    @counter
    def evaluate_log_density(self, x):
        v = x[:, 0]
        log_density_v = D.Normal(0.0, 3.0).log_prob(v)
        variance_other = th.exp(v)
        other_dim = self.data_ndim - 1
        cov_other = th.eye(other_dim, device=self.device).expand(x.shape[0], -1, -1)
        cov_other = th.vmap(lambda x, y: x * y)(variance_other, cov_other)
        mean_other = th.zeros((x.shape[0], other_dim), device=self.device)
        log_density_other = D.MultivariateNormal(mean_other, cov_other).log_prob(x[:, 1:])

        return log_density_v + log_density_other

    def sample(self, batch_size):
        shape1 = (batch_size,) + (1,)
        shape2 = (batch_size,) + (self.dim - 1,)
        x1 = 3 * th.randn(*shape1)
        scale_rest = th.sqrt(th.exp(x1))
        scale_rest = scale_rest.repeat(1, self.dim - 1)
        x2_n = scale_rest * th.randn(*shape2)
        samples = th.cat([x1, x2_n], dim=-1)
        return samples


