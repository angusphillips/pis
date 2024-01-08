import torch as th
import torch.distributions as D

from .base_set import BaseSet, counter


class Brownian(BaseSet):
    def __init__(self, len_data, dim, is_linear=True, device='cpu'):
        super().__init__(len_data, device=device, is_linear=is_linear)
        self.data = th.ones(dim, dtype=float).to(device)  # pylint: disable= not-callable
        self.data_ndim = dim

        self.observed_locs = th.tensor(
            [
                0.21592641,
                0.118771404,
                -0.07945447,
                0.037677474,
                -0.27885845,
                -0.1484156,
                -0.3250906,
                -0.22957903,
                -0.44110894,
                -0.09830782,
                float('nan'),
                float('nan'),
                float('nan'),
                float('nan'),
                float('nan'),
                float('nan'),
                float('nan'),
                float('nan'),
                float('nan'),
                float('nan'),
                -0.8786016,
                -0.83736074,
                -0.7384849,
                -0.8939254,
                -0.7774566,
                -0.70238715,
                -0.87771565,
                -0.51853573,
                -0.6948214,
                -0.6202789,
            ]
        ).to(self.device)

    def cal_gt_big_z(self):  # pylint: disable=no-self-use
        return 1

    def get_gt_disc(self, x):
        return -self.evaluate_log_pdf(x)

    def viz_pdf(self, fsave="density.png", lim=3):  # pylint: disable=no-self-use
        pass

    @counter
    def evaluate_log_pdf(self, x):
        # x_og = x
        x = x.double()
        log_jacobian_term = -th.log(1 + th.exp(-x[:, 0])) - th.log(1 + th.exp(-x[:, 1]))

        s = th.tensor(1e-10, dtype=th.float64, device=self.device)
        # s = th.tensor(0.0, device=self.device)
        x = th.cat([(th.log(1 + th.exp(x[:, 0]))+s).unsqueeze(-1), (th.log(1 + th.exp(x[:, 1]))+s).unsqueeze(-1), x[:, 2:]], dim=1)
        inn_noise_prior = D.Normal(th.tensor([0.0], device=self.device), th.tensor([2.0], device=self.device)).log_prob(th.log(x[:, 0])) - th.log(x[:, 0])
        obs_noise_prior = D.Normal(th.tensor([0.0], device=self.device), th.tensor([2.0], device=self.device)).log_prob(th.log(x[:, 1])) - th.log(x[:, 1])  

        # scale = x[:, 0]
        # ok = D.Normal.arg_constraints["scale"].check(scale)
        # bad_elements = scale[~ok]
        # og_bad_elements = x_og[:, 0][~ok]
        # print(bad_elements)
        # print(og_bad_elements)

        hidden_loc_0_prior = D.Normal(th.tensor([0.0], device=self.device), scale=x[:, 0]).log_prob(x[:, 2])
        hidden_loc_priors = hidden_loc_0_prior
        for i in range(29):
            hidden_loc_priors += D.Normal(loc=x[:, i+2], scale=x[:, 0]).log_prob(x[:, i+3])

        log_prior = inn_noise_prior + obs_noise_prior + hidden_loc_priors

        inds_not_nan = th.argwhere(~th.isnan(self.observed_locs)).flatten()

        log_lik = th.zeros((x.shape[0]), device=self.device)
        for i in inds_not_nan:
            ll = D.Normal(loc=x[:, i+2].unsqueeze(-1), scale=x[:,1].unsqueeze(-1)).log_prob(self.observed_locs[i].expand(x.shape[0], 1))
            log_lik += ll[:, 0]

        log_posterior = log_prior + log_lik

        return log_posterior + log_jacobian_term

    def sample(self, batch_size):
        pass