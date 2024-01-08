import pickle
import torch as th
import torch.distributions as D

from .base_set import BaseSet, counter


class LogReg(BaseSet):
    def __init__(self, data_path, len_data, dim, is_linear=True, device='cpu'):
        super().__init__(len_data, device=device, is_linear=is_linear)
        self.data = th.ones(dim, dtype=float).to(device) 
        
        with open(data_path, mode="rb") as f:
            u, y = pickle.load(f)
            u = th.tensor(u, device=device)
            y = th.tensor(y, device=device)

        # pre-processing
        y = (y + 1) // 2
        mean = th.mean(u, axis=0)
        std = th.std(u, unbiased=False, axis=0)
        std[std == 0.0] = 1.0
        u = (u - mean) / std
        # Add column for intercept term
        extra = th.ones((u.shape[0], 1), device=device)
        u = th.hstack([extra, u])
        dim = u.shape[1]
        self.data_ndim = dim

        self.y = y
        self.u = u

        mean = th.zeros(dim, device=self.device)
        cov = th.eye(dim, device=self.device)
        self.prior = D.MultivariateNormal(loc=mean, covariance_matrix=cov)

    def cal_gt_big_z(self):  # pylint: disable=no-self-use
        return 1

    def get_gt_disc(self, x):
        return -self.evaluate_log_pdf(x)

    def viz_pdf(self, fsave="density.png", lim=3):  # pylint: disable=no-self-use
        pass

    @counter
    def evaluate_log_pdf(self, x):
        batched_log_density = th.zeros(x.shape[0], device=self.device)
        for i in range(x.shape[0]):
            def log_bernoulli(u_, y_):
                log_sigmoids = -th.log(1 + th.exp(-th.dot(u_, x[i,:])))
                log_1_minus_sigmoids = -th.log(1 + th.exp(th.dot(u_, x[i,:])))
                return y_ * log_sigmoids + (1 - y_) * log_1_minus_sigmoids
            log_lik_terms = th.vmap(log_bernoulli)(self.u, self.y)
            log_posterior = th.sum(log_lik_terms) + self.prior.log_prob(x[i,:])
            batched_log_density[i] = log_posterior

        return batched_log_density

    def sample(self, batch_size):
        pass