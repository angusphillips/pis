import math
import torch
import torch.distributions as D
import torch.linalg as linalg

from .base_set import BaseSet, counter

class Difficult2D(BaseSet):
    def __init__(self, len_data, dim, is_linear=True, device='cuda'): 
        super().__init__(len_data, device=device, is_linear=is_linear)
        self.device = device 

        self.data = torch.ones(dim, dtype=torch.float64, device=device)
        self.data_ndim = dim

        self.n_components = 6
        self.mean_a = torch.tensor([3.0, 0.0], device=device) 
        self.mean_b = torch.tensor([-2.5, 0.0], device=device)  
        self.mean_c = torch.tensor([2.0, 3.0], device=device)  
        self.means = torch.stack((self.mean_a, self.mean_b, self.mean_c), dim=0)
        self.cov_a = torch.tensor([[0.7, 0.0], [0.0, 0.05]], device=device) 
        self.cov_b = torch.tensor([[0.7, 0.0], [0.0, 0.05]], device=device) 
        self.cov_c = torch.tensor([[1.0, 0.95], [0.95, 1.0]], device=device) 
        self.covs = torch.stack((self.cov_a, self.cov_b, self.cov_c), dim=0)
        self.all_means = torch.tensor(
            [[3.0, 0.0], [0.0, 3.0], [-2.5, 0.0], [0.0, -2.5], [2.0, 3.0], [3.0, 2.0]],
            device=device,
        )
        self.all_covs = torch.tensor(
            [
                [[0.7, 0.0], [0.0, 0.05]],
                [[0.05, 0.0], [0.0, 0.7]],
                [[0.7, 0.0], [0.0, 0.05]],
                [[0.05, 0.0], [0.0, 0.7]],
                [[1.0, 0.95], [0.95, 1.0]],
                [[1.0, 0.95], [0.95, 1.0]],
            ],
            device=device,
        )

    def cal_gt_big_z(self):
        return 1

    def get_gt_disc(self, x):
        return -self.evaluate_log_density(x)

    def viz_pdf(self, fsave="density.png"):
        pass

    def raw_log_density(self, x: torch.Tensor) -> torch.Tensor:
        log_weights = torch.log(torch.tensor([1.0 / 3, 1.0 / 3.0, 1.0 / 3.0], device=self.device))  
        l = linalg.cholesky(self.covs)
        y = linalg.solve_triangular(l, (x.unsqueeze(0) - self.means).unsqueeze(-1), upper=False)[:,:,0]
        mahalanobis_term = -1 / 2 * torch.einsum("...i,...i->...", y, y)
        n = self.means.shape[-1]
        normalizing_term = -n / 2 * torch.log(2 * torch.tensor(math.pi, device=self.device)) - torch.log(
            l.diagonal(dim1=-2, dim2=-1)
        ).sum(dim=1)
        individual_log_pdfs = mahalanobis_term + normalizing_term
        mixture_weighted_pdfs = individual_log_pdfs + log_weights
        return torch.logsumexp(mixture_weighted_pdfs, dim=0)

    def make_2d_invariant(self, log_density, x: torch.Tensor) -> torch.Tensor:
        density_a = log_density(x)
        density_b = log_density(torch.flip(x, dims=[-1]))
        return torch.logaddexp(density_a, density_b) - torch.log(torch.tensor(2.0, device=self.device))

    @counter
    def evaluate_log_density(self, x: torch.Tensor) -> torch.Tensor:
        density_func = lambda x: self.make_2d_invariant(self.raw_log_density, x)
        return torch.vmap(density_func)(x)

    def sample(self, batch_size: int):
        batched_sample_shape = (batch_size, self.data_ndim)
        components = torch.randint(0, self.n_components, (batch_size,), device=self.device)
        means = self.all_means[components]
        covs = self.all_covs[components]
        samples = D.MultivariateNormal(means, covs).sample()
        assert samples.shape == batched_sample_shape
        return samples
