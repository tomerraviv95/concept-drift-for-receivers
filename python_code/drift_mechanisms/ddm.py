import torch


class DriftDDM:

    def __init__(self, alpha: float, beta: float):
        self.alpha = alpha
        self.beta = beta
        self.mu_t_prev = None
        self.sigma_t_prev = None
        self.mu_t_prev = None
        self.sigma_t_prev = None

    def check_drift(self, samples_vector: torch.Tensor):
        mu_t = samples_vector.mean()
        sigma_t = torch.sqrt(mu_t * (1 - mu_t) / len(samples_vector))
        if self.mu_t_prev is None and self.sigma_t_prev is None:
            self.mu_t_prev = mu_t
            self.sigma_t_prev = sigma_t
        if mu_t + sigma_t > self.mu_t_prev + self.alpha * self.sigma_t_prev:
            self.mu_t_prev = mu_t
            self.sigma_t_prev = sigma_t
            return 1
        self.mu_t_prev = self.beta * mu_t + (1 - self.beta) * self.mu_t_prev
        self.sigma_t_prev = self.beta * sigma_t + (1 - self.beta) * self.sigma_t_prev
        return 0
