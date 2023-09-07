import torch


class DriftDDM:

    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        self.mu_t_prev = None
        self.sigma_t_prev = None
        self.reset()

    def reset(self):
        """
        Resets the change detector parameters.
        """
        self.mu_t_prev = 0.1
        self.sigma_t_prev = 0.0

    def check_drift(self, samples_vector):
        mu_t = samples_vector.mean()
        sigma_t = torch.sqrt(mu_t * (1 - mu_t) / float(len(samples_vector)))
        if mu_t + sigma_t > self.mu_t_prev + self.alpha * self.sigma_t_prev:
            self.reset()
            return 1
        self.mu_t_prev = self.beta * mu_t + (1 - self.beta) * self.mu_t_prev
        self.sigma_t_prev = self.beta * sigma_t + (1 - self.beta) * self.sigma_t_prev
        return 0
