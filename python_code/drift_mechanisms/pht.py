import math


class DriftPHT:

    def __init__(self, beta=0.3, delta=1e-1,lambda_value=3e-2):
        self.beta = beta
        self.delta = delta
        self.lambda_value = lambda_value
        self.mu_t_prev = 0
        self.previous_distance = math.inf

    def check_drift(self, samples_vector):
        average = samples_vector.mean()
        mu_t = self.beta * average + (1-self.beta) * self.mu_t_prev
        norm = abs(samples_vector - mu_t)
        differences_vector = norm - self.delta
        differences_vector[differences_vector<0] = 0
        distance = differences_vector.mean().item()
        distance_diff = abs(distance - self.previous_distance)
        if distance_diff != math.inf and distance_diff > self.lambda_value:
            self.mu_t_prev = average
            self.previous_distance = math.inf
            return 1
        self.mu_t_prev=mu_t
        self.previous_distance=distance
        return 0