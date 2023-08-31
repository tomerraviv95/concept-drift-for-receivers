import torch
from torch import nn

from python_code import DEVICE
from python_code.channel.modulator import BPSKModulator

HIDDEN_SIZE = 60


class DNNDetector(nn.Module):
    """
    The DNNDetector Network Architecture
    """

    def __init__(self, n_user, n_ant):
        super(DNNDetector, self).__init__()
        self.n_user = n_user
        self.n_ant = n_ant
        self.base_rx_size = self.n_ant
        self.n_states = BPSKModulator.constellation_size ** n_ant
        self.initialize_dnn()

    def initialize_dnn(self):
        layers = [nn.Linear(self.base_rx_size, HIDDEN_SIZE),
                  nn.ReLU(),
                  nn.Linear(HIDDEN_SIZE, self.n_states),
                  nn.Softmax()]
        self.net = nn.Sequential(*layers).to(DEVICE)

    def forward(self, rx: torch.Tensor) -> torch.Tensor:
        soft_estimation = self.net(rx)
        return soft_estimation
