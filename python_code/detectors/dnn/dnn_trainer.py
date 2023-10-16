import numpy as np
import torch

from python_code.channel.channels_hyperparams import N_USER, N_ANT
from python_code.detectors.dnn.dnn_detector import DNNDetector
from python_code.detectors.trainer import Trainer
from python_code.utils.config_singleton import Config
from python_code.utils.hotelling_test_utils import run_hotelling_test
from python_code.utils.trellis_utils import calculate_mimo_states, calculate_symbols_from_states

conf = Config()

EPOCHS = 400

HT_s0_t_0 = [[] for _ in range(N_USER)]
prev_ht_s0 = [[] for _ in range(N_USER)]
HT_s1_t_0 = [[] for _ in range(N_USER)]
prev_ht_s1 = [[] for _ in range(N_USER)]
HT_t_2 = [[] for _ in range(N_USER)]


class DNNTrainer(Trainer):
    """Form the trainer class.

    Keyword arguments:

    """

    def __init__(self):
        self.memory_length = 1
        self.n_user = N_USER
        self.n_ant = N_ANT
        self.train_user = [True] * N_USER
        self.lr = 5e-3
        self.ht = [0] * N_USER
        self.prev_ht_s1 = [[] for _ in range(self.n_user)]
        self.prev_ht_s0 = [[] for _ in range(self.n_user)]
        super().__init__()

    def __str__(self):
        return 'DNN Detector'

    def _initialize_detector(self):
        """
            Loads the DNN detector
        """
        self.detector = DNNDetector(self.n_user, self.n_ant)

    def calc_loss(self, est: torch.Tensor, tx: torch.IntTensor) -> torch.Tensor:
        """
        Cross Entropy loss - distribution over states versus the gt state label
        :param est: [1,transmission_length,n_states], each element is a probability
        :param tx: [1, transmission_length]
        :return: loss value
        """
        gt_states = calculate_mimo_states(self.n_ant, tx)
        loss = self.criterion(input=est, target=gt_states)
        return loss

    def forward(self, rx: torch.Tensor) -> torch.Tensor:
        soft_estimation = self.detector(rx.float())
        estimated_states = torch.argmax(soft_estimation, dim=1)
        detected_word = calculate_symbols_from_states(self.n_ant, estimated_states).long()
        return detected_word

    def _online_training(self, tx: torch.Tensor, rx: torch.Tensor):
        """
        Online training module - trains on the detected word.
        Start from the previous weights, or from scratch.
        :param tx: transmitted word
        :param rx: received word
        """
        if not conf.fading_in_channel:
            self._initialize_detector()
        self.optimizer = torch.optim.Adam(self.detector.parameters(), lr=self.lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        # run training loops
        loss = 0
        for i in range(EPOCHS):
            # pass through detector
            soft_estimation = self.detector(rx.float())
            current_loss = self.run_train_loop(est=soft_estimation, tx=tx)
            loss += current_loss

    def forward_pilot(self, rx: torch.Tensor, tx: torch.Tensor) -> torch.Tensor:
        # detect and decode
        soft_estimation = self.detector(rx.float())
        probs_per_symbol = self.calculate_prob_per_symbol(soft_estimation)
        # detect and decode
        ht_s0_t_0 = [[] for _ in range(self.n_user)]
        ht_s1_t_0 = [[] for _ in range(self.n_user)]
        ht_mat = [[] for _ in range(self.n_user)]
        for user in range(self.n_user):
            rx_s0_idx = [i for i, x in enumerate(tx[:, user]) if x == 0]
            rx_s1_idx = [i for i, x in enumerate(tx[:, user]) if x == 1]
            # HT
            ht_s0_t_0[user] = probs_per_symbol[rx_s0_idx, user].cpu().numpy()
            ht_s1_t_0[user] = probs_per_symbol[rx_s1_idx, user].cpu().numpy()
            if np.shape(self.prev_ht_s0[user])[0] != 0:
                run_hotelling_test(ht_mat, ht_s0_t_0, ht_s1_t_0, self.prev_ht_s0, self.prev_ht_s1, None, tx, user)
            # save previous distribution
            self.prev_ht_s0[user] = ht_s0_t_0[user].copy()
            self.prev_ht_s1[user] = ht_s1_t_0[user].copy()
        if np.prod(np.shape(ht_mat[self.n_user - 1])) != 0:
            self.ht = ht_mat
        estimated_states = torch.argmax(soft_estimation, dim=1)
        detected_word = calculate_symbols_from_states(self.n_ant, estimated_states).long()
        return detected_word

    def calculate_prob_per_symbol(self, soft_estimation):
        first_user_logits = torch.sum(soft_estimation[:, 1::2], keepdim=True, dim=1)
        second_user_logits = torch.sum(soft_estimation[:, 2:4], keepdim=True, dim=1) + \
                             torch.sum(soft_estimation[:, 6:8], keepdim=True, dim=1) + \
                             torch.sum(soft_estimation[:, 10:12], keepdim=True, dim=1) + \
                             torch.sum(soft_estimation[:, 14:], keepdim=True, dim=1)
        third_user_logits = torch.sum(soft_estimation[:, 4:8], keepdim=True, dim=1) + \
                            torch.sum(soft_estimation[:, 12:], keepdim=True, dim=1)
        fourth_user_logits = torch.sum(soft_estimation[:, 8:], keepdim=True, dim=1)
        probs_per_symbol = torch.concat([first_user_logits,
                                         second_user_logits,
                                         third_user_logits,
                                         fourth_user_logits], dim=1)
        probs_per_symbol = probs_per_symbol.detach()
        return probs_per_symbol
