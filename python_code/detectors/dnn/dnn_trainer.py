import numpy as np
import torch

from python_code.channel.channels_hyperparams import N_USER, N_ANT
from python_code.detectors.dnn.dnn_detector import DNNDetector
from python_code.detectors.trainer import Trainer
from python_code.utils.config_singleton import Config
from python_code.utils.trellis_utils import calculate_mimo_states, calculate_symbols_from_states

conf = Config()

EPOCHS = 400

HT_s0_t_0 = [[] for _ in range(N_USER)]
HT_s0_t_1 = [[] for _ in range(N_USER)]
HT_s1_t_0 = [[] for _ in range(N_USER)]
HT_s1_t_1 = [[] for _ in range(N_USER)]
HT_t_2 = [[] for _ in range(N_USER)]


class DNNTrainer(Trainer):
    """Form the trainer class.

    Keyword arguments:

    """

    def __init__(self):
        self.memory_length = 1
        self.n_user = N_USER
        self.n_ant = N_ANT
        self.probs_vec = None
        self.pilots_probs_vec = None
        self.train_user = [True] * N_USER
        self.lr = 5e-3
        self.ht = [0] * N_USER
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

    def forward(self, rx: torch.Tensor, probs_vec: torch.Tensor = None) -> torch.Tensor:
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

    def forward_pilot(self, rx: torch.Tensor, tx: torch.Tensor, probs_vec: torch.Tensor = None) -> torch.Tensor:
        global HT_s0_t_0, HT_s0_t_1
        global HT_s1_t_0, HT_s1_t_1
        # detect and decode
        soft_estimation = self.detector(rx.float())
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
        for user in range(self.n_user):
            rx_s0_idx = [i for i, x in enumerate(tx[:, user]) if x == 0]
            rx_s1_idx = [i for i, x in enumerate(tx[:, user]) if x == 1]
            ## HT
            HT_s0_t_0[user] = probs_per_symbol[rx_s0_idx, user].detach().cpu().numpy()
            HT_s1_t_0[user] = probs_per_symbol[rx_s1_idx, user].detach().cpu().numpy()
            if np.shape(HT_s0_t_1[user])[0] != 0:
                # Symbol 0
                n0_t_0 = np.shape(HT_s0_t_0[user])[0]
                sample_mean_t_0_s0 = np.sum(HT_s0_t_0[user]) / n0_t_0
                cov_mat_t_0 = np.cov(HT_s0_t_0[user])
                n0_t_1 = np.shape(HT_s0_t_1[user])[0]
                sample_mean_t_1_s0 = np.sum(HT_s0_t_1[user]) / n0_t_1
                cov_mat_t_1 = np.cov(HT_s0_t_1[user])
                pooled_cov_mat_s0 = ((n0_t_0 - 1) * cov_mat_t_0 + (n0_t_1 - 1) * cov_mat_t_1) / \
                                    (n0_t_0 + n0_t_1 - 2)
                # Symbol 1
                n1_t_0 = np.shape(HT_s1_t_0[user])[0]
                sample_mean_t_0_s1 = np.sum(HT_s1_t_0[user]) / n1_t_0
                cov_mat_t_0 = np.cov(HT_s1_t_0[user])
                n1_t_1 = np.shape(HT_s1_t_1[user])[0]
                sample_mean_t_1_s1 = np.sum(HT_s1_t_1[user]) / n1_t_1
                cov_mat_t_1 = np.cov(HT_s1_t_1[user])
                pooled_cov_mat_s1 = ((n1_t_0 - 1) * cov_mat_t_0 + (n1_t_1 - 1) * cov_mat_t_1) / \
                                    (n1_t_0 + n1_t_1 - 2)
                # If linear add weigthed by number of samples each symbol
                HT_t_2_s0 = (n0_t_0 * n0_t_1) / (n0_t_0 + n0_t_1) * \
                            np.transpose(sample_mean_t_0_s0 - sample_mean_t_1_s0) * \
                            np.transpose(pooled_cov_mat_s0) * (sample_mean_t_0_s0 - sample_mean_t_1_s0)
                HT_t_2_s1 = (n1_t_0 * n1_t_1) / (n1_t_0 + n1_t_1) * \
                            np.transpose(sample_mean_t_0_s1 - sample_mean_t_1_s1) * \
                            np.transpose(pooled_cov_mat_s1) * (sample_mean_t_0_s1 - sample_mean_t_1_s1)
                # combine
                pilot_number = tx[:, user].shape[0]
                HT_t_2[user] = n0_t_0 / pilot_number * HT_t_2_s0 + n1_t_0 / pilot_number * HT_t_2_s1

            # save previous distribution
            HT_s0_t_1[user] = HT_s0_t_0[user].copy()
            HT_s1_t_1[user] = HT_s1_t_0[user].copy()

        if np.prod(np.shape(HT_t_2[self.n_user - 1])) != 0:
            self.ht = HT_t_2
        estimated_states = torch.argmax(soft_estimation, dim=1)
        detected_word = calculate_symbols_from_states(self.n_ant, estimated_states).long()
        return detected_word
