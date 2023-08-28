from typing import List

import numpy as np
import torch
from torch import nn

from python_code import DEVICE
from python_code.channel.channels_hyperparams import N_ANT, N_USER
from python_code.channel.modulator import BPSKModulator
from python_code.detectors.deepsic.deep_sic_detector import DeepSICDetector
from python_code.detectors.trainer import Trainer
from python_code.utils.config_singleton import Config
from python_code.utils.constants import HALF, NUM_BINS
import matplotlib.pyplot as plt

conf = Config()
ITERATIONS = 5
EPOCHS = 250
WINDOW = 1

dist_probs_vec = []
KL_P_X = [[] for i in range(N_USER)]
KL_Q_X = [[] for i in range(N_USER)]
KL_5_probs_vec = []
KL_model_probs_vec = []
KLm_P_X = [[[] for j in range(ITERATIONS)] for i in range(N_USER)]
KLm_P_X_s0 = [[[] for j in range(ITERATIONS)] for i in range(N_USER)]
KLm_P_X_s0_plot = [[] for i in range(N_USER)]
KLm_P_X_s1 = [[[] for j in range(ITERATIONS)] for i in range(N_USER)]
KLm_Q_X = [[[] for j in range(ITERATIONS)] for i in range(N_USER)]
KLm_Q_X_s0 = [[[] for j in range(ITERATIONS)] for i in range(N_USER)]
KLm_Q_X_s1 = [[[] for j in range(ITERATIONS)] for i in range(N_USER)]
HT_s0_t_0 = [[[] for j in range(ITERATIONS)] for i in range(N_USER)]
HT_s0_t_1 = [[[] for j in range(ITERATIONS)] for i in range(N_USER)]
HT_s0_t_1_multivariate = [[[] for j in range(ITERATIONS)] for i in range(N_USER)]
HT_s1_t_0 = [[[] for j in range(ITERATIONS)] for i in range(N_USER)]
HT_s1_t_1 = [[[] for j in range(ITERATIONS)] for i in range(N_USER)]
HT_t_2 = [[[] for j in range(ITERATIONS)] for i in range(N_USER)]
HT_t_2_multivariate = [[] for i in range(N_USER)]
HT_s0_plot = [[] for i in range(N_USER)]
HT_s0_vec_users = []
prob_vec_plot = []


def prob_to_BPSK_symbol(p: torch.Tensor) -> torch.Tensor:
    """
    prob_to_symbol(x:PyTorch/Numpy Tensor/Array)
    Converts Probabilities to BPSK Symbols by hard threshold: [0,0.5] -> '-1', [0.5,1] -> '+1'
    :param p: probabilities vector
    :return: symbols vector
    """
    return torch.sign(p - HALF)


class DeepSICTrainer(Trainer):
    """Form the trainer class.

    Keyword arguments:

    """

    def __init__(self):
        self.memory_length = 1
        self.n_user = N_USER
        self.n_ant = N_ANT
        self.train_user = [True] * N_USER
        self.lr = 1e-3
        self.ht = [0] * N_USER
        super().__init__()

    def __str__(self):
        return 'DeepSIC'

    def init_priors(self):
        self.probs_vec = HALF * torch.ones(conf.block_length - conf.pilot_size, N_ANT).to(DEVICE).float()
        self.pilots_probs_vec = HALF * torch.ones(conf.pilot_size, N_ANT).to(DEVICE).float()

    def _initialize_detector(self):
        self.detector = [[DeepSICDetector().to(DEVICE) for _ in range(ITERATIONS)] for _ in
                         range(self.n_user)]  # 2D list for Storing the DeepSIC Networks

    def calc_loss(self, est: torch.Tensor, tx: torch.IntTensor) -> torch.Tensor:
        """
        Cross Entropy loss - distribution over states versus the gt state label
        """
        return self.criterion(input=est, target=tx.long())

    @staticmethod
    def preprocess(rx: torch.Tensor) -> torch.Tensor:
        return rx.float()

    def train_model(self, single_model: nn.Module, tx: torch.Tensor, rx: torch.Tensor):
        """
        Trains a DeepSIC Network
        """
        self.optimizer = torch.optim.Adam(single_model.parameters(), lr=self.lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        single_model = single_model.to(DEVICE)
        loss = 0
        y_total = self.preprocess(rx)
        for _ in range(EPOCHS):
            soft_estimation = single_model(y_total)
            current_loss = self.run_train_loop(soft_estimation, tx)
            loss += current_loss

    def train_models(self, model: List[List[DeepSICDetector]], i: int, tx_all: List[torch.Tensor],
                     rx_all: List[torch.Tensor]):
        for user in range(self.n_user):
            if self.train_user[user]:
                self.train_model(model[user][i], tx_all[user], rx_all[user])

    def _online_training(self, tx: torch.Tensor, rx: torch.Tensor):
        """
        Main training function for DeepSIC trainer. Initializes the probabilities, then propagates them through the
        network, training sequentially each network and not by end-to-end manner (each one individually).
        """
        if not conf.fading_in_channel:
            self._initialize_detector()
        initial_probs = tx.clone()
        tx_all, rx_all = self.prepare_data_for_training(tx, rx, initial_probs)
        # Training the DeepSIC network for each user for iteration=1
        self.train_models(self.detector, 0, tx_all, rx_all)
        # Initializing the probabilities
        probs_vec = HALF * torch.ones(tx.shape).to(DEVICE)
        # Training the DeepSICNet for each user-symbol/iteration
        for i in range(1, ITERATIONS):
            # Generating soft symbols for training purposes
            probs_vec = self.calculate_posteriors(self.detector, i, probs_vec, rx)
            # Obtaining the DeepSIC networks for each user-symbol and the i-th iteration
            tx_all, rx_all = self.prepare_data_for_training(tx, rx, probs_vec)
            # Training the DeepSIC networks for the iteration>1
            self.train_models(self.detector, i, tx_all, rx_all)

    def forward(self, rx: torch.Tensor, probs_vec: torch.Tensor = None) -> torch.Tensor:
        # detect and decode

        for i in range(ITERATIONS):
            probs_vec = self.calculate_posteriors(self.detector, i + 1, probs_vec, rx)
        detected_word = BPSKModulator.demodulate(prob_to_BPSK_symbol(probs_vec.float()))

        return detected_word

    def forward_pilot(self, rx: torch.Tensor, tx: torch.Tensor, probs_vec: torch.Tensor = None) -> torch.Tensor:
        # detect and decode
        linear = 1
        multivariate = 0

        global HT_s0_t_0, HT_s0_t_1, HT_s0_plot, HT_s0_vec_users, HT_s0_t_1_multivariate
        global HT_s1_t_0, HT_s1_t_1
        global prob_vec_plot
        KL_model_sum = [[[] for j in range(ITERATIONS)] for jj in range(N_USER)]

        for i in range(ITERATIONS):
            probs_vec = self.calculate_posteriors(self.detector, i + 1, probs_vec, rx)
            temp_probs_vec = []
            for user in range(self.n_user):
                rx_s0_idx = [i for i, x in enumerate(tx[:, user]) if x == 0]
                rx_s1_idx = [i for i, x in enumerate(tx[:, user]) if x == 1]
                ## HT
                HT_s0_t_0[user][i] = probs_vec[rx_s0_idx, user].numpy()
                HT_s1_t_0[user][i] = probs_vec[rx_s1_idx, user].numpy()
                if np.shape(HT_s0_t_1[user][i])[0] != 0:  # initialize first comparison:
                    # Do for symbol 0
                    n0_t_0 = np.shape(HT_s0_t_0[user][i])[0]
                    sample_mean_t_0_s0 = np.sum(HT_s0_t_0[user][i]) / n0_t_0
                    cov_mat_t_0 = np.cov(HT_s0_t_0[user][i])
                    n0_t_1 = np.shape(HT_s0_t_1[user][i])[0]
                    sample_mean_t_1_s0 = np.sum(HT_s0_t_1[user][i]) / n0_t_1
                    cov_mat_t_1 = np.cov(HT_s0_t_1[user][i])
                    pooled_cov_mat_s0 = ((n0_t_0 - 1) * cov_mat_t_0 + (n0_t_1 - 1) * cov_mat_t_1) / \
                                        (n0_t_0 + n0_t_1 - 2)

                    # Do for symbol 1
                    n1_t_0 = np.shape(HT_s1_t_0[user][i])[0]
                    sample_mean_t_0_s1 = np.sum(HT_s1_t_0[user][i]) / n1_t_0
                    cov_mat_t_0 = np.cov(HT_s1_t_0[user][i])
                    n1_t_1 = np.shape(HT_s1_t_1[user][i])[0]
                    sample_mean_t_1_s1 = np.sum(HT_s1_t_1[user][i]) / n1_t_1
                    cov_mat_t_1 = np.cov(HT_s1_t_1[user][i])
                    pooled_cov_mat_s1 = ((n1_t_0 - 1) * cov_mat_t_0 + (n1_t_1 - 1) * cov_mat_t_1) / \
                                        (n1_t_0 + n1_t_1 - 2)

                    # If linear add weigthed by number of samples each symbol
                    HT_t_2_s0 = (n0_t_0 * n0_t_1) / (n0_t_0 + n0_t_1) * \
                            np.transpose(sample_mean_t_0_s0 - sample_mean_t_1_s0) * \
                            np.transpose(pooled_cov_mat_s0) * (sample_mean_t_0_s0 - sample_mean_t_1_s0)
                    if linear:
                        pilot_number = tx[:, user].shape[0]
                        HT_t_2_s1 = (n1_t_0 * n1_t_1) / (n1_t_0 + n1_t_1) * \
                                    np.transpose(sample_mean_t_0_s1 - sample_mean_t_1_s1) * \
                                    np.transpose(pooled_cov_mat_s1) * (sample_mean_t_0_s1 - sample_mean_t_1_s1)
                        HT_t_2[user][i] = n0_t_0/pilot_number * HT_t_2_s0 + n1_t_0/pilot_number * HT_t_2_s1
                    else:
                        HT_t_2[user][i] = (n0_t_0 * n0_t_1) / (n0_t_0 + n0_t_1) * \
                                      np.transpose(sample_mean_t_0_s0 - sample_mean_t_1_s0) * \
                                      np.transpose(pooled_cov_mat_s0) * (sample_mean_t_0_s0 - sample_mean_t_1_s0)
                    # save previous distribution
                HT_s0_t_1[user][i] = HT_s0_t_0[user][i].copy()
                HT_s1_t_1[user][i] = HT_s1_t_0[user][i].copy()

                if i == ITERATIONS - 1:
                    temp_probs_vec.append(probs_vec[rx_s0_idx, user].numpy())

        prob_vec_plot.append(temp_probs_vec)  # save last iteration probs vec for all users

        if multivariate:
            if np.shape(HT_s0_t_1_multivariate[user][i])[0] != 0:  # initialize first comparison:
                for user in range(self.n_user):
                    n0_t_0 = np.shape(HT_s0_t_0[user][i])[0]
                    sample_mean_t_0_s0 = np.sum(HT_s0_t_0[user], axis=1) / n0_t_0
                    cov_mat_t_0 = np.cov(HT_s0_t_0[user])
                    n0_t_1 = np.shape(HT_s0_t_1_multivariate[user])[0]
                    sample_mean_t_1_s0 = np.sum(HT_s0_t_1_multivariate[user], axis=1) / n0_t_1
                    cov_mat_t_1 = np.cov(HT_s0_t_1_multivariate[user])
                    pooled_cov_mat_s0 = ((n0_t_0 - 1) * cov_mat_t_0 + (n0_t_1 - 1) * cov_mat_t_1) / \
                                        (n0_t_0 + n0_t_1 - 2)

                    HT_t_2_s0 = (n0_t_0 * n0_t_1) / (n0_t_0 + n0_t_1) * \
                                np.matmul(np.transpose(sample_mean_t_0_s0 - sample_mean_t_1_s0), \
                                np.matmul(pooled_cov_mat_s0, (sample_mean_t_0_s0 - sample_mean_t_1_s0)))
                    HT_t_2_multivariate[user] = HT_t_2_s0
                HT_s0_plot.append([row for row in HT_t_2_multivariate])
            for i in range(ITERATIONS):
                for user in range(self.n_user):
                    HT_s0_t_1_multivariate[user][i] = HT_s0_t_0[user][i].copy()
        elif np.prod(np.shape(HT_t_2[self.n_user-1][ITERATIONS - 1])) != 0:
            HT_s0_plot.append([row[ITERATIONS - 1] for row in HT_t_2])
            #HT_s0_plot.append([row[4] for row in HT_t_2])
            self.ht = [row[ITERATIONS - 1] for row in HT_t_2]

#        HT_s0_vec_users.append([row[ITERATIONS - 1] for row in KL_model_sum])


        detected_word = BPSKModulator.demodulate(prob_to_BPSK_symbol(probs_vec.float()))

        return detected_word

    def prepare_data_for_training(self, tx: torch.Tensor, rx: torch.Tensor, probs_vec: torch.Tensor) -> [
        torch.Tensor, torch.Tensor]:
        """
        Generates the data for each user
        """
        tx_all = []
        rx_all = []
        for k in range(self.n_user):
            idx = [user_i for user_i in range(self.n_user) if user_i != k]
            current_y_train = torch.cat((rx, probs_vec[:, idx].reshape(rx.shape[0], -1)), dim=1)
            tx_all.append(tx[:, k])
            rx_all.append(current_y_train)
        return tx_all, rx_all

    def calculate_posteriors(self, model: List[List[nn.Module]], i: int, probs_vec: torch.Tensor,
                             rx: torch.Tensor) -> torch.Tensor:
        """
        Propagates the probabilities through the learnt networks.
        """
        next_probs_vec = torch.zeros(probs_vec.shape).to(DEVICE)
        for user in range(self.n_user):
            idx = [user_i for user_i in range(self.n_user) if user_i != user]
            input = torch.cat((rx, probs_vec[:, idx].reshape(rx.shape[0], -1)), dim=1)
            preprocessed_input = self.preprocess(input)
            with torch.no_grad():
                output = self.softmax(model[user][i - 1](preprocessed_input))
            next_probs_vec[:, user] = output[:, 1:].reshape(next_probs_vec[:, user].shape)
        return next_probs_vec

    def plot_kl(self):
        global HT_s0_plot, HT_s0_vec_users
        global prob_vec_plot
        fig1 = plt.figure(1)  # user0 prob vec
        # fig2 = plt.figure(2) #user1 prob vec
        # fig3 = plt.figure(3) #user2 prob vec
        fig4 = plt.figure(4)  # t2 test
        # bins = np.linspace(0, 1, NUM_BINS)
        # xi = list(range(len(bins)))
        # for idx in range(self.n_user):
        #     subplot1 = fig1.add_subplot(2, 2, idx + 1)
        #     subplot1.plot(np.array(KL_5_probs_vec[idx]))
        #     subplot2 = fig2.add_subplot(2, 2, idx + 1)
        #     subplot2.plot(np.array(KL_model_probs_vec_T[idx]))

        for block in range(6):
            subplot1 = fig1.add_subplot(3, 6, 1 + block)
            x = np.arange(prob_vec_plot[block][0].shape[0])
            subplot1.scatter(x, np.array(prob_vec_plot[block][0]), s=1)
            subplot1 = fig1.add_subplot(3, 6, 7 + block)
            x = np.arange(prob_vec_plot[block][1].shape[0])
            subplot1.scatter(x, np.array(prob_vec_plot[block][1]), s=1)
            subplot1 = fig1.add_subplot(3, 6, 13 + block)
            x = np.arange(prob_vec_plot[block][2].shape[0])
            subplot1.scatter(x, np.array(prob_vec_plot[block][2]), s=1)
            # subplot1 = fig1.add_subplot(4, 6, 19 + block)
            # subplot1.plot(np.array(KLm_P_X_s0_plot[block][3]))
            # subplot1 = fig1.add_subplot(2, 3, 1 + block)
            # subplot1.stem(HT_s0_vec_users[block][0])
            # subplot1.set_xticks(xi)
            # subplot1.set_xticklabels(bins)
        #
        #     subplot3 = fig3.add_subplot(2, 3, 1 + block)
        #     subplot3.stem(np.array(HT_s0_vec_users[block][2]))
        #     plt.xticks(xi, bins)

        HT_s0_plot_T = np.transpose(HT_s0_plot[4:])
        for idx in range(self.n_user):
            subplot4 = fig4.add_subplot(2, 2, idx + 1)
            subplot4.stem(np.array(HT_s0_plot_T[idx]))
