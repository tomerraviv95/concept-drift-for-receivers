import os

import numpy as np
import random
import scipy.io

from dir_definitions import MIMO_COST2100_DIR
from python_code.channel.channels_hyperparams import N_ANT, N_USER
from python_code.utils.config_singleton import Config
from python_code.utils.constants import HALF

conf = Config()

MAX_FRAMES = 25


class Cost2100MIMOChannel:
    @staticmethod
    def calculate_channel(n_ant: int, n_user: int, frame_ind: int, fading: bool) -> np.ndarray:
        total_h = np.empty([n_user, n_ant])
        one_user_moving = False
        test_ht = True
        main_folder = 1 + (frame_ind // MAX_FRAMES)
        if one_user_moving:
            path_to_mat = os.path.join(MIMO_COST2100_DIR, f'h_one_user_moving.mat')
            h_user = scipy.io.loadmat(path_to_mat)['h_channel_response_mag'][:, :N_USER, frame_ind]
            h_user = np.transpose(h_user)
            #print(h_user)
            total_h = h_user
        elif test_ht:
            #h_user = np.tile([1., 0.6, 0.3, 0.2], (4, 1))
            h_user = [[0.8, 0.4, 0.4, 0.2], [0.4, 0.8, 0.4, 0.2],
                      [0.2, 0.4, 0.8, 0.4], [0.2, 0.4, 0.4, 0.8]]
            # random channel "flips"
            random.seed(frame_ind)
            user_id = random.randrange(4)
            change_prob = random.randrange(100)
            # change_dict = {0: [0.2, 0.4, 0.4, 0.8],
            #                1: [0.2, 0.4, 0.8, 0.4],
            #                2: [0.4, 0.8, 0.4, 0.2],
            #                3: [0.8, 0.4, 0.4, 0.2]}
            change_exist = {0: 0, 1: 0, 2: 0, 3: 0}
            if change_prob > 80:
                if change_exist[user_id] == 1:
                    change_exist[user_id] = 0
                else:
                    change_exist[user_id] = 1
            for j in range(4):
                if change_exist[j] == 1:
                    temp = h_user[j]
                    h_user[j] = h_user[(j + 1) % n_user]
                    h_user[(j + 1) % n_user] = temp
            # change only one user channel taps
            #if frame_ind >= 3:
                #h_user[0] = [0.2, 0.4, 0.4, 0.8]
                #h_user[1] = [0.2, 0.4, 0.8, 0.4]
                #h_user[2] = [0.4, 0.8, 0.4, 0.2]
            total_h = h_user
        else:
            for i in range(1, n_user + 1):
                path_to_mat = os.path.join(MIMO_COST2100_DIR, f'{main_folder}', f'h_{i}.mat')
                h_user = scipy.io.loadmat(path_to_mat)['norm_channel'][frame_ind % MAX_FRAMES, :N_USER]
                total_h[i - 1] = HALF * h_user
            total_h[np.arange(n_user), np.arange(n_user)] = 1
        return total_h

    @staticmethod
    def transmit(s: np.ndarray, h: np.ndarray, snr: float) -> np.ndarray:
        """
        The MIMO COST2100 Channel
        :param s: to transmit symbol words
        :param snr: signal-to-noise value
        :param h: channel coefficients
        :return: received word
        """
        conv = Cost2100MIMOChannel._compute_channel_signal_convolution(h, s)
        sigma = 10 ** (-0.1 * snr)
        w = np.sqrt(sigma) * np.random.randn(N_ANT, s.shape[1])
        y = conv + w
        return y

    @staticmethod
    def _compute_channel_signal_convolution(h: np.ndarray, s: np.ndarray) -> np.ndarray:
        conv = np.matmul(h, s)
        return conv
