import random
from typing import List

import numpy as np
import torch
from torch.nn import CrossEntropyLoss, MSELoss
from torch.optim import RMSprop, Adam, SGD

from python_code import DEVICE
from python_code.channel.channel_dataset import ChannelModelDataset
from python_code.channel.channels_hyperparams import N_USER
from python_code.drift_mechanisms.drift_mechanism_wrapper import DriftMechanismWrapper, TRAINING_TYPES
from python_code.utils.config_singleton import Config
from python_code.utils.constants import ChannelModes
from python_code.utils.metrics import calculate_ber, calculate_error_rate

conf = Config()

random.seed(conf.seed)
torch.manual_seed(conf.seed)
torch.cuda.manual_seed(conf.seed)
np.random.seed(conf.seed)

alarms_dict = {}


class Trainer(object):
    """
    Implements the meta-trainer class. Every trainer must inherent from this base class.
    It implements the evaluation method, initializes the dataloader and the detector.
    It also defines some functions that every inherited trainer must implement.
    """

    def __init__(self):
        # initialize matrices, datasets and detector
        self._initialize_dataloader()
        self._initialize_detector()
        self.softmax = torch.nn.Softmax(dim=1)  # Single symbol probability inference

    def get_name(self):
        return self.__name__()

    def _initialize_detector(self):
        """
        Every trainer must have some base detector model
        """
        self.detector = None

    # calculate train loss
    def calc_loss(self, est: torch.Tensor, tx: torch.Tensor) -> torch.Tensor:
        """
         Every trainer must have some loss calculation
        """
        pass

    # setup the optimization algorithm
    def deep_learning_setup(self):
        """
        Sets up the optimizer and loss criterion
        """
        if conf.optimizer_type == 'Adam':
            self.optimizer = Adam(filter(lambda p: p.requires_grad, self.detector.parameters()),
                                  lr=self.lr)
        elif conf.optimizer_type == 'RMSprop':
            self.optimizer = RMSprop(filter(lambda p: p.requires_grad, self.detector.parameters()),
                                     lr=self.lr)
        elif conf.optimizer_type == 'SGD':
            self.optimizer = SGD(filter(lambda p: p.requires_grad, self.detector.parameters()),
                                 lr=self.lr)
        else:
            raise NotImplementedError("No such optimizer implemented!!!")
        if conf.loss_type == 'CrossEntropy':
            self.criterion = CrossEntropyLoss().to(DEVICE)
        elif conf.loss_type == 'MSE':
            self.criterion = MSELoss().to(DEVICE)
        else:
            raise NotImplementedError("No such loss function implemented!!!")

    def _initialize_dataloader(self):
        """
        Sets up the data loader - a generator from which we draw batches, in iterations
        """
        self.channel_dataset = ChannelModelDataset(block_length=conf.block_length,
                                                   pilots_length=conf.pilot_size,
                                                   blocks_num=conf.blocks_num)
        self.dataloader = torch.utils.data.DataLoader(self.channel_dataset)

    def _online_training(self, tx: torch.Tensor, rx: torch.Tensor):
        """
        Every detector trainer must have some function to adapt it online
        """
        pass

    def forward(self, rx: torch.Tensor) -> torch.Tensor:
        """
        Every trainer must have some forward pass for its detector
        """
        pass

    def forward_pilot(self, rx_pilot: torch.Tensor, tx_pilot: torch.Tensor) -> torch.Tensor:
        """
        Every trainer must have some forward pass for its detector
        """
        return self.forward(rx_pilot)

    def init_priors(self):
        """
        DeepSIC employs this initialization
        """
        pass

    def evaluate(self) -> List[float]:
        """
        The online evaluation run. Main function for running the experiments of sequential transmission of pilots and
        data blocks for the paper.
        :return: list of ber per timestep
        """
        print(f'Evaluating concept drift of type: {conf.mechanism}')
        total_ber = []
        block_idn_train = [0 for _ in
                           range(conf.blocks_num)]  # to keep track of index of block where the model was retrained
        if conf.mechanism == TRAINING_TYPES.DRIFT.name:
            print(conf.drift_detection_method)
        # draw words for a given snr
        transmitted_words, received_words, hs = self.channel_dataset.__getitem__(snr_list=[conf.snr])
        # either None or in case of DeepSIC intializes the priors
        self.init_priors()
        # initialize concept drift mechanism_type
        drift_mechanism = DriftMechanismWrapper(conf.mechanism)
        kwargs = {'block_ind': -1}
        # detect sequentially
        for block_ind in range(conf.blocks_num):
            print('*' * 20)
            # get current word and channel
            tx, h, rx = transmitted_words[block_ind], hs[block_ind], received_words[block_ind]
            # split words into data and pilot part
            tx_pilot, tx_data = tx[:conf.pilot_size], tx[conf.pilot_size:]
            rx_pilot, rx_data = rx[:conf.pilot_size], rx[conf.pilot_size:]
            if (conf.is_online_training and drift_mechanism.is_train(kwargs)):
                print('re-training')
                if conf.channel_type in ChannelModes.MIMO.name and conf.mechanism == TRAINING_TYPES.DRIFT.name:
                    for idx in range(N_USER):
                        self.train_user[idx] = True
                # re-train the detector
                self._online_training(tx_pilot, rx_pilot)
                block_idn_train[block_ind] = 1
            # detect data part after training on the pilot part
            detected_word = self.forward(rx_data)
            # calculate accuracy
            ber = calculate_ber(detected_word, tx_data[:, :rx.shape[1]])
            print(f'current: {block_ind, ber}')
            detected_pilot = self.forward_pilot(rx_pilot, tx_pilot)
            error_rate = calculate_error_rate(detected_pilot, tx_pilot[:, :rx.shape[1]])
            kwargs = {'block_ind': block_ind, 'error_rate': error_rate, 'rx': rx, 'ht': self.ht}
            total_ber.append(ber)

        print(f'Final ser: {sum(total_ber) / len(total_ber)}, Total Re-trains: {sum(block_idn_train)}')
        return total_ber, block_idn_train

    def run_train_loop(self, est: torch.Tensor, tx: torch.Tensor) -> float:
        # calculate loss
        loss = self.calc_loss(est=est, tx=tx)
        current_loss = loss.item()
        # back propagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return current_loss
