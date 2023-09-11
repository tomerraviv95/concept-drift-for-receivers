from typing import Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng

from python_code.channel.channels_hyperparams import MEMORY_LENGTH
from python_code.channel.modulator import BPSKModulator
from python_code.channel.siso_channels.cost_siso_channel import Cost2100SISOChannel
from python_code.utils.config_singleton import Config
from python_code.utils.constants import ChannelModels
from python_code.utils.trellis_utils import break_transmitted_siso_word_to_symbols

mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24
mpl.rcParams['font.size'] = 18
mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['figure.figsize'] = [9.5, 6.45]
mpl.rcParams['axes.titlesize'] = 28
mpl.rcParams['axes.labelsize'] = 28
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.markersize'] = 8
mpl.rcParams['legend.fontsize'] = 18
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
conf = Config()

SISO_CHANNELS_DICT = {ChannelModels.Cost2100.name: Cost2100SISOChannel}


class SISOChannel:
    def __init__(self, block_length: int, pilots_length: int):
        self._block_length = block_length
        self._pilots_length = pilots_length
        self._bits_generator = default_rng(seed=conf.seed)
        self.tx_length = MEMORY_LENGTH
        self.h_shape = [1, MEMORY_LENGTH]
        self.rx_length = 1

    def _transmit(self, h: np.ndarray, snr: float) -> Tuple[np.ndarray, np.ndarray]:
        # create pilots and data
        b_pilots = self._bits_generator.integers(0, BPSKModulator.constellation_size,
                                                 size=(1, self._pilots_length)).reshape(1, -1)
        b_data = self._bits_generator.integers(0, BPSKModulator.constellation_size,
                                               size=(1, self._block_length - self._pilots_length))
        b = np.concatenate([b_pilots, b_data], axis=1).reshape(1, -1)
        # add zero bits
        padded_b = np.concatenate([np.zeros([b.shape[0], MEMORY_LENGTH - 1]), b, np.zeros([b.shape[0], MEMORY_LENGTH])],
                                  axis=1)
        # modulation
        s = BPSKModulator.modulate(padded_b)
        # transmit through noisy channel
        rx = SISO_CHANNELS_DICT[conf.channel_model].transmit(s=s, h=h, snr=snr, memory_length=MEMORY_LENGTH)
        symbols, rx = break_transmitted_siso_word_to_symbols(MEMORY_LENGTH, b), rx.T
        return symbols[:-MEMORY_LENGTH + 1], rx[:-MEMORY_LENGTH + 1]

    def get_vectors(self, snr: float, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # get channel values
        # transmit through noisy channel
        if conf.channel_model == ChannelModels.Cost2100.name:
            h = Cost2100SISOChannel.calculate_channel(MEMORY_LENGTH, fading=conf.fading_in_channel, index=index)
        else:
            raise ValueError("No such channel model!!!")
        tx, rx = self._transmit(h, snr)
        return tx, h, rx


if __name__ == "__main__":
    channel_dataset = SISOChannel(block_length=conf.block_length, pilots_length=conf.pilot_size)
    total_h_mag = []
    for t in range(conf.blocks_num):
        tx, h, rx = channel_dataset.get_vectors(conf.snr, t)
        total_h_mag.append(h)
    total_h_mag = np.concatenate(total_h_mag)
    plt.figure()
    for i in range(MEMORY_LENGTH):
        plt.plot(total_h_mag[:, i], label=f'user {i + 1}', linewidth=3.2)
    plt.ylabel(r'Magnitude')
    plt.xlabel(r'Block Index')
    plt.grid(True, which='both')
    plt.legend(loc='upper right')
    plt.ylim([0, 1])
    plt.show()
