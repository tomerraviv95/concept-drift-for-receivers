from typing import Tuple

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from numpy.random import default_rng

from python_code.channel.channels_hyperparams import N_ANT, N_USER
from python_code.channel.mimo_channels.cost_mimo_channel import Cost2100MIMOChannel
from python_code.channel.modulator import BPSKModulator
from python_code.utils.config_singleton import Config
from python_code.utils.constants import ChannelModels

R = 9 / 6.45

mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24
mpl.rcParams['font.size'] = 18
mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['figure.figsize'] = [9.5 * R, 6.45 * R]
mpl.rcParams['axes.titlesize'] = 28
mpl.rcParams['axes.labelsize'] = 28
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.markersize'] = 8
mpl.rcParams['legend.fontsize'] = 18* R
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
conf = Config()

MIMO_CHANNELS_DICT = {ChannelModels.Cost2100.name: Cost2100MIMOChannel}


class MIMOChannel:
    def __init__(self, block_length: int, pilots_length: int):
        self._block_length = block_length
        self._pilots_length = pilots_length
        self._bits_generator = default_rng(seed=conf.seed)
        self.tx_length = N_USER
        self.h_shape = [N_ANT, N_USER]
        self.rx_length = N_ANT

    def _transmit(self, h: np.ndarray, snr: float) -> Tuple[np.ndarray, np.ndarray]:
        tx_pilots = self._bits_generator.integers(0, BPSKModulator.constellation_size,
                                                  size=(self._pilots_length, N_USER))
        tx_data = self._bits_generator.integers(0, BPSKModulator.constellation_size,
                                                size=(self._block_length - self._pilots_length, N_USER))
        tx = np.concatenate([tx_pilots, tx_data])
        # modulation
        s = BPSKModulator.modulate(tx.T)
        # pass through channel
        rx = MIMO_CHANNELS_DICT[conf.channel_model].transmit(s=s, h=h, snr=snr)
        return tx, rx.T

    def get_vectors(self, snr: float, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # get channel values
        if conf.channel_model == ChannelModels.Cost2100.name:
            h = Cost2100MIMOChannel.calculate_channel(N_ANT, N_USER, index, conf.fading_in_channel)
        else:
            raise ValueError("No such channel model!!!")
        tx, rx = self._transmit(h, snr)
        return tx, h, rx


if __name__ == "__main__":
    channel_dataset = MIMOChannel(block_length=conf.block_length, pilots_length=conf.pilot_size)
    total_h_mag = []
    for t in range(conf.blocks_num):
        tx, h, rx = channel_dataset.get_vectors(conf.snr, t)
        total_h_mag.append(h)
    total_h_mag = np.array(total_h_mag)
    fig, axs = plt.subplots(N_ANT, sharex=True)
    for j in range(N_ANT):
        current_axis = axs[j]
        for i in range(N_USER):
            current_axis.plot(total_h_mag[:, j, i], label=f'user {i + 1}', linewidth=3.2)

        current_axis.grid(True, which='both')
        current_axis.set_ylim([0, 1])
        current_axis.set_ylabel(f'Ant. {j + 1}')
    fig.supxlabel(r'Block Index',fontsize=28* R)
    fig.supylabel(r'Magnitude',fontsize=28* R)
    axs[3].legend(loc='lower right')
    plt.show()
