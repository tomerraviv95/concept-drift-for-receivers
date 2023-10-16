from enum import Enum
from typing import Dict

from python_code.channel.channels_hyperparams import N_USER
from python_code.drift_mechanisms.ddm import DriftDDM
from python_code.drift_mechanisms.ht import DriftHT
from python_code.drift_mechanisms.pht import DriftPHT
from python_code.utils.config_singleton import Config
from python_code.utils.constants import ChannelModes

conf = Config()


class DriftMechanismWrapper:

    def __init__(self, mechanism_type: str):
        self.n_users = 1 if conf.channel_type == ChannelModes.SISO.name else N_USER
        self.drift_mechanism_list = [DRIFT_MECHANISMS_DICT[mechanism_type]() for _ in range(self.n_users)]

    def is_train(self, kwargs: Dict):
        if kwargs['block_ind'] == -1:
            return True
        for user, drift_mechanism in enumerate(self.drift_mechanism_list):
            if drift_mechanism.is_train(user=user, **kwargs):
                return True
        return False


class AlwaysDriftMechanism:
    def is_train(self, **kwargs: Dict):
        return True


class PeriodicMechanism:
    def is_train(self, block_ind: int, **kwargs: Dict):
        if (block_ind + 1) % conf.period == 0:
            return True


class DRIFT_TYPES(Enum):
    DDM = 'DDM'
    PHT = 'PHT'
    HT = 'HT'


class DriftDetectionDriven:
    def __init__(self):
        DATA_DRIVEN_DRIFT_DETECTORS_DICT = {
            DRIFT_TYPES.DDM.name: DriftDDM,
            DRIFT_TYPES.PHT.name: DriftPHT,
            DRIFT_TYPES.HT.name: DriftHT
        }
        self.drift_detector = DATA_DRIVEN_DRIFT_DETECTORS_DICT[conf.drift_detection_method]
        if conf.drift_detection_method == DRIFT_TYPES.DDM.name:
            self.drift_detector = self.drift_detector(alpha=conf.drift_detection_method_hp['alpha_ddm'],
                                                      beta=conf.drift_detection_method_hp['beta_ddm'])
        elif conf.drift_detection_method == DRIFT_TYPES.PHT.name:
            self.drift_detector = self.drift_detector(beta=conf.drift_detection_method_hp['beta_pht'],
                                                      delta=conf.drift_detection_method_hp['delta_pht'],
                                                      lambda_value=conf.drift_detection_method_hp['lambda_pht'])
        elif conf.drift_detection_method == DRIFT_TYPES.HT.name:
            self.drift_detector = self.drift_detector(threshold=conf.drift_detection_method_hp['ht_threshold'])

    def is_train(self, user: int, **kwargs: Dict):
        if conf.drift_detection_method == DRIFT_TYPES.DDM.name:
            return self.drift_detector.check_drift(kwargs['error_rate'][:, user])
        elif conf.drift_detection_method == DRIFT_TYPES.PHT.name:
            return self.drift_detector.check_drift(kwargs['rx'][:, user])
        elif conf.drift_detection_method == DRIFT_TYPES.HT.name:
            return self.drift_detector.check_drift(kwargs['ht'][user])
        else:
            raise ValueError('Drift detection method not recognized!!!')


DRIFT_MECHANISMS_DICT = {
    'always': AlwaysDriftMechanism,
    'periodic': PeriodicMechanism,
    'drift': DriftDetectionDriven
}
