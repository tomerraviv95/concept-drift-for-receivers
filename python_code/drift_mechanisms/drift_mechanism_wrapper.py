from python_code.channel.channels_hyperparams import N_USER
from python_code.drift_mechanisms.ddm import DriftDDM
from python_code.drift_mechanisms.ht import DriftHT
from python_code.drift_mechanisms.pht import DriftPHT
from python_code.utils.config_singleton import Config

conf = Config()


class DriftMechanismWrapper:

    def __init__(self, mechanism_type: str):
        if 'SISO' in conf.channel_type:
            self.n_users = 1
        else:
            self.n_users = N_USER
        self.drift_mechanism_list = [DRIFT_MECHANISMS_DICT[mechanism_type]() for _ in range(self.n_users)]

    def is_train(self, kwargs):
        if kwargs['block_ind'] == -1:
            return True
        for user, drift_mechanism in enumerate(self.drift_mechanism_list):
            if drift_mechanism.is_train(user=user, **kwargs):
                return True
        return False


class AlwaysDriftMechanism:
    def is_train(self, **kwargs):
        return True


class PeriodicMechanism:
    def is_train(self, block_ind, **kwargs):
        if (block_ind + 1) % conf.period == 0:
            return True


class DriftDetectionDriven:
    def __init__(self):
        DATA_DRIVEN_DRIFT_DETECTORS_DICT = {
            'DDM': DriftDDM,
            'PHT': DriftPHT,
            'HT': DriftHT
        }
        self.drift_detector = DATA_DRIVEN_DRIFT_DETECTORS_DICT[conf.drift_detection_method]
        if conf.drift_detection_method == 'DDM':
            self.drift_detector = self.drift_detector(alpha=conf.drift_detection_method_hp['alpha_ddm'],
                                                      beta=conf.drift_detection_method_hp['beta_ddm'])
        elif conf.drift_detection_method == 'PHT':
            self.drift_detector = self.drift_detector(beta=conf.drift_detection_method_hp['beta_pht'],
                                                      delta=conf.drift_detection_method_hp['delta_pht'],
                                                      lambda_value=conf.drift_detection_method_hp['lambda_pht'])
        elif conf.drift_detection_method == 'HT':
            self.drift_detector = self.drift_detector(threshold=conf.drift_detection_method_hp['ht_threshold'])

    def is_train(self, user, **kwargs):
        if conf.drift_detection_method == 'DDM':
            return self.drift_detector.check_drift(kwargs['error_rate'][:, user])
        elif conf.drift_detection_method == 'PHT':
            return self.drift_detector.check_drift(kwargs['rx'][:, user])
        elif conf.drift_detection_method == 'HT':
            return self.drift_detector.check_drift(kwargs['ht'][user])
        else:
            raise ValueError('Drift detection method not recognized!!!')


DRIFT_MECHANISMS_DICT = {
    'always': AlwaysDriftMechanism,
    'periodic': PeriodicMechanism,
    'drift': DriftDetectionDriven
}
