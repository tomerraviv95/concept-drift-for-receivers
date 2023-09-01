import python_code.drift_mechanisms.pht as drift_detection
from python_code.utils.config_singleton import Config

conf = Config()


class DriftMechanismWrapper:

    def __init__(self, type: str):
        self.drift_mechanism = DRIFT_MECHANISMS_DICT[type]

    def is_train(self, *args):
        return self.drift_mechanism.is_train(*args)


class AlwaysDriftMechanism:
    @staticmethod
    def is_train(*args):
        return True


class PeriodicMechanism:
    @staticmethod
    def is_train(*args):
        if args[0] % conf.period == 0 or 0:
            return True


class DriftDetectionDriven:
    @staticmethod
    def is_train(*args):
        if args[0] == 0:
            return True
        alarm_len = len(drift_detection.alarm)
        for ai in range(alarm_len):
            if drift_detection.alarm[ai][args[0] - 1] == 1:  # retrain only when alarm was signaled
                return True
        return False


DRIFT_MECHANISMS_DICT = {
    'always': AlwaysDriftMechanism,
    'periodic': PeriodicMechanism,
    'drift': DriftDetectionDriven
}
