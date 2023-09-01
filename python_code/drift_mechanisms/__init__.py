from python_code.drift_mechanisms.ddm import DriftDDM
from python_code.drift_mechanisms.ht import DriftHT
from python_code.drift_mechanisms.pht import DriftPHT
from python_code.utils.config_singleton import Config

conf = Config()

DRIFT_ALGORITHM_DICT = {
    'DDM': DriftDDM(conf.out_control_level, conf.min_instances_ddm),
    'PHT': DriftPHT,
    'HT': DriftHT
}
