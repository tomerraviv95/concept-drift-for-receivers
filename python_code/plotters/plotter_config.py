from enum import Enum
from typing import Tuple, List, Dict

from python_code.utils.constants import ChannelModes, DetectorType, ChannelModels


class PlotType(Enum):
    ViterbiNetFigure = 'ViterbiNetFigure'
    RNNFigure = 'RNNFigure'
    DeepSICFigure = 'DeepSICFigure'
    DNNFigure = 'DNNFigure'


def get_config(label_name: str) -> Tuple[List[Dict], list, list, str, str, str]:
    if label_name == PlotType.ViterbiNetFigure.name:
        params_dicts = [
            {'snr': 12, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.SISO.name,
             'blocks_num': 100, 'channel_model': ChannelModels.Cost2100.name,
             'period': 12, 'block_length': 10000, 'pilot_size': 500, 'drift_detection_method': None,
             'drift_detection_method_hp': None
             }
        ]
        methods_list = [
            'Periodic',
            'DriftDetectionDriven',
            'Always'
        ]
        drift_detection_methods = [
            {'drift_detection_method': 'DDM',
             'drift_detection_method_hp': {'out_control_level': 4, 'min_instances_ddm': 1500},
             },
            {'drift_detection_method': 'PHT',
             'drift_detection_method_hp': {'min_instances_pht': 30000, 'threshold': 2, 'delta': 1},
             },
        ]
        values = list(range(100))
        xlabel, ylabel = 'block_index', 'BER'
        plot_type = 'plot_ber_aggregated'
    elif label_name == PlotType.RNNFigure.name:
        params_dicts = [
            {'snr': 12, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.SISO.name,
             'blocks_num': 100, 'channel_model': ChannelModels.Cost2100.name,
             'period': 10, 'block_length': 10000, 'pilot_size': 2000, 'drift_detection_method': None,
             'drift_detection_method_hp': None
             }
        ]
        methods_list = [
            'Periodic',
            'DriftDetectionDriven',
            'Always'
        ]
        drift_detection_methods = [
            {'drift_detection_method': 'DDM',
             'drift_detection_method_hp': {'out_control_level': 4, 'min_instances_ddm': 6000},
             },
            {'drift_detection_method': 'PHT',
             'drift_detection_method_hp': {'min_instances_pht': 30000, 'threshold': 2, 'delta': 1},
             },
        ]
        values = list(range(100))
        xlabel, ylabel = 'block_index', 'BER'
        plot_type = 'plot_ber_aggregated'
    elif label_name == PlotType.DeepSICFigure.name:
        params_dicts = [
            {'snr': 12, 'detector_type': DetectorType.model.name, 'channel_type': ChannelModes.MIMO.name,
             'blocks_num': 50, 'channel_model': ChannelModels.Cost2100.name,
             'period': 5, 'block_length': 10000, 'pilot_size': 500, 'drift_detection_method': None,
             'drift_detection_method_hp': None
             }
        ]
        methods_list = [
            'Periodic',
            'DriftDetectionDriven',
            'Always'
        ]
        drift_detection_methods = [
            {'drift_detection_method': 'DDM',
             'drift_detection_method_hp': {'out_control_level': 4, 'min_instances_ddm': 2000},
             },
            {'drift_detection_method': 'PHT',
             'drift_detection_method_hp': {'min_instances_pht': 40000, 'threshold': 2, 'delta': 1},
             },
            {'drift_detection_method': 'HT',
             'drift_detection_method_hp': {'ht_threshold': 2.5},
             },
        ]
        values = list(range(50))
        xlabel, ylabel = 'block_index', 'BER'
        plot_type = 'plot_ber_aggregated'
    elif label_name == PlotType.DNNFigure.name:
        params_dicts = [
            {'snr': 12, 'detector_type': DetectorType.black_box.name, 'channel_type': ChannelModes.MIMO.name,
             'blocks_num': 50, 'channel_model': ChannelModels.Cost2100.name,
             'period': 5, 'block_length': 10000, 'pilot_size': 1000, 'drift_detection_method': None,
             'drift_detection_method_hp': None
             }
        ]
        methods_list = [
            'Periodic',
            'DriftDetectionDriven',
            'Always'
        ]
        drift_detection_methods = [
            {'drift_detection_method': 'DDM',
             'drift_detection_method_hp': {'out_control_level': 4, 'min_instances_ddm': 3000},
             },
            {'drift_detection_method': 'PHT',
             'drift_detection_method_hp': {'min_instances_pht': 30000, 'threshold': 2, 'delta': 1},
             },
            {'drift_detection_method': 'HT',
             'drift_detection_method_hp': {'ht_threshold': 2.5},
             },
        ]
        values = list(range(50))
        xlabel, ylabel = 'block_index', 'BER'
        plot_type = 'plot_ber_aggregated'
    else:
        raise ValueError('No such plot mechanism_type!!!')

    return params_dicts, methods_list, values, xlabel, ylabel, plot_type, drift_detection_methods
