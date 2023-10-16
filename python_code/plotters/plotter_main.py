from python_code.plotters.plotter_config import get_config, PlotType
from python_code.plotters.plotter_methods import compute_ser_for_method, RunParams
from python_code.plotters.plotter_utils import plot_by_values

if __name__ == '__main__':
    run_over = True  # whether to run over previous results
    trial_num = 1  # number of trials per point estimate, used to reduce noise by averaging results  of multiple runs
    run_params_obj = RunParams(run_over=run_over,
                               trial_num=trial_num)

    label_name = PlotType.DeepSICFigure

    print(label_name.name)
    params_dicts, methods_list, values, xlabel, ylabel, plot_type, drift_methods_list = get_config(label_name.name)
    all_curves = []

    for method in methods_list:
        print(method)
        for params_dict in params_dicts:
            print(params_dict)
            if method is 'DriftDetectionDriven':
                for drift_detection_params in drift_methods_list:
                    # set the drift detection method
                    params_dicts_drift = params_dict
                    params_dicts_drift['drift_detection_method'] = drift_detection_params['drift_detection_method']
                    params_dicts_drift['drift_detection_method_hp'] = drift_detection_params[
                        'drift_detection_method_hp']
                    compute_ser_for_method(all_curves, method, params_dicts_drift, run_params_obj)
            else:
                compute_ser_for_method(all_curves, method, params_dict, run_params_obj)
    plot_by_values(all_curves, values, xlabel, ylabel, plot_type)
