import numpy as np
from skmultiflow.drift_detection import DDM, PageHinkley
from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector

alarm = [[]]
avg_training = 0
training_log = {'DDM': [0, 0], 'PHT': [0, 0], 'HT': [0, 0]}
avg_training_ddm = 0
avg_training_pht = 0
avg_training_ht = 0
MAX_SAMPLES = 3000


class PageHinkley(BaseDriftDetector):
    """ Page-Hinkley method for concept drift detection.

    Notes
    -----
    This change detection method works by computing the observed
    values and their mean up to the current moment. Page-Hinkley
    won't output warning zone warnings, only change detections.
    The method works by means of the Page-Hinkley test [1]_. In general
    lines it will detect a concept drift if the observed mean at
    some instant is greater then a threshold value lambda.

    References
    ----------
    .. [1] E. S. Page. 1954. Continuous Inspection Schemes.
       Biometrika 41, 1/2 (1954), 100–115.

    Parameters
    ----------
    min_instances: int (default=30)
        The minimum number of instances before detecting change.
    delta: float (default=0.005)
        The delta factor for the Page Hinkley test.
    threshold: int (default=50)
        The change detection threshold (lambda).
    alpha: float (default=1 - 0.0001)
        The forgetting factor, used to weight the observed value
        and the mean.
        """


    def __init__(self, min_instances=30, delta=0.005, threshold=50, alpha=1 - 0.0001):
        super().__init__()
        self.min_instances = min_instances
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha
        self.x_mean = None
        self.sample_count = None
        self.sum = None
        self.reset()

    def reset(self):
        """ reset

        Resets the change detector parameters.

        """
        super().reset()
        self.sample_count = 1
        self.x_mean = 0.0
        self.sum = 0.0

    def add_element(self, x):
        """ Add a new element to the statistics

        Parameters
        ----------
        x: numeric value
            The observed value, from which we want to detect the
            concept change.

        Notes
        -----
        After calling this method, to verify if change was detected, one
        should call the super method detected_change, which returns True
        if concept drift was detected and False otherwise.

        """
        self.x_mean = self.x_mean + (x - self.x_mean) / float(self.sample_count)
        self.sum = max(0., self.alpha * self.sum + (abs(x - self.x_mean) - self.delta))

        self.sample_count += 1

        self.estimation = self.x_mean
        self.in_concept_change = False
        self.in_warning_zone = False

        self.delay = 0

        if self.sample_count < self.min_instances:
            return None

        if self.sum > self.threshold:
            self.in_concept_change = True


def reset_alarm(index):
    global alarm
    initial_len = len(alarm)
    if index > initial_len - 1:
        alarm.append([])
    alarm[index] = []


def avergae_num_retraining(drift_detection_method):
    # calculate new average num of training
    global alarm, avg_training, training_log
    global avg_training_ddm, avg_training_pht, avg_training_ht
    avg_training = training_log[drift_detection_method][0]  # take curr avg training number for this method
    trials = training_log[drift_detection_method][1]
    alarm_sum = np.sum(alarm, axis=0)
    alarm_sum = [[0 if alarm_sum[i] == 0 else 1 for i, value in enumerate(alarm_sum)]]
    for ai in range(len(alarm_sum)):  # FIXME for MIMO
        if len(alarm_sum[ai]) > 0:
            avg_training = (avg_training * trials + alarm_sum[ai].count(1)) / (trials + 1)

    # update new statistics
    training_log[drift_detection_method][0] = avg_training
    training_log[drift_detection_method][1] = trials + 1
    print(drift_detection_method + " avg: " + str(training_log[drift_detection_method][0]) + "trial: " +
          str(training_log[drift_detection_method][1]))


class DriftDetector:

    def __init__(self, type: str):
        self.drift_detector = DRIFT_ALGORITHM_DICT[type]

    def initialize(self, index):
        return self.drift_detector.__init__(self, index)

    def analyze_samples(self, index, samples_vector):
        return self.drift_detector.analyze_samples(self, index, samples_vector)

    # Set hyper parameters for method if were set by config file
    def set_hyper(self, args):
        return self.drift_detector.set_hyper(self, args)

    # Return name of drift detection method
    def alarm_dict_key(self):
        return self.drift_detector.alarm_dict_key(self)


class DriftDDM:

    def __init__(self, index):
        self.ddm = DDM()
        reset_alarm(index)

    def set_hyper(self, args):
        if args == 'Default':
            return
        # self.ddm.warning_level = args['warning']
        self.ddm.out_control_level = args['out_control_level']
        self.ddm.min_instances = args['min_instances_ddm']

    def alarm_dict_key(self):
        return 'ddm alarm: ' + str(self.ddm.out_control_level)

    def analyze_samples(self, index, samples_vector):
        # DDM analyzez the error rate given by comparing pilot bits ONLY
        global alarm
        alarms = []
        for sample in samples_vector:
            self.ddm.add_element(sample.cpu())
            alarms.append(self.ddm.detected_change())
        should_retrain = any(alarms)
        if should_retrain:
            self.ddm.reset()
            alarm[index].append(1.0)
        else:
            alarm[index].append(0.0)


class DriftPHT:

    def __init__(self, index):
        self.pht = PageHinkley()
        reset_alarm(index)

    def analyze_samples(self, index, samples_vector):
        global alarm
        alarms = []
        for sample in samples_vector:
            self.pht.add_element(sample.cpu())
            alarms.append(self.pht.in_concept_change)
        # print(f"mean: {self.pht.x_mean},{self.pht.sum},Alarms: {sum(alarms)}")
        if sum(alarms) > 0.01 * len(samples_vector):
            self.pht.reset()
            alarm[index].append(1.0)
        else:  # PHT method has no warning
            alarm[index].append(0.0)

    def alarm_dict_key(self):
        return 'pht threshold: ' + str(self.pht.threshold)

    def set_hyper(self, args):
        self.pht.threshold = args['threshold']
        self.pht.delta = args['delta']
        self.pht.min_instances = args['min_instances_pht']


class DriftHT:

    def __init__(self, index):
        self.threshold = 1e-5
        reset_alarm(index)
        # global alarm
        # initial_len = len(alarm)
        # if index > initial_len - 1:
        #     alarm.append([])
        # alarm[index] = []

    def set_hyper(self, args):
        if args == 'Default':
            return
        self.threshold = args['threshold']

    def analyze_samples(self, index, user_ht_value):
        global alarm
        flag_alarm = 0.0

        # receives the ht value already
        if user_ht_value > self.threshold:
            flag_alarm = 1.0

        # if self.model:
        #     kl_vec = samples_vector[0]
        #
        # else:
        #     kl_vec = samples_vector[1]
        #     # if np.shape(kl_vec[0])[0] and abs(kl_vec[-1][index] - kl_vec[-2][index]) > self.threshold:
        #     #     flag_alarm = 1.0
        # if np.shape(kl_vec)[0] > 1 and abs(kl_vec[-1][index] - kl_vec[-2][index]) > self.threshold:
        #     flag_alarm = 1.0

        alarm[index].append(flag_alarm)

    def alarm_dict_key(self):
        return 'HT threshold: ' + str(self.threshold)


DRIFT_ALGORITHM_DICT = {
    'DDM': DriftDDM,
    'PHT': DriftPHT,
    'HT': DriftHT
}
