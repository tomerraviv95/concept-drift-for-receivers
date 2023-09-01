from skmultiflow.drift_detection import PageHinkley
from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector

alarm = [[]]
sums = []


def reset_alarm(index):
    global alarm
    initial_len = len(alarm)
    if index > initial_len - 1:
        alarm.append([])
    alarm[index] = []


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


class DriftPHT:

    def __init__(self, index):
        self.pht = PageHinkley()
        reset_alarm(index)

    def analyze_samples(self, index, samples_vector):
        global alarm, sums
        alarms = []

        for sample in samples_vector:
            self.pht.add_element(sample.cpu())
            alarms.append(self.pht.in_concept_change)
        sums.append(self.pht.sum)

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
