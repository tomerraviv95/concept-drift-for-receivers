

class PageHinkley2(BaseDriftDetector):
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

    def add_block(self, x):
        """ Add a new vector of elements to the statistics

        Parameters
        ----------
        x: blocks of values
            The observed value, from which we want to detect the
            concept change.
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


class DriftPHT2:

    def __init__(self, index):
        self.pht = PageHinkley2()
        reset_alarm(index)

    def analyze_samples(self, index, samples_vector):
        global alarm, sums
        alarms = []

        self.pht.add_element(samples_vector.mean().cpu())
        alarms.append(self.pht.in_concept_change)

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