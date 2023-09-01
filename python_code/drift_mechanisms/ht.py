class DriftHT:

    def __init__(self):
        self.threshold = 1e-5
        self.alarm = [[]]

    def reset_alarm(self, index):
        initial_len = len(self.alarm)
        if index > initial_len - 1:
            self.alarm.append([])
        self.alarm[index] = []

    def set_hyper(self, args):
        if args == 'Default':
            return
        self.threshold = args['ht_threshold']

    def analyze_samples(self, index, user_ht_value):
        flag_alarm = 0.0

        # receives the ht value already
        if user_ht_value > self.threshold:
            flag_alarm = 1.0

        self.alarm[index].append(flag_alarm)

    def alarm_dict_key(self):
        return 'HT threshold: ' + str(self.threshold)
