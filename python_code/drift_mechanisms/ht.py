class DriftHT:

    def __init__(self, threshold):
        self.threshold = threshold

    def check_drift(self, user_ht_value):
        # receives the ht value already
        print(user_ht_value)
        if user_ht_value > self.threshold:
            return 1
        return 0
