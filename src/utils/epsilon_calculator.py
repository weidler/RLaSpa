import math


class EpsilonCalculator(object):
    def __init__(self, initial_epsilon, min_epsilon, epsilon_decay):
        self.initial_epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay

    def calculate(self, iteration):
        """
        Method that calculate epsilon depending of the training iteration number. It converges to
        min_epsilon as bigger the iteration number is

        :param iteration: iteration number
        :return: epsilon for the iteration
        """
        return self.min_epsilon + (self.initial_epsilon - self.min_epsilon) * math.exp(
            -1. * iteration / self.epsilon_decay)
