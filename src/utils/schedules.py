#!/usr/bin/python
# -*- coding: utf-8 -*-
import math


class Schedule(object):
    def value(self, time_step):
        """
        Returns the current value of the parameter given the time step 't' of the optimization procedure.

        :param time_step: time step
        :return: value of the schedule
        """
        raise NotImplementedError()


class ConstantSchedule(Schedule):
    def __init__(self, value: float):
        """
        Value remains constant over time.

        :param value: Constant value of the schedule
        """
        self._v = value

    def value(self, t):
        return self._v


def linear_interpolation(left_value: float, right_value: float, alpha: float):
    """
    Takes value to the left and to the right of t according to the `endpoints`. Alpha is the fraction of distance
    from left endpoint to right endpoint that t has covered.

    :param left_value: left value of the desired point
    :param right_value: right value of the desired point
    :param alpha: distance covered by t
    :return: middle point
    """
    return left_value + alpha * (right_value - left_value)


class PiecewiseSchedule(Schedule):
    def __init__(self, endpoints: [(int, int)], interpolation=linear_interpolation, outside_value=None):
        """
        Piecewise schedule

        :param endpoints: list of pairs `(time, value)` meanining that schedule should output
            `value` when `t==time`. All the values for time must be sorted in
            an increasing order. When t is between two times, e.g. `(time_a, value_a)`
            and `(time_b, value_b)`, such that `time_a <= t < time_b` then value outputs
            `interpolation(value_a, value_b, alpha)` where alpha is a fraction of
            time passed between `time_a` and `time_b` for time `t`.
        :param interpolation: lambda float, float, float: float
            a function that takes value to the left and to the right of t according
            to the `endpoints`. Alpha is the fraction of distance from left endpoint to
            right endpoint that t has covered. See linear_interpolation for example.
        :param outside_value: float
            if the value is requested outside of all the intervals sepecified in
            `endpoints` this value is returned. If None then AssertionError is
            raised when outside value is requested.
        """
        idxes = [e[0] for e in endpoints]
        assert idxes == sorted(idxes)
        self._interpolation = interpolation
        self._outside_value = outside_value
        self._endpoints = endpoints

    def value(self, t):
        for (l_t, l), (r_t, r) in zip(self._endpoints[:-1], self._endpoints[1:]):
            if l_t <= t < r_t:
                alpha = float(t - l_t) / (r_t - l_t)
                return self._interpolation(l, r, alpha)
        # t does not belong to any of the pieces, so doom.
        assert self._outside_value is not None
        return self._outside_value


class LinearSchedule(Schedule):
    def __init__(self, schedule_timesteps: int, final_p: float, initial_p=1.0):
        """
        Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.

        :param schedule_timesteps: Number of timesteps for which to linearly anneal initial_p to final_p
        :param final_p: initial output value
        :param initial_p: final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


class ExponentialSchedule(Schedule):
    def __init__(self, initial_p: float, min_p: float, decay: int):
        """
        Method that calculate p depending of the training iteration number. It converges to
        min_p as bigger the iteration number is.

        :param initial_p: initial value
        :param min_p: minimal value
        :param decay: step where majority of the 'p' will be reduced
        """
        self.initial_epsilon = initial_p
        self.min_epsilon = min_p
        self.epsilon_decay = decay

    def value(self, time_step):
        return self.min_epsilon + (self.initial_epsilon - self.min_epsilon) * math.exp(
            -1. * time_step / self.epsilon_decay)
