import math
import random


def choose_by_probability(items, p):
    r = random.random()
    sorted_probabilities = sorted(enumerate(p), key=lambda x : x[1])

    index = 0
    for tup in sorted_probabilities:
        index = tup[0]
        r -= tup[1]
        if r < 0: break

    return items[index]


def boltzmann_explore(values, T):
    bmprobs = [math.exp(v/T)/sum([math.exp(vprime/T) for vprime in values]) for v in values]
    return choose_by_probability(range(len(bmprobs)), bmprobs)