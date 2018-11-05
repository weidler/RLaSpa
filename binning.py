import numpy as np
from sklearn.utils.extmath import cartesian


def binThese(lowerBound, upperBound, bins=10, cutOffInf=10000):
    assert len(upperBound) == len(lowerBound)
    numFeatures = len(upperBound)
    featureBins = np.zeros((numFeatures, bins+1))
    for i in range(numFeatures):
        lb = -cutOffInf if lowerBound[i] <= -cutOffInf else lowerBound[i]
        ub = cutOffInf if upperBound[i] >= cutOffInf else upperBound[i]
        featureBins[i] = np.histogram_bin_edges([lb, ub], bins=bins)
    return featureBins[:, :-1]


stateBins = binThese([-5, 0, 0.08], [5, np.inf, 0.2], bins=10, cutOffInf=5)
allStates = cartesian(stateBins)

print(stateBins)
print(allStates[0, :])
print(allStates[1, :])
randomState = [-4.8, 0.7, 0.104]
print(f'randomState: {randomState}')
# temp = allStates-randomState
# temp = temp[temp[:, 0] >= 0]
# temp = temp[temp[:, 1] >= 0]
# temp = temp[temp[:, 2] >= 0]
# print(f'bin: {temp[0]}')
