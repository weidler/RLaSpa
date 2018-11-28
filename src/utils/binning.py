import numpy as np
from sklearn.utils.extmath import cartesian
import time


def binThese(lowerBound, upperBound, bins=10, cutOffInf=10000):
    assert len(upperBound) == len(lowerBound)
    numFeatures = len(upperBound)
    featureBins = np.zeros((numFeatures, bins+1))
    for i in range(numFeatures):
        lb = -cutOffInf if lowerBound[i] <= -cutOffInf else lowerBound[i]
        ub = cutOffInf if upperBound[i] >= cutOffInf else upperBound[i]
        featureBins[i] = np.histogram_bin_edges([lb, ub], bins=bins)
    return featureBins[:, :-1]

def getState(stateBins, features):
    assert len(stateBins) == len(features)
    bin = [np.where(stateBins[i] <= f)[0][-1] for i, f in enumerate(features)]
    return [stateBins[i, f] for i, f in enumerate(bin)]

def getStateNew(stateBins, features):
    assert len(stateBins) == len(features)
    ### THIS IS NEW
    # bin = [np.where(stateBins[i] <= f)[0][-1] for i, f in enumerate(features)]
    bin = [int((f-stateBins[i][0])/(stateBins[i][1]-stateBins[i][0])) for i, f in enumerate(features)]
    ###
    # print(f'bin {bin}')
    # print(f'binIndex: {int(np.sum(bin * np.power(len(stateBins[0]), np.linspace(len(features)-1, 0, len(features)))))}')
    # stateIndex = feature1*binSize^numRestFeatures + feature2*binSize^numRestFeatures...
    # stateIndex = int(np.sum(bin * np.power(len(stateBins[0]), np.linspace(len(features)-1, 0, len(features)))))
    return [stateBins[i, f] for i, f in enumerate(bin)]


stateBins = binThese([-5, 0, 0.08], [5, np.inf, 0.2], bins=10, cutOffInf=5)
allStates = cartesian(stateBins)

print(stateBins)
randomState = [-4.8, 0.7, 0.104]
print(f'randomState: {randomState}')
bin = [np.where(stateBins[i] <= f)[0][-1] for i, f in enumerate(randomState)]
print(f'bin indices: {bin}')
print(f'bin should be [-5, 0.5, 0.092]: {getState(stateBins, randomState)}')
print(f'bin should be [-5, 0.5, 0.092]: {getStateNew(stateBins, randomState)}')

randomState = [-5, 4.8, 0.163]
print(f'randomState: {randomState}')
print(f'bin should be [-5, 4.5, 0.152]: {getState(stateBins, randomState)}')
print(f'bin should be [-5, 4.5, 0.152]: {getStateNew(stateBins, randomState)}')

randomState = [2.4, 2.3, 0.180]
print(f'randomState: {randomState}')
print(f'bin should be [2, 2, 0.176]: {getState(stateBins, randomState)}')
print(f'bin should be [2, 2, 0.176]: {getStateNew(stateBins, randomState)}')

startTime = time.time()
getState(stateBins, randomState)
print(f'Old: {time.time()-startTime}')
start = time.time()
getStateNew(stateBins, randomState)
print(f'New: {time.time()-startTime}')
