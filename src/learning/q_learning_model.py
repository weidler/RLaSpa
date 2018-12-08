import gym
import numpy as np
import torch
from src.representation.autoencoder import Autoencoder
from sklearn.utils.extmath import cartesian
import matplotlib.pyplot as plt
from src.utils.task_dictionary import task_dict
from src.utils.exploration import boltzmann_explore
import pickle
import time


def binThese(lowerBound, upperBound, bins=10, cutOffInf=10000):
    assert len(upperBound) == len(lowerBound)
    numFeatures = len(upperBound)
    featureBins = np.zeros((numFeatures, bins + 1))
    for i in range(numFeatures):
        lb = -cutOffInf if lowerBound[i] <= -cutOffInf else lowerBound[i]
        ub = cutOffInf if upperBound[i] >= cutOffInf else upperBound[i]
        featureBins[i] = np.histogram_bin_edges([lb, ub], bins=bins)
    return featureBins[:, :-1]


def binModel(lowerBound, upperBound, numFeatures, bins=10):
    featureBins = np.zeros((numFeatures, bins + 1))
    for i in range(numFeatures):
        lb = lowerBound
        ub = upperBound
        featureBins[i] = np.histogram_bin_edges([lb, ub], bins=bins)
    return featureBins[:, :-1]


def getState(stateBins, features):
    # print(len(stateBins))
    # print(len(features))
    assert len(stateBins) == len(features)
    bin = [np.where(stateBins[i] <= f)[0][-1] for i, f in enumerate(features)]
    # stateIndex = feature1*binSize^numRestFeatures + feature2*binSize^numRestFeatures...
    stateIndex = int(np.sum(bin * np.power(len(stateBins[0]), np.linspace(len(features) - 1, 0, len(features)))))
    return stateIndex


def run(task_names, num_iters=1, alpha=0.9, gamma=0.99, num_episodes=1000, verbose=False):

    for iter_idx in range(num_iters):

        for task_name in task_names:

            env = gym.make(task_dict[task_name])
            env.reset()  # returns initial state
            q_values = np.zeros([len(allStates), env.action_space.n])
            episode_reward = np.zeros(total_episode)

            for episode in range(1, num_episodes):
                done = False
                total_reward, reward = 0, 0
                state = env.reset()
                # here we need to find the state of all states
                # transform to encoder latent representation
                # TODO: in case of sas.model, also need to append action
                # state = np.pad(state, (0, 2), 'constant', constant_values=(0))

                # print(len(state), num_latent_features)
                state = np.pad(state, (0, num_input_features - len(state)), mode='constant')
                state = net.activation(net.encoder(torch.tensor(state).float())).tolist()

                stateInd = getState(stateBins, state)
                updateQ = []
                while not done:
                    action = boltzmann_explore(q_values[stateInd], T)

                    state, reward, done, info = env.step(action)  # 2
                    total_reward += reward

                    state = np.pad(state, (0, num_input_features - len(state)), mode='constant')
                    latentRepr = net.activation(net.encoder(torch.tensor(state).float())).tolist()
                    indState2 = getState(stateBins, latentRepr)  # find the state bin

                    # qValues[stateInd, action] += alpha * (reward + np.max(qValues[indState2]) - qValues[stateInd, action])  # 3
                    q_values[stateInd, action] = reward + gamma * np.max(q_values[indState2])  # 3

                    # add state and action for later backtracking
                    updateQ.append([stateInd, action])
                    stateInd = indState2
                    if done:
                        updateQ.pop()

                # update the qValues that led to that state
                while len(updateQ) > 0:
                    oldStateInd, oldaction = updateQ.pop()
                    q_values[oldStateInd, oldaction] = reward + gamma * np.max(q_values[stateInd])
                    stateInd, action = oldStateInd, oldaction

                episode_reward[episode] = total_reward

                if episode % 50 == 0:
                    print('Episode {}, Total Reward: {}'.format(episode, total_reward))

            print('Saving q-values...')
            with open('_'.join(['../../data/q_vals', task_name, model_name, str(iter_idx)]) + '.pkl', 'wb') as f:
                pickle.dump(q_values, f)

            if verbose:
                with open('_'.join(['../../data/rewards', task_name, model_name, str(iter_idx)]) + '.pkl', 'wb') as f:
                    pickle.dump(episode_reward, f)

            print('Testing...')
            done = False
            state = env.reset()
            test_total_reward = 0
            while not done:
                env.render()
                time.sleep(0.01)

                state = np.pad(state, (0, num_input_features - len(state)), mode='constant')
                state = net.activation(net.encoder(torch.tensor(state).float())).tolist()
                stateInd = getState(stateBins, state)

                # stateInd = np.where(allStates == state)[0][0]
                action = np.argmax(q_values[stateInd])
                state, reward, done, _ = env.step(action)
                test_total_reward += reward
            print("Ended with reward: ", test_total_reward)

            env.close()


if __name__ == '__main__':
    task_names = ['cartpole', 'mountain_car']
    model_name = 'very-simple-multi-task'  # which auto-encoder to use

    # get shape of saved weights
    net_weights = torch.load('../../models/' + model_name + '.model')
    encoder_shape = list(net_weights.values())[0].size()
    decoder_shape = list(net_weights.values())[len(net_weights) - 1].size()

    # init autoencoder according to size of loaded weights
    net = Autoencoder(encoder_shape[1], encoder_shape[0], decoder_shape[0])
    net.load_state_dict(torch.load('../../models/' + model_name + '.model'))
    input_template = np.zeros(encoder_shape[1])

    num_input_features = net.encoder.in_features
    num_latent_features = net.encoder.out_features

    num_bins = 100

    stateBins = binModel(0, 1, num_latent_features, bins=num_bins)
    allStates = cartesian(stateBins)  # round

    # totalReward = 0
    alpha = 0.618
    gamma = 0.995
    T = 3
    total_episode = 100000

    run(task_names, 10, alpha, gamma, total_episode, verbose=True)
