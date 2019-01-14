import datetime
import os

import torch

from src.policy.policy import _Policy
from src.representation.representation import _RepresentationLearner


def update_agent_model(current, target):
    """
    Method that updates the target model with the values of the one used to train and explore continuously.

    :param current: model used to train continuously
    :param target: model that will use the agent after finishing the training
    """
    target.load_state_dict(current.state_dict().copy())


def save_model(model, filename: str):
    """
    Method to save the network configuration to disk. Filename could also include the route where the file
    will be created.

    :param model: current model to be saved
    :param filename: filename of the file where the model will be saved.
    """
    torch.save(model.state_dict(), filename)


def load_model(model, config_file: str):
    """
    Method that retrieve the configuration from a file and update the configuration of the passed model.

    :param model:
    :param config_file:
    """
    config = torch.load(config_file)
    model.load_state_dict(config)


def save_checkpoint(state: {}, episode: int, out_dir: str, learner: str) -> None:
    """
    Method that saves a dictionary to checkpoint in output directory/file

    :param state: the state to save
    :param episode: current episode
    :param out_dir: checkpoint directory
    :param learner: which learner it is (policy or representation)
    :return:
    """
    final_dir = os.path.join(out_dir, str(episode))
    if not os.path.exists(os.path.join(final_dir)):
        os.makedirs(final_dir)
    torch.save(state, os.path.join(final_dir, "{}_{}.ckpt".format(learner, episode)))


def apply_checkpoint(policy: _Policy, repr: _RepresentationLearner, ckpt_path: str) -> int:
    """
    Method that loads training status stored in checkpoint

    :param policy: policy learner
    :param repr: repr learner
    :param ckpt_path: path to checkpoint folder,
    for example "ckpt/ParallelAgent_ObstaclePathing_JanusPixel_DoubleDeepQNetwork/TIMESTAMP"
    :return: latest episode saved in checkpoint
    """
    assert os.path.isdir(ckpt_path)

    latest_episode = max([int(dir_name) for dir_name in os.listdir(ckpt_path)])
    policy_path = os.path.join(ckpt_path, str(latest_episode), 'policy_{}.ckpt'.format(latest_episode))
    repr_path = os.path.join(ckpt_path, str(latest_episode), 'repr_{}.ckpt'.format(latest_episode))

    assert os.path.isfile(policy_path)
    assert os.path.isfile(repr_path)

    policy.restore_from_state(torch.load(policy_path))
    repr.restore_from(torch.load(repr_path))
    print("=> loaded checkpoint '{}'".format(ckpt_path))

    return latest_episode


def get_checkpoint_dir(config: str) -> str:
    """
    Method that generates check point directory name based on agent type, representation learner type,
    policy type, and current time stamp

    :param config: configuration description (agent type, representation learner type, policy type)
    :return: directory name
    """
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    return os.path.join(os.getcwd(), "ckpt", config, timestamp)
