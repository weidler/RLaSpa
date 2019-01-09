import torch
import os, datetime

def update_agent_model(current, target):
    """
    Method that updates the target model with the values of the one used to train and explore continuously.

    :param current: model used to train continuously
    :param target: model that will use the agent after finishing the training
    """
    target.load_state_dict(current.state_dict())


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


def save_checkpoint(state: {}, out_dir: str, filename: str) -> None:
    """
    Method that saves a dictionary to checkpoint in output directory/file

    :param state:
    :param out_dir:
    :param filename:
    :return:
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    torch.save(state, os.path.join(out_dir, filename))


def load_checkpoint(policy, ckpt_path: str) -> int:
    """
    Method that loads training status stored in checkpoint to policy

    :param policy: a subclass extending from policy
    :param ckpt_path: path to checkpoint
    :return: starting episode (episode stored in check point)
    """
    if os.path.isfile(ckpt_path):
        print("=> loading checkpoint '{}'".format(ckpt_path))
        checkpoint = torch.load(ckpt_path)
        policy.restore_from_state(checkpoint)
        start_episode = checkpoint["episode"]
        print("=> loaded checkpoint '{}' (episode {})".format(ckpt_path, start_episode))
        return start_episode
    else:
        print("=> cannot find checkpoint '{}', aborting".format(ckpt_path))
        exit()


def get_checkpoint_dir(config: str) -> str:
    """
    Method that generates check point directory name based on agent type, representation learner type,
    policy type, and current time stamp

    :param config: configuration description (agent type, representation learner type, policy type)
    :return: directory name
    """
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    return os.path.join(os.getcwd(), "ckpt", config, timestamp)
