import torch


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
