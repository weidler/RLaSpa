""" Abstract class for custom tasks. """

import abc
import matplotlib.pyplot as plt


class Task(abc.ABC):

    @abc.abstractmethod
    def step(self):
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_pixelbased_representation(self):
        raise NotImplementedError

    def visualize(self, img=None):
        if not img: img = self.get_pixelbased_representation()
        plt.imshow(img, cmap="binary", origin="upper")
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.show()
