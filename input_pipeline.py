import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class PairsDataset(Dataset):

    def __init__(self, folder, size, is_training):
        """
        Arguments:
            folder: a string, the path to a folder with images.
            size: an integer.
            is_training: a boolean.
        """
        self.names = os.listdir(folder)
        self.folder = folder
        self.size = (size, size)
        self.is_training = is_training

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        """
        Returns:
            A: a float tensor with shape [1, h, w].
            B: a float tensor with shape [3, h, w].
        """

        name = self.names[i]
        path = os.path.join(self.folder, name)
        image = cv2.imread(path)

        edges, image = np.split(image, 2, axis=1)
        edges = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)
        edges = np.expand_dims(edges, 2)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.is_training:
            image = np.concatenate([edges, image], axis=2)
            image = random_crop(image, gamma=0.9)
            image = random_flip(image)
            image = cv2.resize(image, self.size, cv2.INTER_LINEAR)
            edges, image = np.split(image, [1], axis=2)

        A = torch.FloatTensor(edges).permute(2, 0, 1).div(255.0)
        B = torch.FloatTensor(image).permute(2, 0, 1).div(255.0)
        return A, B


def random_crop(image, gamma=0.5):
    """
    Arguments:
        image: a numpy array with shape [h, w, c].
        gamma: a float number from zero to one.
    Returns:
        a numpy array with shape [s, s, c],
        where gamma * min(h, w) < s < min(h, w).
    """
    h, w, _ = image.shape
    min_dimension = min(h, w)

    # size and position of a crop
    size = np.random.randint(int(min_dimension * gamma), min_dimension)
    y = np.random.randint(0, height - size)
    x = np.random.randint(0, width - size)

    ymin, xmin, ymax, xmax = y, x, y + size, x + size
    image = image[ymin:ymax, xmin:xmax]
    return image


def random_flip(image):

    if np.random.rand() > 0.5:
        image = np.flip(image, axis=1)

    return image
