"""
This file contains the script used for subsampling at the beginning of the experimental process
"""
import numpy as np


def based_on_fred_dist(data: np.ndarray, word: str):
    """
    Returns subset of the dataset based on word frequency distribution
    :param data: ND array of video path , number of frames , ground truth
    :param word: Word to use for subsampling
    :return: ND array
    """
    subsample = []
    to_delete = []
    for i in range(data.shape[0]):
        if str(data[i][-1]).find(word.upper()) != -1:
            subsample.append(data[i])
            to_delete.append(i)
    return np.array(subsample), np.delete(data, to_delete, axis=0)
