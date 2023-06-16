"""
This file contains a self adapted dataloader for the videos and their ground truth sentences.
"""
import torch
import numpy as np
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    """
    Dataset loader for videos
    """

    def __init__(self, feature_tab: np.ndarray, label: np.ndarray):
        """
        Initial constructor
        :param feature_tab: ND Array containing video features
        :param label : ND Array containing ground truth
        """
        self.label = label
        self.feature_tab = feature_tab

    def __len__(self):
        return len(self.feature_tab)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        label = torch.LongTensor(self.label[idx])
        feature = torch.from_numpy(self.feature_tab[idx])
        feature = torch.DoubleTensor(feature)
        return feature, label
