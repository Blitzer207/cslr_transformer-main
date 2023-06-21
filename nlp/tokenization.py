"""
This file contains script for tokenization an
"""
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
import numpy as np
import os

MULTISINGER = True
NAME = "info_multi.npy" if MULTISINGER else "info_si5.npy"
CURRENT = os.getcwd().split("nlp")[0]


def tokenize(dataset_split: str): #查看单个数据集中单词的数量和频率分布
    """
    Gets the number of tokens and their frequency distribution of a dataset split.
    :param dataset_split: specify the dataset split to be used. Takes value train, dev, test.
    :return: tokens, frequency distribution
    """
    full_text = None
    NAME = "info_multi.npy" if MULTISINGER else "info_si5.npy"
    if dataset_split != "train" and dataset_split != "dev" and dataset_split != "test": #判断数据集是否正确
        raise ValueError(
            "Please give a valid dataset split. This is either train, dev or test"
        )
    else:
        dataset_split = "/" + dataset_split + "_"
        data = np.load(CURRENT + dataset_split + NAME, allow_pickle=True) #加载数据集
        full_text = " ".join(s[-1] for s in data) #将数据集中的单词拼接成一个字符串
    tokens = word_tokenize(full_text, language="german") #对字符串进行分词，将标点符号和单词分叉开了
    freq_distribution = FreqDist(tokens) #统计每个单词出现的次数
    return tokens, freq_distribution


def tokenize_all(save: bool = False): # 查看整个数据集中单词的数量和频率分布
    """
    Gets the number of tokens and their frequency distribution of the whole dataset.
    :param: save: indicates whether to save the frequency distribution or not
    :return: tokens, frequency distribution
    """
    NAME = "info_multi.npy" if MULTISINGER else "info_si5.npy"
    file_name = "full_gloss_multi.npy" if MULTISINGER else "full_gloss_si5.npy"
    train = np.load(CURRENT + "/train_" + NAME, allow_pickle=True) #加载数据集
    test = np.load(CURRENT + "/test_" + NAME, allow_pickle=True)
    dev = np.load(CURRENT + "/dev_" + NAME, allow_pickle=True)
    full_train = " ".join(s[-1] for s in train) #将数据集中的单词拼接成一个字符串
    full_dev = " ".join(s[-1] for s in dev)
    full_test = " ".join(s[-1] for s in test)
    full = full_train + " " + full_dev + " " + full_test #将三个数据集的字符串拼接成一个字符串
    tokens = word_tokenize(full, language="german") #对字符串进行分词
    freq_distribution = FreqDist(tokens) 

    if save is True:
        vocab = {}
        for idx, word in enumerate(freq_distribution):
            vocab[word] = idx
        np.save(file_name, vocab)
    return tokens, freq_distribution
