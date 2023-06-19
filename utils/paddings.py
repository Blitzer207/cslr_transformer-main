"""
This file contains scripts for data padding # 这段代码包含了数据填充的脚本
"""
from nltk.tokenize import word_tokenize
import numpy as np


def text_to_seq(
    vocab: dict, seq: str, final_size: int = 300, padding_character: int = 1136
):
    """
    Transforms sequence of tokens to sequence of ints with additional padding # 将令牌序列转换为带有附加填充的整数序列
    :param vocab: Vocabulary dictionary to use for mapping # 用于映射的词汇表字典
    :param seq: sequence to map # 要映射的序列
    :param final_size: fixed defined length # 固定定义的长度
    :param padding_character: padding character to use when length is not enough. # 长度不够时使用的填充字符。
    :return: fixed length sequences # 固定长度序列
    
    """ ""
    encode = [] # 用于存储编码后的序列
    tokens = word_tokenize(seq, language="german") # 对输入的词语序列进行分词，得到一个词语列表。
    for word in tokens: # ，将每个词语通过词汇字典映射为一个整数，并将这个整数添加到编码列表中。
        encode.append(vocab[word])
    while len(encode) != final_size: # 如果编码列表的长度不等于固定长度，就将填充字符添加到编码列表中。
        encode.append(padding_character)

    return encode


def pad_feature_vector(x: np.ndarray, final_size: tuple = (300, 512)): #将一个二维数组转换为一个固定大小的二维数组，如(300,512)
    """
    Transforms variable input sequence into fixed inout sequences
    :param x: input array
    :param final_size: final size of the output array
    :return: fixed output arrays
    """
    temp = []
    first_dim, last_dim = final_size
    while x.shape[0] != first_dim:
        while x.shape[1] != last_dim:
            diff = int(last_dim - x.shape[1])
            for row in x:
                temp.append(
                    np.concatenate((row, np.zeros(shape=(diff,), dtype=float)), axis=0)
                )
            x = np.array(temp, dtype=float)
        x = np.append(x, [np.zeros(shape=(x.shape[-1],), dtype=float)], axis=0) # [np.zeros(shape=(x.shape[-1],), dtype=float)]是一个二维数组，shape为(1,512)，并最后添加到x数组的最后一行。

    return x
