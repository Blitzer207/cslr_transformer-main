"""
This file contains scripts for data padding
"""
from nltk.tokenize import word_tokenize
import numpy as np


def text_to_seq(
    vocab: dict, seq: str, final_size: int = 300, padding_character: int = 1136
):
    """
    Transforms sequence of tokens to sequence of ints with additional padding
    :param vocab: Vocabulary dictionary to use for mapping
    :param seq: sequence to map
    :param final_size: fixed defined length
    :param padding_character: padding character to use when length is not enough.
    :return: fixed length sequences
    """ ""
    encode = []
    tokens = word_tokenize(seq, language="german")
    for word in tokens:
        encode.append(vocab[word])
    while len(encode) != final_size:
        encode.append(padding_character)

    return encode


def pad_feature_vector(x: np.ndarray, final_size: tuple = (300, 512)):
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
        x = np.append(x, [np.zeros(shape=(x.shape[-1],), dtype=float)], axis=0)

    return x
