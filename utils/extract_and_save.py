"""
This file handles the whole feature extraction task.
"""
from datetime import datetime
import numpy as np
from tqdm import tqdm
from utils.paddings import text_to_seq
from feature_extraction import global_extractor


def extract_features(vocab: dict, data: np.ndarray, name: str, padding_charachter: int):
    """
    Extract features and save them as numpy array.
    :param padding_charachter: Padding character to use when encoding tokens.
    :param vocab: Vocabulary file to use for extraction
    :param data: ND array of video path , number of frame and ground truth
    :param name: name under which to save
    :return: ND array of extracted features and labels
    """
    features, label = [], []
    print("Feature Extraction for " + name + " begins >>>>>>>>>>>>>>>>>>\n")
    now = datetime.now()
    print("Beginning time : ", now)
    for video in tqdm(data, desc="Feature Extraction for " + name):
        path = str(video[0])
        sentence = str(video[-1])
        features.append(global_extractor.extract_from_full_video(path_to_video=path))
        label.append(text_to_seq(vocab=vocab, seq=sentence, padding_character=padding_charachter))
    np.save(name + "_features", arr=np.array(features), allow_pickle=True)
    np.save(name + "_label", arr=np.array(label), allow_pickle=True)
    print("Feature Extraction done <<<<<<<<<<<<<<\n")
    now = datetime.now()
    print("Ending time : ", now)

    return np.array(features), np.array(label)
