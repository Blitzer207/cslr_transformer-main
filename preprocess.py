"""
This file prepares the dataset and performs all preprocessing steps.
"""
from data_preprocessing.data_preparation import PrepareFiles
import nltk
from nlp import tokenization
import numpy as np
from utils import extract_and_save
import argparse
import os

parser = argparse.ArgumentParser(description="Arguments for the preprocessing script")
parser.add_argument(
    "--multisigner",
    type=bool,
    help="Boolean to specify if we use multisigner data or signer independent data.",
    default=True,
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--path_to_dataset",
    type=str,
    default="temp/phoenix2014-release",
    help="String to specify the path to the extracted dataset",
)
args = parser.parse_args()
if __name__ == "__main__":
    nltk.download("punkt")
    print("Preparing data infos....")
    tokenization.MULTISINGER = args.multisigner
    P = PrepareFiles(dataset_dir=args.path_to_dataset)
    P.prepare_set(dataset_split="train", multisigner=args.multisigner)
    P.prepare_set(dataset_split="dev", multisigner=args.multisigner)
    P.prepare_set(dataset_split="test", multisigner=args.multisigner)
    print("Generating vocabulary....")
    tokens, dist = tokenization.tokenize_all(save=True)
    file = open("word_distribution.txt", "w")
    file.write("Word distribution in the currently used dataset:")
    file.write('\n')
    file.write("Overall number of tokens in the dataset: {0} . Vocabulary size of"
               " the dataset: {1}".format(str(len(tokens)), str(len(dist))))
    file.write('\n')
    file.write('\n')
    for k in sorted(dist, key=dist.get, reverse=True):
        file.write(f"{k} : {dist[k]}")
        file.write('\n')
    file.close()
    to_use = "multi" if args.multisigner else "si5"
    PADDING = 1296 if args.multisigner else 1136
    print("Starting feature extraction....")
    if os.path.exists(os.getcwd() + "/" + to_use + "_training_features.npy") and os.path.exists(
            os.getcwd() + "/" + to_use + "_validation_features.npy" and os.path.exists(
                os.getcwd() + "/" + to_use + "_testing_features.npy")
    ):
        print("All data splits  are already available and ready for training")
    else:
        print("Dataset extraction starts...")
        training_set = np.load("train_info_" + to_use + ".npy", allow_pickle=True)
        evaluation_set = np.load("dev_info_" + to_use + ".npy", allow_pickle=True)
        testing_set = np.load("test_info_" + to_use + ".npy", allow_pickle=True)
        vocab = np.load("full_gloss_" + to_use + ".npy", allow_pickle=True)
        vocab = dict(enumerate(vocab.flatten()))
        vocab = vocab[0]
        training_features, training_label = extract_and_save.extract_features(
            vocab=vocab, data=training_set, name=to_use + "_training", padding_charachter=PADDING)
        evaluation_features, evaluation_label = extract_and_save.extract_features(
            vocab=vocab, data=evaluation_set, name=to_use + "_validation", padding_charachter=PADDING)
        testing_features, testing_label = extract_and_save.extract_features(
            vocab=vocab, data=testing_set, name=to_use + "_testing", padding_charachter=PADDING)
        print("All data splits are now available and ready for training")
