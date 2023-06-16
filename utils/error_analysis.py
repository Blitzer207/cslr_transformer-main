"""
This file contains the script to perform error analysis after computation
"""
import os
import re
import pandas as pd
import numpy as np
from jiwer import wer

CURRENT = os.getcwd().split("utils")[0]


def substitution_error(ground_truth: list, prediction: list, multi: bool):
    """
    Finds which words were not well represented in the testing
    :param ground_truth: array of ground truth
    :param prediction: array of predictions
    :param multi: Boolean to indicate if we are dealing with the multisigner or not
    :return: All substitution errors, correct predicted glosses, and possibly signers
    """
    for_analysis = None
    errors = []
    not_errors = []
    signer_err = []
    false_sentences = []
    correct_sent = 0
    if multi is True:
        for_analysis = pd.read_pickle(CURRENT + "/signer_test.pkl")
    for i in range(len(ground_truth)):
        gt_sentence = ground_truth[i]
        pred_sentence = prediction[i]
        if float(wer(truth=gt_sentence, hypothesis=pred_sentence)) == float(0):
            if for_analysis is not None:
                for_analysis = for_analysis.drop(for_analysis[for_analysis["annotation"] == str(gt_sentence)].index)
                signer_err.append(str(gt_sentence))
                print("Correct sentence " + gt_sentence)
            correct_sent += 1
        else:
            gt_sentence = ground_truth[i].split()
            pred_sentence = prediction[i].split()
            in_c = 0
            for j in range(len(gt_sentence)):
                if gt_sentence[j] == pred_sentence[j]:
                    # The -> sign is just a replacement for mapped to . This is used to save space in the plot
                    not_errors.append(gt_sentence[j] + " -> " + pred_sentence[j])
                else:
                    errors.append(gt_sentence[j] + " -> " + pred_sentence[j])
                    in_c += 1
            false_sentences.append([ground_truth[i], in_c])
    print(" {0} sentences fully well predicted".format(correct_sent))
    if multi is True:
        for_analysis.to_pickle("signer_test_after.pkl")
    return not_errors, errors, signer_err


def analyse_errors(file: str):
    """
    Analyses the errors and generate statistics on number of error per signer
    :param file: path to ndarray of false predictions generated in the previous function
    :return: All substitution errors per signer
    """
    c = 0
    signer_error = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    for_analysis = pd.read_pickle(CURRENT + "/signer_test.pkl")
    new_for_analysis = for_analysis.to_numpy()
    signer, annotation = np.hsplit(new_for_analysis, 2)
    annotation, signer = list(annotation), list(signer)
    for i in range(len(signer)):
        signer[i] = re.sub('\W+', '', str(signer[i]))
    for i in range(len(annotation)):
        annotation[i] = re.sub('\W+', '', str(annotation[i]))
    wer = np.load(file)
    for i in range(wer.shape[0]):
        we = str(wer[i][0]).split("\'  \' ")
        w = str(we[0]).replace(" ", "")
        if w in annotation:
            a = annotation.index(w)
            ir = str(signer[a])
            s = int(''.join(x for x in ir if x.isdigit()))
            v = int(''.join(x for x in str(wer[i]) if x.isdigit()))
            signer_error[s - 1] += v
            signer[a] = "good"
        else:
            print("not found" + str(wer[i]))
            c += 1
            y = wer[i]

    unique, counts = np.unique(signer, return_counts=True)
    print("Error not identified", c)
    return


if __name__ == "__main__":
    print(
        substitution_error(
            [
                "I am building a transformer model to perform German automatic speech recognition"
            ],
            [
                "I am building a transformer model to perform English automatic sign language recognition"
            ],
            multi=True),
    )
