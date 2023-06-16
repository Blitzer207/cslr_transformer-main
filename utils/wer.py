"""
This file contains scripts to compute the word error rate and print out all results.
"""
from jiwer import wer, compute_measures


def seq_to_text(
        ground_truth,
        prediction,
        vocab: dict,
        padding_character: int = 1136,
        remove_padding: bool = True,
):
    """
    From array retrieve sentence
    :param padding_character: Value used for padding during encoding
    :param vocab: Corpus vocabulary.
    :param ground_truth: Ground truth form data set
    :param prediction: Prediction of model
    :param remove_padding: Remove padding
    :return: Tuple of sentences
    """
    decoded_ground_truth, decoded_prediction = "", ""
    if remove_padding is True:
        dismiss = 0
        for value in ground_truth:
            if value != padding_character:
                dismiss += 1
            else:
                break
        ground_truth = ground_truth[:dismiss]
        prediction = prediction[:dismiss]
    for value in ground_truth:
        if value == padding_character:
            decoded_ground_truth = decoded_ground_truth + " PADDING"
        for key, values in vocab.items():
            if values == value:
                decoded_ground_truth = decoded_ground_truth + " " + key
    for value in prediction:
        if value == padding_character:
            decoded_prediction = decoded_prediction + " PADDING"
        for key, values in vocab.items():
            if values == value:
                decoded_prediction = decoded_prediction + " " + key
    return decoded_ground_truth, decoded_prediction


def seq_to_text_reverse(
        ground_truth,
        prediction,
        vocab: dict,
        padding_character: int = 1136,
        remove_padding: bool = True,
):
    """
    From array retrieve sentence
    :param padding_character: Value used for padding during encoding
    :param vocab: Corpus vocabulary.
    :param ground_truth: Ground truth form data set
    :param prediction: Prediction of model
    :param remove_padding: Remove padding
    :return: Tuple of sentences
    """
    decoded_ground_truth, decoded_prediction = "", ""
    if remove_padding is True:
        dismiss = 0
        for value in ground_truth:
            if value != padding_character:
                dismiss += 1
            else:
                dismiss = 300
                break
        ground_truth = ground_truth[:dismiss]
        for value in reversed(prediction):
            if value == padding_character:
                dismiss -= 1
            else:
                break
        prediction = prediction[:dismiss]
    for value in ground_truth:
        if value == padding_character:
            decoded_ground_truth = decoded_ground_truth + " PADDING"
        for key, values in vocab.items():
            if values == value:
                decoded_ground_truth = decoded_ground_truth + " " + key
    for value in prediction:
        if value == padding_character:
            decoded_prediction = decoded_prediction + " PADDING"
        for key, values in vocab.items():
            if values == value:
                decoded_prediction = decoded_prediction + " " + key
    return decoded_ground_truth, decoded_prediction


def compute_wer(ground_truth: list, prediction: list) -> float:
    """
    Computes the word error rate metric
    :param ground_truth: array of ground truth
    :param prediction: array of predictions
    :return: WER in percentage
    """

    return wer(truth=ground_truth, hypothesis=prediction)


def compute_wer_complete(ground_truth: list, prediction: list) -> float:
    """
    Computes the word error rate metric
    :param ground_truth: array of ground truth
    :param prediction: array of predictions
    :return: WER in percentage and number of insertions, deletions and substitution errors.
    """
    result = compute_measures(truth=ground_truth, hypothesis=prediction)

    return result


if __name__ == "__main__":
    print(
        compute_wer(
            [
                "I am building a transformer model to perform German automatic speech recognition"
            ],
            [
                "I am building a transformer model to perform English automatic sign language recognition"
            ],
        )
    )
