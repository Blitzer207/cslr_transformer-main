"""
Initial idea
"""
import cv2
import os
import mediapipe as mp
import numpy as np


def extract_coordinates(path_to_image: str) -> (np.ndarray, np.ndarray, int):
    """
    Extracts the X,Y and Z coordinates of the 21 Landmarks of the image.
    For the moment we are just dealing with X and Y.
    :param path_to_image: path to image file
    :return: n dimensional array number of hands x landmarks*3
    """
    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_tracking_confidence=0.8,
        min_detection_confidence=0.8,
    )
    left_hand = []
    right_hand = []
    n_detected_hands = 0
    image = cv2.flip(cv2.imread(path_to_image), 1)
    # Convert the BGR image to RGB before processing.
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.multi_handedness is None:
        print("Number of detected hands : {0}".format(n_detected_hands))
    else:
        n_detected_hands = len(results.multi_handedness)
        for n in results.multi_handedness:
            h = n.classification[0].label
            i = n.classification[0].index
            if h.lower() == "left":
                land = results.multi_hand_landmarks[i]
                for mark in land.landmark:
                    left_hand.append([mark.x, mark.y])
            else:
                land = results.multi_hand_landmarks[i]
                for mark in land.landmark:
                    right_hand.append((mark.x, mark.y))
    return np.array(left_hand), np.array(right_hand), n_detected_hands


def extract_from_full_video(path_to_video: str) -> list:
    """
    Extracts hand pose for a complete video
    :param path_to_video:
    :return: list of extracted features from all frames
    """
    img_file_template = ".avi_pid0_fn{:06d}-0.png"
    frame_names = os.listdir(path_to_video)
    final = []
    for idx, file in enumerate(frame_names):
        # linux fix
        tab = file.split(".")
        file = tab[0] + img_file_template.format(idx)
        path = os.path.join(path_to_video, file)
        left, right, n_detected_hands = extract_coordinates(path_to_image=path)
        if n_detected_hands == 2:
            final.append([left, right])
        elif n_detected_hands == 0:
            if idx == 0:
                final.append(
                    [
                        np.zeros(shape=(21, 2), dtype=float),
                        np.zeros(shape=(21, 2), dtype=float),
                    ]
                )
            else:
                final.append(final[-1])
        else:
            if idx == 0:
                if left.shape == (0,):
                    final.append([np.zeros(shape=(21, 2), dtype=float), right])
                else:
                    final.append([left, np.zeros(shape=(21, 2), dtype=float)])
            else:
                if left.shape == (0,):
                    final.append([final[-1][0], right])
                else:
                    final.append([left, final[-1][1]])

    return final
