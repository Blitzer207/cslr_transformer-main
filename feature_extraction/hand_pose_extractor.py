"""
This file contains methods for double-checking the number of extracted hands in the video.
Mainly used to know and confirm missing landmarks in the hand region.
Library from @https://github.com/cvzone/cvzone
"""

import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np

detector = HandDetector(detectionCon=0.5, maxHands=2)


def extract_hands(
    path_to_image: str, show: bool = False
) -> (np.ndarray, np.ndarray, int):
    """
    Extracts the X,Y and Z coordinates of the 21 Landmarks of the image.
    :param show: Bool to specify if a plot of the result is necessary.
    :param path_to_image: path to image file
    :return: The number of detected hands.
    """
    left_hand = []
    right_hand = []
    n_detected_hands = 0
    image = cv2.imread(filename=path_to_image)
    hands, image = detector.findHands(image)
    if hands:
        # Hand 1
        hand1 = hands[0]
        lmList1 = hand1["lmList"]
        bbox1 = hand1["bbox"]
        centerPoint1 = hand1["center"]
        handType1 = hand1["type"]

        fingers1 = detector.fingersUp(hand1)
        n_detected_hands = 1
        if len(hands) == 2:
            # Hand 2
            hand2 = hands[1]
            lmList2 = hand2["lmList"]
            bbox2 = hand2["bbox"]
            centerPoint2 = hand2["center"]
            handType2 = hand2["type"]

            fingers2 = detector.fingersUp(hand2)
            n_detected_hands = 2

    if show is True:
        cv2.imshow("Image", image)
        cv2.waitKey(3000)

    return n_detected_hands
