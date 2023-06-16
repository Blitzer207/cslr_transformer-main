"""
This file contains the main feature extraction methods.
It uses the MediaPipe Framework from Google @ https://google.github.io/mediapipe/
"""
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import mediapipe as mp
from utils.paddings import pad_feature_vector
from feature_extraction.hand_pose_extractor import extract_hands
from utils.imputation import impute
from utils.additionnal_features import get_frame_diff, speed, center, center_face, speed_face
from data_preprocessing.preprocess import resize_img

mp_holistic = mp.solutions.holistic.Holistic(
    static_image_mode=False,
    model_complexity=2,
    enable_segmentation=True,
    refine_face_landmarks=True,
)
mp_drawing = mp.solutions.drawing_utils
FACE_REDUCTION: bool = True
FACE_LANDMARKS_TO_CONSIDER = [464, 185, 40, 39, 37, 0, 267, 270, 269, 409,
                              453, 146, 91, 181, 84, 17, 314, 405, 321, 375,
                              452, 184, 74, 73, 11, 302, 303, 304, 408, 384,
                              451, 77, 90, 180, 85, 16, 315, 404, 320, 375,
                              450, 199, 200, 386, 387, 388, 466, 163, 359, 385,
                              449, 448, 227, 234, 116, 123, 117, 111, 118, 50,
                              348, 261, 330, 266, 347, 280, 425, 352, 346, 22,
                              221, 156, 46, 53, 52, 65, 55, 113, 225, 224, 223,
                              247, 30, 29, 27, 28, 56, 190, 130, 33, 246, 161,
                              160, 159, 158, 157, 173, 31, 228, 229, 230, 231,
                              232, 233, 285, 295, 282, 283, 276, 413, 441, 442,
                              443, 444, 445, 342, 414, 286, 258, 257, 259, 260,
                              467, 362, 398]


def plot(image, results):
    """
    Plot results of extracted points
    :param image: BGR Image
    :param results: results of landmark extractions
    :return:
    """
    # image[image < 255] = 255
    # plt.imshow(image)
    if FACE_REDUCTION is False:
        unique, counts = np.unique(FACE_LANDMARKS_TO_CONSIDER, return_counts=True)
        for i in range(478):
            if i not in unique:
                results.face_landmarks.landmark[i].x = np.nan
                results.face_landmarks.landmark[i].y = np.nan
                results.face_landmarks.landmark[i].z = np.nan
        mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            mp.solutions.holistic.FACEMESH_CONTOURS,
            mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1),
        )
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp.solutions.holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=1, circle_radius=1),
    )
    mp_drawing.draw_landmarks(
        image,
        results.left_hand_landmarks,
        mp.solutions.holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=1, circle_radius=1),
    )
    mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp.solutions.holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=1, circle_radius=1),
    )

    plt.imshow(image)
    plt.show()


def extract_all(
        path_to_image: str, show: bool = False, preprocessing={"resize": "210x300px"}
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Extract all landmarks from the image.
    :param preprocessing: Dictionary with preprocessing method to apply
    :param show: Bool to specify if a plot of the result is necessary.
    :param path_to_image: path to the image.
    :return: 2 ND arrays of size (21x3) for both the left and the right hand,
    one ND array of size (478x3) for the face
    and one ND array of size (33x4) for the body .
    """
    image = None
    if preprocessing is None:
        image = cv2.imread(filename=path_to_image)
    else:
        image = resize_img(img_path=path_to_image, size=preprocessing["resize"])
    body, face, left_hand, right_hand = [], [], [], []
    image.flags.writeable = False
    results = mp_holistic.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if show is True:
        plot(image=image, results=results)
    if FACE_REDUCTION is True:
        if results.left_hand_landmarks:
            for point in results.left_hand_landmarks.landmark:
                left_hand.append([point.x, point.y, point.z])
        if results.right_hand_landmarks:
            for point in results.right_hand_landmarks.landmark:
                right_hand.append([point.x, point.y, point.z])
        if results.pose_landmarks:
            for point in results.pose_landmarks.landmark:
                body.append([point.x, point.y, point.z, point.visibility])
    else:
        if results.left_hand_landmarks:
            for point in results.left_hand_landmarks.landmark:
                left_hand.append([point.x, point.y])
        if results.right_hand_landmarks:
            for point in results.right_hand_landmarks.landmark:
                right_hand.append([point.x, point.y])
        if results.pose_landmarks:
            for point in results.pose_landmarks.landmark:
                body.append([point.x, point.y])
        if results.face_landmarks:
            unique, counts = np.unique(FACE_LANDMARKS_TO_CONSIDER, return_counts=True)
            for i in range(478):
                if i in unique:
                    face.append([results.face_landmarks.landmark[i].x,
                                 results.face_landmarks.landmark[i].y])
    return np.array(left_hand), np.array(right_hand), np.array(face), np.array(body)


def extract_from_full_video(
        path_to_video: str,
        padding: bool = False,
        frame_diff: bool = True,
        impute_data: bool = True,
        add_speed_center: bool = True,
        add_speed_marker: bool = True,
        add_center: bool = True,
) -> np.ndarray:
    """
    Extracts hand pose for a complete video
    :param add_speed_center: Specify if we add speed of the center of the hand as feature
    :param add_center: if to add the center of each hand or not
    :param add_speed_marker: Specify if we add speed of each marker of the hand as feature
    :param impute_data: Specify if we apply imputation
    :param frame_diff: Specify if we add the frame difference in features
    :param padding: boolean to specify if we do padding
    :param path_to_video: path to the folder of the video
    :return: list of extracted features from all frames
    To successfully deactivate padding, uncomment the lines in the code
    """
    img_file_template = ".avi_pid0_fn{:06d}-0.png"
    frame_names = os.listdir(path_to_video)
    final = []
    final_input = []
    for idx, file in enumerate(frame_names):
        # linux fix
        tab = file.split(".")
        file = tab[0] + img_file_template.format(idx)
        path = os.path.join(path_to_video, file)
        left, right, face, body = extract_all(path_to_image=path)
        num_detected_hands = extract_hands(path_to_image=path)

        if FACE_REDUCTION is False:
            if idx == 0:
                if (
                        left.shape != (0,)
                        and right.shape != (0,)
                        and face.shape != (0,)
                        and body.shape != (0,)
                ):
                    final.append([left, right, face, body])
                    final_input.append(
                        np.concatenate((left, right, face, body), axis=None)
                    )
                else:
                    if left.shape == (0,):
                        if num_detected_hands == 2:
                            left = np.full((21, 2), np.nan)
                        else:
                            left = np.zeros(shape=(21, 2), dtype=float)
                    if right.shape == (0,):
                        if num_detected_hands == 2:
                            right = np.full((21, 2), np.nan)
                        else:
                            right = np.zeros(shape=(21, 2), dtype=float)
                    if face.shape == (0,):
                        face = np.zeros(shape=(124, 2), dtype=float)
                    if body.shape == (0,):
                        body = np.zeros(shape=(33, 2), dtype=float)
                    final.append([left, right, face, body])
                    final_input.append(
                        np.concatenate((left, right, face, body), axis=None)
                    )
            else:
                if (
                        left.shape != (0,)
                        and right.shape != (0,)
                        and face.shape != (0,)
                        and body.shape != (0,)
                ):
                    final.append([left, right, face, body])
                    final_input.append(
                        np.concatenate((left, right, face, body), axis=None)
                    )
                else:
                    """
                    if left.shape == (0,):
                        left = final[-1][0]
                    if right.shape == (0,):
                        right = final[-1][1]
                    """
                    if left.shape == (0,):
                        if num_detected_hands == 2:
                            left = np.full((21, 2), np.nan)
                        else:
                            left = np.zeros(shape=(21, 2), dtype=float)
                    if right.shape == (0,):
                        if num_detected_hands == 2:
                            right = np.full((21, 2), np.nan)
                        else:
                            right = np.zeros(shape=(21, 2), dtype=float)
                    if face.shape == (0,):
                        face = final[-1][2]
                    if body.shape == (0,):
                        body = final[-1][-1]
                    final.append([left, right, face, body])
                    final_input.append(
                        np.concatenate((left, right, face, body), axis=None)
                    )
        else:
            if idx == 0:
                if left.shape != (0,) and right.shape != (0,) and body.shape != (0,):
                    final.append([left, right, face, body])
                    final_input.append(
                        np.concatenate((left, right, face, body), axis=None)
                    )
                else:
                    if left.shape == (0,):
                        if num_detected_hands == 2:
                            left = np.full((21, 3), np.nan)
                        else:
                            left = np.zeros(shape=(21, 3), dtype=float)
                    if right.shape == (0,):
                        if num_detected_hands == 2:
                            right = np.full((21, 3), np.nan)
                        else:
                            right = np.zeros(shape=(21, 3), dtype=float)
                    if body.shape == (0,):
                        body = np.zeros(shape=(33, 4), dtype=float)
                    final.append([left, right, face, body])
                    final_input.append(
                        np.concatenate((left, right, face, body), axis=None)
                    )
            else:
                if left.shape != (0,) and right.shape != (0,) and body.shape != (0,):
                    final.append([left, right, face, body])
                    final_input.append(
                        np.concatenate((left, right, face, body), axis=None)
                    )
                else:
                    """
                    if left.shape == (0,):
                        left = final[-1][0]
                    if right.shape == (0,):
                        right = final[-1][1]
                    """
                    if left.shape == (0,):
                        if num_detected_hands == 2:
                            left = np.full((21, 3), np.nan)
                        else:
                            left = np.zeros(shape=(21, 3), dtype=float)
                    if right.shape == (0,):
                        if num_detected_hands == 2:
                            right = np.full((21, 3), np.nan)
                        else:
                            right = np.zeros(shape=(21, 3), dtype=float)
                    if body.shape == (0,):
                        body = final[-1][-1]
                    final.append([left, right, face, body])
                    final_input.append(
                        np.concatenate((left, right, face, body), axis=None)
                    )
    final_input = np.array(final_input, dtype=float)

    if impute_data is True:
        final_input = impute(final_input)
    if frame_diff is True:
        temp = []
        diff = get_frame_diff(final_input)
        for row in range(final_input.shape[0]):
            temp.append(np.append(final_input[row], diff[row]))
        final_input = np.array(temp, dtype=float)

    if add_center is True:
        temp = []
        left, right = None, None
        if FACE_REDUCTION is False:
            left, right = center_face(final_input)
        else:
            left, right = center(final_input)
        for row in range(final_input.shape[0]):
            temp.append(
                np.concatenate((final_input[row], left[row], right[row]), axis=0)
            )
        final_input = np.array(temp, dtype=float)
    if add_speed_center is True:
        temp = []
        left, right = None, None
        if FACE_REDUCTION is False:
            left, right = speed_face(part="center", data=final_input)
        else:
            left, right = speed(part="center", data=final_input)
        for row in range(final_input.shape[0]):
            temp.append(
                np.concatenate((final_input[row], left[row], right[row]), axis=0)
            )
        final_input = np.array(temp, dtype=float)
    if add_speed_marker is True:
        temp = []
        left, right = None, None
        if FACE_REDUCTION is False:
            left, right = speed_face(part="marker", data=final_input)
        else:
            left, right = speed(part="marker", data=final_input)
        for row in range(final_input.shape[0]):
            temp.append(
                np.concatenate((final_input[row], left[row], right[row]), axis=0)
            )
        final_input = np.array(temp, dtype=float)
    if padding is True:
        final_input = pad_feature_vector(x=final_input)
    return final_input
