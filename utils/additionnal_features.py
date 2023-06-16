"""
This file contains scripts for the different data augmentation technique used.
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.io import savemat


def get_frame_diff(data: np.ndarray):
    """
    Computes the frame difference between two frames
    :param data: Input with extracted features per frame
    :return:
    """
    y = np.zeros(shape=(data.shape[0]), dtype=float)
    for i in range(data.shape[0]):
        if i == 0 or i == data.shape[0] - 1:
            y[i] = 0.0
        else:
            next = i + 1
            y[i] = np.linalg.norm(data[next] - data[i])
    return np.array(y)


def center(data: np.ndarray):
    """
    Computes the center of both hands
    :param data: ND array of data extracted.
    :return: center of the hand per frame.
    """
    left_hand, right_hand = [], []
    for row in data:
        x, y, z = 0, 1, 2
        x_arr, y_arr, z_arr = [], [], []
        while z != 128:
            x_arr.append(row[x])
            x += 3
            y_arr.append(row[y])
            y += 3
            z_arr.append(row[z])
            z += 3
            if len(z_arr) == 21 and z == 65:
                left_hand.append([np.mean(x_arr), np.mean(y_arr), np.mean(z_arr)])
                x_arr, y_arr, z_arr = [], [], []

        right_hand.append([np.mean(x_arr), np.mean(y_arr), np.mean(z_arr)])
    return np.array(left_hand, dtype=float), np.array(right_hand, dtype=float)


def center_face(data: np.ndarray):
    """
    Computes the center of both hands when the additional face landmarks are considered
    :param data: ND array of data extracted.
    :return: center of the hand per frame.
    """
    left_hand, right_hand = [], []
    for row in data:
        x, y = 0, 1
        x_arr, y_arr = [], []
        while y != 83:
            x_arr.append(row[x])
            x += 2
            y_arr.append(row[y])
            y += 2
            if len(y_arr) == 21 and y == 43:
                left_hand.append([np.mean(x_arr), np.mean(y_arr)])
                x_arr, y_arr = [], []

        right_hand.append([np.mean(x_arr), np.mean(y_arr)])
    return np.array(left_hand, dtype=float), np.array(right_hand, dtype=float)


def speed(part: str, data: np.ndarray):
    """
    Computes speed of each hand in the picture.
    :param part: Specify the part to use to compute the speed
    :param data: ND Array of extracted data.
    :return: Speed
    """
    speed_left, speed_right = [], []
    time = 0.04
    if part.lower() == "center":
        left, right = center(data)
        for i in range(left.shape[0]):
            if i == 0 or i == left.shape[0] - 1:
                speed_left.append([0.0, 0.0, 0.0])
            else:
                s_x = (left[i + 1][0] - left[i][0]) / time
                s_y = (left[i + 1][1] - left[i][1]) / time
                s_z = (left[i + 1][2] - left[i][2]) / time
                speed_left.append([s_x, s_y, s_z])
        for i in range(right.shape[0]):
            if i == 0 or i == right.shape[0] - 1:
                speed_right.append([0.0, 0.0, 0.0])
            else:
                s_x = (right[i + 1][0] - right[i][0]) / time
                s_y = (right[i + 1][1] - right[i][1]) / time
                s_z = (right[i + 1][2] - right[i][2]) / time
                speed_right.append([s_x, s_y, s_z])
        return np.array(speed_left), np.array(speed_right)
    elif part.lower() == "marker":
        for idx in range(data.shape[0]):
            if idx == 0 or idx == data.shape[0] - 1:
                speed_left.append(np.zeros(shape=(63,), dtype=float))
                speed_right.append(np.zeros(shape=(63,), dtype=float))
            else:
                speed_row_left, speed_row_right = [], []
                for i in range(126):
                    if i < 63:
                        s = (data[idx + 1][i] - data[idx][i]) / time
                        speed_row_left.append(s)
                    else:
                        s = (data[idx + 1][i] - data[idx][i]) / time
                        speed_row_right.append(s)
                speed_left.append(speed_row_left)
                speed_right.append(speed_row_right)
        return np.array(speed_left), np.array(speed_right)
    else:
        raise ValueError("Invalid argument part. Please specify either center or mark.")


def speed_face(part: str, data: np.ndarray):
    """
    Computes speed of each hand in the picture when the additional face landmarks are considered
    :param part: Specify the part to use to compute the speed
    :param data: ND Array of extracted data.
    :return: Speed
    """
    speed_left, speed_right = [], []
    time = 0.04
    if part.lower() == "center":
        left, right = center_face(data)
        for i in range(left.shape[0]):
            if i == 0 or i == left.shape[0] - 1:
                speed_left.append([0.0, 0.0])
            else:
                s_x = (left[i + 1][0] - left[i][0]) / time
                s_y = (left[i + 1][1] - left[i][1]) / time
                speed_left.append([s_x, s_y])
        for i in range(right.shape[0]):
            if i == 0 or i == right.shape[0] - 1:
                speed_right.append([0.0, 0.0])
            else:
                s_x = (right[i + 1][0] - right[i][0]) / time
                s_y = (right[i + 1][1] - right[i][1]) / time
                speed_right.append([s_x, s_y])
        return np.array(speed_left), np.array(speed_right)
    elif part.lower() == "marker":
        for idx in range(data.shape[0]):
            if idx == 0 or idx == data.shape[0] - 1:
                speed_left.append(np.zeros(shape=(42,), dtype=float))
                speed_right.append(np.zeros(shape=(42,), dtype=float))
            else:
                speed_row_left, speed_row_right = [], []
                for i in range(84):
                    if i < 42:
                        s = (data[idx + 1][i] - data[idx][i]) / time
                        speed_row_left.append(s)
                    else:
                        s = (data[idx + 1][i] - data[idx][i]) / time
                        speed_row_right.append(s)
                speed_left.append(speed_row_left)
                speed_right.append(speed_row_right)
        return np.array(speed_left), np.array(speed_right)
    else:
        raise ValueError("Invalid argument part. Please specify either center or mark.")


def evolution_in_frame(
        hand: str,
        axis: str,
        data: np.ndarray,
        name: str,
        show: bool = False,
        factor: float = 3.0,
):
    """
    Draw evolution of the hand displacement
    :param name: name for storing
    :param axis: specify which axis to consider. x,y or z
    :param hand: specify which hand to consider.
    :param factor: multiplication factor.
    :param data: input data
    :param show: Indicate if the plot should be shown
    :return:
    """
    axe = None
    x, y, u, v = 0, 0, 0, 0
    left_hand, right_hand = center(data=data)
    fig, ax = plt.subplots()
    if axis.lower() == "x":
        axe = 0
    elif axis.lower() == "y":
        axe = 1
    elif axis.lower() == "z":
        axe = 2
    else:
        raise ValueError("Please set the parameter axis as string . x,y or z")

    if hand.lower() == "left":
        for idx, line in enumerate(left_hand):
            if idx == 0:
                ax.quiver(0.0, 0.0, 0.0, 0.0, units="xy", scale=5)
            elif idx == left_hand.shape[0]:
                break
            else:
                x = float(idx)
                y = float(left_hand[idx - 1][axe] * factor)
                u = float(0)
                v = float(left_hand[idx][axe] * factor)
                ax.quiver(x, y, u, v, units="xy", scale=5)

    elif hand.lower() == "right":
        for idx, line in enumerate(right_hand):
            if idx == 0:
                ax.quiver(0.0, 0.0, 0.0, 0.0, units="xy", scale=5)
            elif idx == right_hand.shape[0]:
                break
            else:
                x = float(idx)
                y = float(right_hand[idx - 1][axe] * factor)
                u = float(0)
                v = float(right_hand[idx][axe] * factor)
                ax.quiver(x, y, u, v, units="xy", scale=5)
    else:
        raise ValueError("Please set the parameter hand to left or right as string")
    plt.grid()
    plt.xlim(-1, left_hand.shape[0])
    plt.ylim(-1, int(factor))

    plt.title(
        "Hand motion analysis of the " + hand + " hand on the " + axis + " axis",
        fontsize=10,
    )
    plt.xlabel(r"$F_n$")
    plt.ylabel(r"$\vec v_x$")
    if show is True:
        plt.show()
        plt.clf()
    else:
        plt.savefig(
            name
            + "/Hand motion analysis of the "
            + hand
            + " hand on the "
            + axis
            + " axis"
        )
        plt.clf()

    return
