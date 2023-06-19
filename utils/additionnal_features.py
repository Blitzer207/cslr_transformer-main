"""
This file contains scripts for the different data augmentation technique used.
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.io import savemat


def get_frame_diff(data: np.ndarray): # data是一个numpy数组，每一行代表一帧的特征，形状是[帧数，特征数]=[n,398 or 258]
    """
    Computes the frame difference between two frames
    :param data: Input with extracted features per frame
    :return:
    
    这个函数计算两个连续帧之间的差异。它接受一个numpy数组作为输入，该数组中每一行代表一帧的特征。
    函数会遍历每一帧，并计算当前帧和下一帧之间的欧氏距离。然后将这些距离保存在一个新的numpy数组中，并返回。
    """
    y = np.zeros(shape=(data.shape[0]), dtype=float) # data.shape[0]是帧数
    for i in range(data.shape[0]):
        if i == 0 or i == data.shape[0] - 1: # 如果是第一帧或最后一帧，那么就将y[i]置为0
            y[i] = 0.0
        else:
            next = i + 1
            y[i] = np.linalg.norm(data[next] - data[i]) # 计算两帧之间的欧氏距离，欧氏距离是两点之间的距离，负数取绝对值，np.linalg.norm()是计算向量的范数，范数是绝对值的推广
    return np.array(y)


def center(data: np.ndarray): #返回一个numpy数组，每一行代表一帧的特征，形状是[帧数，特征数]=[n,3]
    """
    Computes the center of both hands
    :param data: ND array of data extracted.
    :return: center of the hand per frame.
    
    这个函数计算两只手的中心位置。它接受一个numpy数组作为输入，该数组中每一行代表一帧的特征。
    函数会遍历每一帧，并计算每只手的x、y和z坐标的平均值，然后将这些坐标保存在两个新的numpy数组中，并返回。
    因为data每一帧的特征是[左手x，左手y，左手z，右手x，右手y，右手z]，所以每次循环都是加3
    """
    left_hand, right_hand = [], []
    for row in data:
        x, y, z = 0, 1, 2
        x_arr, y_arr, z_arr = [], [], []
        while z != 128: #125 是[0,1,2,3,...,123,124,125]=[x_L1,y_L1,Z_L1,...,x_R21,y_R21,z_R21]手部信息的最后一个特征的索引
            x_arr.append(row[x])
            x += 3
            y_arr.append(row[y])
            y += 3
            z_arr.append(row[z])
            z += 3
            if len(z_arr) == 21 and z == 65: # 21是左手的特征数，62是左手的最后一个特征的索引
                left_hand.append([np.mean(x_arr), np.mean(y_arr), np.mean(z_arr)])
                x_arr, y_arr, z_arr = [], [], []

        right_hand.append([np.mean(x_arr), np.mean(y_arr), np.mean(z_arr)])
    return np.array(left_hand, dtype=float), np.array(right_hand, dtype=float)


def center_face(data: np.ndarray): #返回两个numpy数组，每一行代表一帧的特征，形状是[帧数，特征数]=[n,2]
    """
    Computes the center of both hands when the additional face landmarks are considered
    :param data: ND array of data extracted.
    :return: center of the hand per frame.
    """
    left_hand, right_hand = [], []
    for row in data:
        x, y = 0, 1
        x_arr, y_arr = [], []
        while y != 83: #这里不应该是 85吗？
            x_arr.append(row[x])
            x += 2
            y_arr.append(row[y])
            y += 2
            if len(y_arr) == 21 and y == 43:
                left_hand.append([np.mean(x_arr), np.mean(y_arr)])
                x_arr, y_arr = [], []

        right_hand.append([np.mean(x_arr), np.mean(y_arr)])
    return np.array(left_hand, dtype=float), np.array(right_hand, dtype=float)


def speed(part: str, data: np.ndarray): #这里应用在FACE_REDUCTION is Ture 返回每一张图片左右手的x,y,z速度
    """
    Computes speed of each hand in the picture.
    :param part: Specify the part to use to compute the speed
    :param data: ND Array of extracted data.
    :return: Speed
    """
    speed_left, speed_right = [], []
    time = 0.04
    if part.lower() == "center": #利用每一帧所有手部关键点的均值作为中心点来计算速度
        left, right = center(data) # center()返回两个numpy数组，每一行代表一帧的特征，形状是[帧数，特征数]=[n,3]
        for i in range(left.shape[0]):
            if i == 0 or i == left.shape[0] - 1: # 如果是第一帧或最后一帧，那么就将speed_left[i]置为[0.0, 0.0, 0.0]
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
    elif part.lower() == "marker": #左右手63个特征点的x,y,z速度
        for idx in range(data.shape[0]): # data的形状是[帧数，特征数]=[n,258] without face，[n,398] with face
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


def speed_face(part: str, data: np.ndarray):#这里应用在FACE_REDUCTION is False 返回每一张图片左右手的x,y速度
    """
    Computes speed of each hand in the picture when the additional face landmarks are considered
    :param part: Specify the part to use to compute the speed
    :param data: ND Array of extracted data.
    :return: Speed
    """
    speed_left, speed_right = [], []
    time = 0.04
    if part.lower() == "center": #利用每一帧所有手部关键点的均值作为中心点来计算速度
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
    elif part.lower() == "marker": #左右手42个特征点的x,y速度
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
        hand: str, #用来指定左右手
        axis: str, #x,y,z
        data: np.ndarray,#包含了输入的数据。
        name: str, #用来指定保存图片的文件名
        show: bool = False, #是否显示图片
        factor: float = 3.0, #用于调整图片大小的因子
):
    """
    Draw evolution of the hand displacement #画出手部位移的演变
    :param name: name for storing
    :param axis: specify which axis to consider. x,y or z
    :param hand: specify which hand to consider.
    :param factor: multiplication factor.
    :param data: input data
    :param show: Indicate if the plot should be shown
    :return:
    """
    axe = None
    x, y, u, v = 0, 0, 0, 0 #x,y是起始点，u,v是终点
    left_hand, right_hand = center(data=data) #返回的是左右手的中心点，返回形状是[n,3]
    fig, ax = plt.subplots() #创建一个新的图形
    if axis.lower() == "x": #指定x,y,z轴
        axe = 0
    elif axis.lower() == "y":
        axe = 1
    elif axis.lower() == "z":
        axe = 2
    else:
        raise ValueError("Please set the parameter axis as string . x,y or z")

    if hand.lower() == "left": #指定左右手
        for idx, line in enumerate(left_hand): #enumerate()函数用于将一个可遍历的数据对象组合为一个索引序列，同时列出数据和数据下标
            if idx == 0:
                ax.quiver(0.0, 0.0, 0.0, 0.0, units="xy", scale=5) #画出箭头，四个参数分别是起点x,y，终点终点u,v
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
                y = float(right_hand[idx - 1][axe] * factor) #这里的factor是用来调整图片大小的
                u = float(0)
                v = float(right_hand[idx][axe] * factor)
                ax.quiver(x, y, u, v, units="xy", scale=5)
    else:
        raise ValueError("Please set the parameter hand to left or right as string")
    plt.grid() #画出网格
    plt.xlim(-1, left_hand.shape[0]) #设置x轴的范围
    plt.ylim(-1, int(factor)) #设置y轴的范围

    plt.title(
        "Hand motion analysis of the " + hand + " hand on the " + axis + " axis", #设置标题
        fontsize=10,
    )
    plt.xlabel(r"$F_n$") #设置x轴的标签
    plt.ylabel(r"$\vec v_x$") #设置y轴的标签
    if show is True:
        plt.show()
        plt.clf()
    else:
        plt.savefig( #保存图片
            name
            + "/Hand motion analysis of the "
            + hand
            + " hand on the "
            + axis
            + " axis"
        )
        plt.clf()

    return
