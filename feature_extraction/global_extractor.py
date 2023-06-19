"""
This file contains the main feature extraction methods.
It uses the MediaPipe Framework from Google @ https://google.github.io/mediapipe/
"""

'''
Q1:
Face_Reudction = True 是不是代表不提取面部特征，转而只手部和姿势特征？
Face_Reudction = False 是不是代表只提取被考虑的面部特征，手部和姿势特征？
'''
import cv2 
import numpy as np
import os
from matplotlib import pyplot as plt
import mediapipe as mp #用来检测人脸、手部、姿势的关键点

from utils.paddings import pad_feature_vector #用于填充特征向量
from feature_extraction.hand_pose_extractor import extract_hands #用于提取手的数量
from utils.imputation import impute #用于填充缺失值
from utils.additionnal_features import get_frame_diff, speed, center, center_face, speed_face #包括帧差、速度、中心、中心面、速度面
from data_preprocessing.preprocess import resize_img #用于缩放图片

# 初始化mediapipe全面模型
mp_holistic = mp.solutions.holistic.Holistic(  #holistic:整体的
    static_imageee c_mode=False, #静态图像模式，如果为True，则输入为一组图像，如果为False，则输入为视频流
    model_complexity=2, #模型复杂度，1为轻量级，2为重量级，3为完全级，越高越准确，但是越慢
    enable_segmentation=True, #启用分割，将图像分为背景和前景，前景是人，背景是其他，这样可以更好的检测人
    refine_face_landmarks=True, #细化面部关键点，例如眼睛、眉毛、嘴巴和鼻子，然后只关注部分关键点
)
mp_drawing = mp.solutions.drawing_utils #绘图工具，用于绘制关键点，例如手、面部、姿势

FACE_REDUCTION: bool = True #是否减少面部关键点
FACE_LANDMARKS_TO_CONSIDER = [464, 185, 40, 39, 37, 0, 267, 270, 269, 409,
                              453, 146, 91, 181, 84, 17, 314, 405, 321, 375,
                              452, 184, 74, 73, 11, 302, 303, 304, 408, 384,
                              451, 77, 90, 180, 85, 16, 315, 404, 320, 375,
                              450, 199, 200, 386, 387, 388, 466, 163, 359, 385,
                              449, 448, 227, 234, 116, 123, 117, 111, 118, 50,
                              348, 261, 330, 266, 347, 280, 425, 352, 346, 22,
                              221, 156, 46, 53, 52, 65, 55, 113, 225, 224, 
                              223,247, 30, 29, 27, 28, 56, 190, 130, 33, 
                              246, 161,160, 159, 158, 157, 173, 31, 228, 229,
                              230, 231, 232, 233, 285, 295, 282, 283, 276, 413, 
                              441, 442,443, 444, 445, 342, 414, 286, 258, 257, 
                              259, 260,467, 362, 398] # 总共125个元素，但是375 出现了两次，所以只有124个不同元素


def plot(image, results):
    """
    Plot results of extracted points
    :param image: BGR Image
    :param results: results of landmark extractions
    :return:
    """
    # image[image < 255] = 255
    # plt.imshow(image)

    # 如果考虑面部特征，忽略的面部特征点设置为nan，Ture代表考虑面部特征，False代表不考虑面部特征
    if FACE_REDUCTION is False:
        unique, counts = np.unique(FACE_LANDMARKS_TO_CONSIDER, return_counts=True) #返回需要关注的面部关键点，不关心他们出现的次数
        for i in range(478):
            if i not in unique: #如果不在需要关注的面部关键点中，则将其置为nan
                results.face_landmarks.landmark[i].x = np.nan
                results.face_landmarks.landmark[i].y = np.nan
                results.face_landmarks.landmark[i].z = np.nan
                
        #绘制面部关键点，绘制的颜色为绿色，绘制的粗细为1，绘制的半径为1
        mp_drawing.draw_landmarks(
            image, #绘制的图像
            results.face_landmarks, #绘制面部关键点
            mp.solutions.holistic.FACEMESH_CONTOURS, #绘制的面部关键点的连接
            mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1), # 指定绘制特征点的颜色、线的粗细和圆的半径
            mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1), # 指定绘制特征点之间连线的颜色、线的粗细和圆的半径
        )
    # 绘制姿势关键点，绘制的颜色为红色，绘制的粗细为1，绘制的半径为1
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp.solutions.holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=1, circle_radius=1),
    )
    
    # 绘制左手关键点，绘制的颜色为蓝色，绘制的粗细为1，绘制的半径为1
    mp_drawing.draw_landmarks(
        image,
        results.left_hand_landmarks,
        mp.solutions.holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=1, circle_radius=1),
    )
    
    # 绘制右手关键点，绘制的颜色为橙色，绘制的粗细为1，绘制的半径为1    
    mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp.solutions.holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=1, circle_radius=1),
    )


    plt.imshow(image) #显示绘制的图像
    plt.show() 

# 从一个图像中提取人体各部位的关键点。这些关键点包括左手、右手、脸部和身体的关键点。函数的输入是一张图像的路径、一个布尔值表示是否展示处理结果以及一个预处理的参数字典。
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
    if preprocessing is None: #如果不进行预处理，则直接读取图像，否则进行预处理
        image = cv2.imread(filename=path_to_image)
    else:
        image = resize_img(img_path=path_to_image, size=preprocessing["resize"]) #调用resize_img函数进行预处理
    body, face, left_hand, right_hand = [], [], [], []
    
    image.flags.writeable = False #将图像设置为只读
    '''
    `results` 是一个 mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList 对象，它包含了 `mp_holistic.process(image)` 处理后的结果。
    mediapipe Holistic 模型用于同时检测面部、手部和姿态关键点。

    下面是 `results` 对象可能包含的一些属性：

    - `face_landmarks`：面部关键点的列表，包含468个关键点，每个关键点包含x, y, z三个坐标值。最终考虑的数量是123个，因为有些关键点不可见。
    - `left_hand_landmarks`：左手关键点的列表，包含21个关键点。[x, y, z]
    - `right_hand_landmarks`：右手关键点的列表，包含21个关键点。 [x, y, z]
    - `pose_landmarks`：身体姿态关键点的列表，包含33个关键点.  [x, y, z, visibility]

    每个关键点是一个 NormalizedLandmark 对象，包含以下属性：

    - `x`，`y` 和 `z`：关键点的坐标。对于2D图像，z通常为0。这些坐标是归一化的，即它们的值在0和1之间。
    - `visibility`：关键点的可见性，值在0和1之间。只有 `pose_landmarks` 包含此属性。

    所以，如果你访问 `results.left_hand_landmarks`，你将得到一个包含21个关键点的列表。每个关键点都是一个 NormalizedLandmark 对象，你可以通过 `.x`, `.y`, `.z` 访问其坐标。

    请注意，如果图像中没有检测到某一类型的关键点（例如，图像中没有左手），那么相应的属性（例如，`results.left_hand_landmarks`）可能为 None。
     '''
    results = mp_holistic.process(image) #调用mp_holistic.process函数，提取图像中的人体各部位的关键点，
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) #将图像从RGB转换为BGR，因为cv2.imshow()函数需要BGR格式的图像。BGR是RGB的倒序，即BGR[0]=RGB[2],BGR[1]=RGB[1],BGR[2]=RGB[0]
    
    # 绘制面部，左右手，姿态关键点
    if show is True:
        plot(image=image, results=results)

    # 如果FACE_REDUCTION为True，则存储左右手，姿态关键点的X,Y,Z 三维坐标和姿态可见性
    if FACE_REDUCTION is True:
        if results.left_hand_landmarks: #这里其实是判断results.left_hand_landmarks是否为None，即是否检测到左手
            for point in results.left_hand_landmarks.landmark:
                left_hand.append([point.x, point.y, point.z]) # left 形状是(21,3)
        if results.right_hand_landmarks:
            for point in results.right_hand_landmarks.landmark:
                right_hand.append([point.x, point.y, point.z]) # right 形状是(21,3)
        if results.pose_landmarks:
            for point in results.pose_landmarks.landmark:
                body.append([point.x, point.y, point.z, point.visibility]) # body 形状是(33,4)
    
    # 如果FACE_REDUCTION为False，则存储左右手，姿态关键点的X，Y二维坐标，以及面部被考虑关键点的X，Y二维坐标
    else:
        if results.left_hand_landmarks:
            for point in results.left_hand_landmarks.landmark:
                left_hand.append([point.x, point.y]) # left 形状是(21,2)
        if results.right_hand_landmarks:
            for point in results.right_hand_landmarks.landmark:
                right_hand.append([point.x, point.y])   # right 形状是(21,2)
        if results.pose_landmarks:
            for point in results.pose_landmarks.landmark:
                body.append([point.x, point.y]) # body 形状是(33,2)
        if results.face_landmarks:
            unique, counts = np.unique(FACE_LANDMARKS_TO_CONSIDER, return_counts=True)
            for i in range(478):
                if i in unique:
                    face.append([results.face_landmarks.landmark[i].x,
                                 results.face_landmarks.landmark[i].y]) # face 形状是(123,2)
    return np.array(left_hand), np.array(right_hand), np.array(face), np.array(body)


# 从一段完整的视频中提取手部关键点
def extract_from_full_video(
        path_to_video: str,
        padding: bool = False, #是否对关键点进行填充
        frame_diff: bool = True, #是否对关键点进行差分
        impute_data: bool = True, #是否对关键点进行插值
        add_speed_center: bool = True, #是否添加手部中心的速度
        add_speed_marker: bool = True, #是否添加手部关键点的速度
        add_center: bool = True, #是否添加手部中心
) -> np.ndarray: # 返回被提取的特征
    """
    Extracts hand pose for a complete video
    :param add_speed_center: Specify if we add speed of the center of the hand as feature
    :param add_center: if to add the center of each hand or not。这里的center是每一张图像中手部关键点每一维均值作为的中心
    :param add_speed_marker: Specify if we add speed of each marker of the hand as feature。 这里的maker是每一张图像中手部关键点的坐标
    :param impute_data: Specify if we apply imputation
    :param frame_diff: Specify if we add the frame difference in features
    :param padding: boolean to specify if we do padding
    :param path_to_video: path to the folder of the video
    :return: list of extracted features from all frames
    To successfully deactivate padding, uncomment the lines in the code
    """
    img_file_template = ".avi_pid0_fn{:06d}-0.png" # 图像文件的命名格式
    frame_names = os.listdir(path_to_video) # 获取视频文件夹中的所有图像文件名，例如 如果你有一个目录 my_directory，它包含两个文件 file1.txt 和 file2.txt，os.listdir('my_directory') 的输出会是 ['file1.txt', 'file2.txt']。
    final = [] 
    final_input = [] 
    # 对于视频中的每一帧 提取手部，面部，姿态关键点，以及手部数量
    for idx, file in enumerate(frame_names): # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
        # linux fix
        tab = file.split(".")
        file = tab[0] + img_file_template.format(idx)# 将图像文件的命名格式与图像文件的索引结合起来，例如，如果图像文件的命名格式为“image_{}.jpg”，图像文件的索引为“1”，则图像文件的名称为“image_1.jpg”
        path = os.path.join(path_to_video, file) # 将图像文件的路径与图像文件的名称结合起来，例如，如果图像文件的路径为“/home/user”，图像文件的名称为“image_1.jpg”，则图像文件的路径为“/home/user/image_1.jpg”
       
        #如果FACE_REDUCTION is False 则 left, right, face, body的形状为（21，2），（21，2），（123，2），（33，2）
        #如果FACE_REDUCTION is True 则 left, right, face, body的形状为（21，3），（21，3），（0，0），（33，3）
        left, right, face, body = extract_all(path_to_image=path) # 提取图像文件中的左右手，面部，姿态关键点，
        num_detected_hands = extract_hands(path_to_image=path) # 提取图像文件中的左右手的数量

        if FACE_REDUCTION is False: # 如果不忽略面部特征，则存储二维数组，形状是(帧数,398)=（21*2+21*2+124*2+33*2）
            if idx == 0:
                if (
                        left.shape != (0,) 
                        and right.shape != (0,)
                        and face.shape != (0,)
                        and body.shape != (0,)
                ):# 检查左右手，面部，姿态关键点是否为空，不为空存储
                    final.append([left, right, face, body]) # 存储左右手，面部，姿态关键点
                    final_input.append(
                        np.concatenate((left, right, face, body), axis=None) # 将左右手，面部，姿态关键点连接起来，即将二维坐标转换为一维坐标 [[1, 2], [3, 4]] -> [1, 2, 3, 4]
                    ) 
                else:
                    if left.shape == (0,): # 如果左手关键点为空，则将左手关键点设置为全为nan的数组
                        if num_detected_hands == 2:
                            left = np.full((21, 2), np.nan) # 确实检测到两只手，但是该帧图像中左手关键点没有被正确检测到，则将左手关键点设置为全为nan的数组，方便之后的插值
                        else:
                            left = np.zeros(shape=(21, 2), dtype=float)  # 检测一只手，但左手没有动作，将左手关键点设置为全为0的数组
                    if right.shape == (0,):
                        if num_detected_hands == 2:
                            right = np.full((21, 2), np.nan)
                        else:
                            right = np.zeros(shape=(21, 2), dtype=float)
                    if face.shape == (0,):
                        face = np.zeros(shape=(124, 2), dtype=float) #面部没有动作，将面部关键点设置为全为0的数组
                    if body.shape == (0,):
                        body = np.zeros(shape=(33, 2), dtype=float) # 姿态没有动作，将姿态关键点设置为全为0的数组
                        
                    final.append([left, right, face, body]) # 存储左右手，面部，姿态关键点
                    final_input.append(
                        np.concatenate((left, right, face, body), axis=None) # 将左右手，面部，姿态关键点连接起来，即将二维坐标转换为一维坐标 [[1, 2], [3, 4]] -> [1, 2, 3, 4]
                    )
            #      
            else:
                # 判断后续帧是否检测到 两只手，一只手，或者没有手，是否检测到脸部和姿态
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
        # 如果忽略面部特征，则存储三维坐标，[[x1,y1,z1],[x2,y2,z2],...] 如果全特征都不为空，则存储为一个final列表，否则将空的特征设置为全为0或者nan的数组在存储为final列表。
        else: 
# 减少面部检测时， final_input的形状是(帧数,258)=（21*3+21*3+33*4）
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
    final_input = np.array(final_input, dtype=float) # 形状是

    # 以下是对final_input进行处理，包括缺失值填充
    if impute_data is True:
        final_input = impute(final_input) # final_input是一个二维数组，形状是(帧数，特征数)=（帧数，398 or 258）
        
    # 帧差分，作用在于
    if frame_diff is True:
        temp = []
        diff = get_frame_diff(final_input)
        for row in range(final_input.shape[0]):
            temp.append(np.append(final_input[row], diff[row]))
        final_input = np.array(temp, dtype=float)

    # add_center和add_speed_center是对final_input进行处理，包括将面部特征和姿态特征分别放在左右手的两侧，或者将面部特征和姿态特征放在中间
    if add_center is True:
        temp = []
        left, right = None, None
        if FACE_REDUCTION is False:
            left, right = center_face(final_input) # left=[[x_1center,y_1center],...] right=[[x_2center,y_2center],...]
        else:
            left, right = center(final_input) # left=[[x_1center,y_1center,z_1center],...] right=[[x_2center,y_2center,z_2center],...
        for row in range(final_input.shape[0]):
            temp.append(
                np.concatenate((final_input[row], left[row], right[row]), axis=0)
            )
        final_input = np.array(temp, dtype=float)
    if add_speed_center is True: # 每一帧图片头部和手部中心点的速度
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
    if add_speed_marker is True:# 每一帧图片头部和手部marker点的速度
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
