"""
This file contains methods for double-checking the number of extracted hands in the video.
Mainly used to know and confirm missing landmarks in the hand region.
Library from @https://github.com/cvzone/cvzone
"""
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np

'''
`detectionCon`参数是HandDetector在进行手部检测时所使用的置信度阈值。这个参数的范围通常在0到1之间，表示检测模型对于检测结果的置信度。如果检测结果的置信度高于这个阈值，那么模型就会认为该检测结果是有效的。

具体来说，如果将`detectionCon`设置得较高，那么只有当模型对检测结果非常有信心时，才会返回这个结果，这可能会降低检测的假阳性率（即将不存在的手部错误地检测为存在），但也可能会增加假阴性率（即将存在的手部错误地检测为不存在）。

相反，如果将`detectionCon`设置得较低，那么即使模型对检测结果的信心不高，也可能会返回这个结果，这可能会增加假阳性率，但也可能会降低假阴性率。

至于`detectionCon`应该设置为多少，这需要根据你的具体应用和需求来决定。一般来说，如果你的应用对假阳性非常敏感，那么你可能需要将`detectionCon`设置得较高；如果你的应用对假阴性非常敏感，那么你可能需要将`detectionCon`设置得较低。

然而，实际的最佳值通常需要通过实验来确定。你可以试着在一些代表性的图像或视频上运行你的程序，并调整`detectionCon`的值，看看哪个值能够得到最好的结果。
'''
detector = HandDetector(detectionCon=0.5, maxHands=2)  # Create a hand detector，


def extract_hands(
    path_to_image: str, show: bool = True #接收两个参数：图像的路径和一个布尔值决定是否显示处理后的图像。
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
    hands, image = detector.findHands(image) # Detect hands in the image,返回一个包含检测到的手部信息的列表，以及在图像上绘制了检测结果的图像。
    if hands: #如果检测到了手部，那么就会执行下面的代码。
        # Hand 1
        hand1 = hands[0] #获取第一个检测到的手部。
        lmList1 = hand1["lmList"] #获取手部的21个关键点的坐标。
        bbox1 = hand1["bbox"] #获取手部的包围框。
        centerPoint1 = hand1["center"] #获取手部的中心点。
        handType1 = hand1["type"] #获取手部的类型，即左手还是右手。

        fingers1 = detector.fingersUp(hand1) #获取手部的五指状态。
        n_detected_hands = 1
        if len(hands) == 2: #如果检测到了两只手部，那么就会执行下面的代码。
            # Hand 2
            hand2 = hands[1] #获取第二个检测到的手部。
            lmList2 = hand2["lmList"]
            bbox2 = hand2["bbox"]
            centerPoint2 = hand2["center"]
            handType2 = hand2["type"]

            fingers2 = detector.fingersUp(hand2)
            n_detected_hands = 2

    if show is True:
        cv2.imshow("Image", image)
        cv2.waitKey(3000)

    return n_detected_hands #返回检测到的手部数量。
