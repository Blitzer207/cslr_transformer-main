"""
This file contains scripts for video reconstitution or video sampling.
"""
import cv2
import glob
from data_preprocessing.preprocess import resize_img


def video_reconstruction(file_name: str, folder_path: str, frame_rate: int = 25):
    """
    Reconstruct a video based on provided frames
    :param file_name: Name to give the video with extension avi.
    :param folder_path: Directory where the image frames are stored.
    :param frame_rate: Reconstruction frame rate.
    :return: video
    """
    img_array = []
    for filename in glob.glob(folder_path + "\\*.png"):
        img = cv2.imread(filename)
        img = resize_img(img_path=filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*"DIVX"), frame_rate, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


def video_decomposition(file_name: str, folder_path: str, ):
    """
     Decompose a video into single frames
     :param file_name: Path to the video file
     :param folder_path: Directory where the image frames will be stored.
     """
    cap = cv2.VideoCapture(file_name)
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False:
            break
        cv2.imwrite(folder_path + 'frame' + str(i) + '.png', frame)
        i += 1
    cap.release()
    cv2.destroyAllWindows()
