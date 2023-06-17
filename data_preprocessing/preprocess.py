"""
This file contains the preprocessing methods.
The only one used in the research was image resizing.

"""

from multiprocessing import Pool # 多进程, 用于加速
from tqdm import tqdm # 进度条, 用于显示进度
import glob # 用于查找符合特定规则的文件路径名
import cv2 # opencv, 用于图像处理 
import re # 正则表达式, 用于匹配字符串
import os # 用于操作系统相关的功能. 用于获取文件路径, 文件名等信息


def run_mp_cmd(processes, process_func, process_args): 
    with Pool(processes) as p:
        outputs = list(
            tqdm(p.imap(process_func, process_args), total=len(process_args))
        )
    return outputs


def run_cmd(func, args):
    return func(args)


def resize_img(img_path: str, size: str = "300x300px"):
    """
    Resizing the image before preprocessing
    :param img_path: Path to the image
    :param size: New size of the image
    :return: new image
    """
    size = tuple(int(res) for res in re.findall("\d+", size))
    img = cv2.imread(img_path)
    img = cv2.resize(img, size, interpolation=cv2.INTER_LANCZOS4)
    return img


def resize_dataset(video_idx: int, size: str, info_dict: dict):
    """
    Resize the whole dataset.
    :param video_idx: video id
    :param size: size to use for resizing
    :param info_dict: Input dictionary with all informations.
    :return:
    """
    info = info_dict[video_idx]
    img_list = glob.glob(f"{info_dict['prefix']}/{info['folder']}")
    for img_path in img_list:
        rs_img = resize_img(img_path, size=size)
        rs_img_path = img_path.replace("210x260px", size)
        rs_img_dir = os.path.dirname(rs_img_path)
        if not os.path.exists(rs_img_dir):
            os.makedirs(rs_img_dir)
            cv2.imwrite(rs_img_path, rs_img)
        else:
            cv2.imwrite(rs_img_path, rs_img)
