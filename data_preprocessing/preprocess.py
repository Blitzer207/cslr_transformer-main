"""
This file contains the preprocessing methods.
The only one used in the research was image resizing.

"""

from multiprocessing import Pool
from tqdm import tqdm
import glob
import cv2
import re
import os


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
