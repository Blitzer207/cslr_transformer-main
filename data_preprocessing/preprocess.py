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

# 多进程并行执行函数process_func
def run_mp_cmd(processes, process_func, process_args): #函数创建一个进程池，并并行执行函数process_func。tqdm用于显示进度条。
    # processes: 进程数，process_func: 函数，process_args: 函数参数，Pool()返回一个进程池对象
    with Pool(processes) as p: # Pool()函数创建一个进程池，参数processes为进程池中的进程数。with语句用于创建一个上下文管理器，用于自动关闭进程池。
        
        # imap()函数返回一个迭代器，每次迭代返回一个函数process_func的返回值，total参数用于指定迭代次数。
        outputs = list( # list()函数将迭代器转换为列表
            tqdm(p.imap(process_func, process_args), total=len(process_args))
        )
    return outputs


def run_cmd(func, args):
    return func(args)


def resize_img(img_path: str, size: str = "300x300px"): # 读取一张图片，将其缩放到指定大小
    """
    Resizing the image before preprocessing
    :param img_path: Path to the image
    :param size: New size of the image
    :return: new image
    """
    # 因为Size是字符串，所以需要用正则表达式匹配出Size中的数字，并将其转换为cv2可接受的元组整数格式
    size = tuple(int(res) for res in re.findall("\d+", size)) 
    img = cv2.imread(img_path)
    img = cv2.resize(img, size, interpolation=cv2.INTER_LANCZOS4) # 采用LANCZOS4插值算法进行缩放
    # cv2.INTER_LANCZOS4: Lanczos插值，其插值核心是sinc函数，可以有效避免图像模糊，但是会导致图像的锐化。
    return img #返回缩放后的图片


def resize_dataset(video_idx: int, size: str, info_dict: dict):
    """
    Resize the whole dataset.
    :param video_idx: video id
    :param size: size to use for resizing
    :param info_dict: Input dictionary with all informations.
    :return:
    """
    info = info_dict[video_idx]
    # info_dict['prefix']代表数据集的路径，info['folder']代表数据集中的文件夹
    img_list = glob.glob(f"{info_dict['prefix']}/{info['folder']}") # glob.glob()函数用于查找符合特定规则的文件路径名
    for img_path in img_list:
        rs_img = resize_img(img_path, size=size)
        rs_img_path = img_path.replace("210x260px", size) #原始路径 /features/fullFrame-210x260px/ 的210x260px替换为 300x300px
        rs_img_dir = os.path.dirname(rs_img_path)
        if not os.path.exists(rs_img_dir):
            os.makedirs(rs_img_dir)
            cv2.imwrite(rs_img_path, rs_img)
        else:
            cv2.imwrite(rs_img_path, rs_img)
