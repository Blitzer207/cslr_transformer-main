import os
import requests
import pandas as pd
import os.path
import numpy as np

"""
data_file 包含 id，folder, signer, annotation 
实例：
id|folder|signer|annotation
01April_2010_Thursday_heute_default-1|01April_2010_Thursday_heute_default-1/1/*.png|Signer04|ICH OSTERN WETTER ZUFRIEDEN MITTAG TEMPERATUR  SUED WARM MEIN NICHT
01August_2011_Monday_heute_default-6|01August_2011_Monday_heute_default-6/1/*.png|Signer07|VIELLEICHT THUERINGEN REGION AUCH BISSCHEN WOLKE BISSCHEN STARK WOLKE ABER SUED MAINZ FLUSS SUEDWEST VIEL NICHT-KEIN WOLKE VIEL SONNE

* 代表通配符，指的是该文件夹下的所有图片对应 一句话
self.samples_info.append((value, count, label)) # 将视频的路径，帧数，标签保存到samples_info中，代表这一个视频表达一个句子。


signer_test.pkl 保存的是测试集的签名者信息，用于后续分析 ['signer', 'annotation']

_info_multi.npy 
    * 数据类型是numpy数组 [(视频的路径，标签，帧数),(视频的路径，标签，帧数),...)]
    * 包含的是整个annotation/manual/train 或者dev或者test的所有信息，每一个数组元素是一个元组（视频的路径，标签，帧数）
    
_signer_distribution_multi 
    数据类型是numpy数组 [Signer04,Singer07,...]
    * 保存的是每个签名者的视频数量，用于后续分析
    
_info_si5 
    * 数据类型是numpy数组 [(视频的路径，标签，帧数),(视频的路径，标签，帧数),...)]
    * 保存的是整个annotation/manual/train 或者dev或者test的所有信息，每一个数组元素是一个元组（视频的路径，标签，帧数）
    
_signer_distribution_si5 
    * 数据类型是numpy数组 [Signer04,Singer07,...]
    * 保存的是每个签名者的视频数量，用于后续分析
PrepareFiles 类返回的是一个numpy数组，包含了整个annotation/manual/train 或者dev或者test的所有信息，一个数组元素是一个元组（视频的路径，标签，帧数）
视屏路径：/home/zhengxiawu/work/SLR/data/phoenix2014-release/manual/train/01April_2010_Thursday_heute_default-1/1/*.png
标签：ICH OSTERN WETTER ZUFRIEDEN MITTAG TEMPERATUR  SUED WARM MEIN NICHT 存储在路径：/home/zhengxiawu/work/SLR/data/phoenix2014-release/annotations/manual/train.corpus.csv
帧数：/home/zhengxiawu/work/SLR/data/phoenix2014-release/manual/train/01April_2010_Thursday_heute_default-1/1/*.png
因此，一个视频对应一个句子，一个句子对应一个标签，一个标签对应一个视频文件夹，一个视频文件夹对应多个图片，一个图片对应一帧。
"""
class PrepareFiles(object):
    """
    This class handles files preparation for data preprocessing for the signer independent dataset.
    It has method for reading the annotation files, getting the path of each video , the ground truth sentence,
    and the number of frames for the video. Additional statistics such as length of sentences, number of samples
    and more can be computed with the output data.
    """

    def __init__(
            self,
            dataset_dir: str = "temp/phoenix2014-release",
            data_available: bool = True,
    ):
        """
        Initial constructor
        :param dataset_dir: Path to the data set till the phoenix2014-release directory
        :param data_available: Boolean to specify if the dataset is available.
        """
        self.dataset_dir = dataset_dir
        self.data_available = data_available
        self.training_set_corpus = None
        self.evaluation_set_corpus = None
        self.test_set_corpus = None
        self.samples_info = [] 

        if self.data_available is False:
            """
            Download dataset and extract if it is not available (This part is linux based)
            """
            url = "https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/2016/phoenix-2014.v3.tar.gz"
            data = requests.get(url, allow_redirects=True)
            open("temp", "wb").write(data.content)
            ziel = os.getcwd() + "/temp/"
            command = "tar -xf archive.tar.gz -C " + ziel
            os.system(command)
            self.dataset_dir = os.getcwd() + "/temp/phoenix2014-release"
        if self.data_available is True and os.path.isdir(self.dataset_dir) is False:
            print(self.dataset_dir, self.data_available)
            raise ValueError(
                "Please either give path to the dataset or set the parameter data_available to False"
            )

    def prepare_set(
            self,
            dataset_split: str = "train",
            multisigner: bool = True
    ) -> np.ndarray:
        """
        Creates a numpy array containing information on the training, developing and test set. Reading .npy files is
        faster than .csv files.
        :param dataset_split: Split to use for preparation
        :param multisigner: Use multi signer dataset or not.
        :return: ND array containing path to video, Ground truth, number of frames of the video.
        
        """
        phoenix = "/phoenix-2014-multisigner" if multisigner else "/phoenix-2014-signerindependent-SI5"
        small = "" if multisigner else "SI5."
        self.training_set_corpus = (
                self.dataset_dir + phoenix + "/annotations/manual/train." + small + "corpus.csv"
        )
        self.evaluation_set_corpus = (
                self.dataset_dir + phoenix + "/annotations/manual/dev." + small + "corpus.csv"
        )
        self.test_set_corpus = (
                self.dataset_dir + phoenix + "/annotations/manual/test." + small + "corpus.csv"
        )
        if dataset_split == "train": 
            data_file = pd.read_csv(
                filepath_or_buffer=self.training_set_corpus, sep="|"
            )
        elif dataset_split == "dev":
            data_file = pd.read_csv(
                filepath_or_buffer=self.evaluation_set_corpus, sep="|"
            )
        elif dataset_split == "test":
            data_file = pd.read_csv(filepath_or_buffer=self.test_set_corpus, sep="|")
            if multisigner is True:
                for_analysis = data_file[['signer', 'annotation']] # for analysis, extract signer and annotation
                # 将读取的数据，按照原来的数据结构保存为pkl文件，方便后续分析，不用每次都读取csv文件。
                # 与 .npy 文件相比，pkl 文件的读取速度更快，但是文件体积更大。但是.npy文件仅仅用于保存numpy数组，而pkl文件可以保存任意数据类型。
                for_analysis.to_pickle("signer_test.pkl") # save as pickle file
        else:
            raise ValueError("The provided split does not exist. Please give test, train or dev as dataset_split "
                             "parameter")
        folder = data_file["folder"]
        signer_distribution = data_file["signer"] # 保存每个视频的签名者
        signer_distribution = np.array(signer_distribution, dtype=str)# 将签名者转换为numpy数组，维度为（1， 709）
        for i in range(len(folder)): # 读取每个视频的路径，帧数，标签
            value = (
                    self.dataset_dir # phoenix2014-release
                    + phoenix        # /phoenix-2014-multisigner or /phoenix-2014-signerindependent-SI5
                    + "/features/fullFrame-210x260px/" 
                    + dataset_split  # train, dev, test
                    + "/"
                    + str(folder[i]).split("*")[0].replace("\\", "/") #这里就是视频的名字，例如 01April_2010_Thursday_heute_default-1/1/*.png，replace是将路径中的\替换为/。
            )
            value.replace('"', "/") # 将路径中的"替换为/
            if os.path.isdir(value) is True: # 判断路径是否存在且为文件夹
                number_of_frames = os.scandir(value[:-1]) #获取 01April_2010_Thursday_heute_default-1/1/ 文件夹下的所有图片
                count = 0
                for file in number_of_frames: #遍历文件夹下的所有图片，计算图片的数量
                    if file.is_file():
                        count += 1
                label = data_file["annotation"][i]
                self.samples_info.append((value, count, label)) # 将视频的路径，帧数，标签保存到samples_info中，代表这一个视频表达一个句子。
            else:
                # 如果路径不存在或者不是文件夹，则打印错误信息，并退出循环，不再读取后续的视频。
                print(   
                    "Directory invalid. Please make sure you are running this on Linux, otherwise you would need to "
                    "tailor the code for your system " 
                )
                break
        #存储一些用于后续处理或分析的数据，特别是为了加快读取速度，数据被存储为 .npy 格式，这是一个用于存储大量数值数据的高效格式。
        ar = np.array(self.samples_info, dtype=str) # 将samples_info转换为numpy数组，维度为（1， 709， 3）
        if multisigner is True: # 保存为.npy文件
            np.save(os.getcwd() + "/" + dataset_split + "_info_multi", ar) 
            np.save(os.getcwd() + "/" + dataset_split + "_signer_distribution_multi", signer_distribution)
        else:
            np.save(os.getcwd() + "/" + dataset_split + "_info_si5", ar)
            np.save(os.getcwd() + "/" + dataset_split + "_signer_distribution_si5", signer_distribution)
        self.samples_info = []
        return ar
