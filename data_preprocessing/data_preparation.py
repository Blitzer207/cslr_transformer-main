import os
import requests
import pandas as pd
import os.path
import numpy as np


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
                for_analysis = data_file[['signer', 'annotation']]
                for_analysis.to_pickle("signer_test.pkl")
        else:
            raise ValueError("The provided split does not exist. Please give test, train or dev as dataset_split "
                             "parameter")
        folder = data_file["folder"]
        signer_distribution = data_file["signer"]
        signer_distribution = np.array(signer_distribution, dtype=str)
        for i in range(len(folder)):
            value = (
                    self.dataset_dir
                    + phoenix
                    + "/features/fullFrame-210x260px/"
                    + dataset_split
                    + "/"
                    + str(folder[i]).split("*")[0].replace("\\", "/")
            )
            value.replace('"', "/")
            if os.path.isdir(value) is True:
                number_of_frames = os.scandir(value[:-1])
                count = 0
                for file in number_of_frames:
                    if file.is_file():
                        count += 1
                label = data_file["annotation"][i]
                self.samples_info.append((value, count, label))
            else:
                print(
                    "Directory invalid. Please make sure you are running this on Linux, otherwise you would need to "
                    "tailor the code for your system "
                )
                break
        ar = np.array(self.samples_info, dtype=str)
        if multisigner is True:
            np.save(os.getcwd() + "/" + dataset_split + "_info_multi", ar)
            np.save(os.getcwd() + "/" + dataset_split + "_signer_distribution_multi", signer_distribution)
        else:
            np.save(os.getcwd() + "/" + dataset_split + "_info_si5", ar)
            np.save(os.getcwd() + "/" + dataset_split + "_signer_distribution_si5", signer_distribution)
        self.samples_info = []
        return ar
