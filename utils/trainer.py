"""
This file contains the trainer designed for this work
"""
import torch
import time
import copy
import os
import gc
import torch.nn as F
import numpy as np
from models.aslr import Transformer
from utils.datasets import VideoDataset
from torch.utils.data import DataLoader
from torchsummary import summary
from utils.wer import compute_wer, seq_to_text
from tqdm import tqdm
from utils.free_memory import free_gpu_cache
from datetime import datetime


class Trainer:
    """
    Creates a trainer
    """

    def __init__(
            self,
            model: Transformer,
            training_set: VideoDataset,
            validation_set: VideoDataset,
            epochs: int,
            loss: str,
            optimizer: str,
            evaluate_during_training: bool,
            learning_rate,
            batch_size,
            device,
            save_epoch,
            padding_character: int,
            path_to_model_checkpoint: str = "",
            load_from_checkpoint: bool = False,
            eval_freq: int = 0,
    ):
        """

        :param model:model to be trained.
        :param training_set: training set to be used.
        :param validation_set: validation set to be used.
        :param epochs: number of epochs to train for
        :param loss: loss to use (for the moment just cross entropy).
        :param optimizer: type of optimizer to use
        :param evaluate_during_training: If we check evaluation loss during training.
        :param learning_rate: learning rate to use.
        :param batch_size: batch size to use for data loading.
        """
        self.padding_character = padding_character
        self.path_to_model_checkpoint = path_to_model_checkpoint
        self.optimizer = optimizer
        self.total_loss = []
        self.total_eval_loss = []
        self.wer = []
        self.lr_curve = []
        self.eval_epoch = []
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.evaluate_during_training = evaluate_during_training
        self.loss = loss
        self.epochs = epochs
        self.validation_set = validation_set
        self.training_set = training_set
        self.model = model
        self.eval_freq = eval_freq
        self.device = device
        self.criterion = None
        self.optimizer_ = None
        if self.evaluate_during_training is True:
            if int(self.eval_freq) == int(0):
                self.eval_freq = int(self.epochs * 0.05)
        else:
            self.eval_freq = self.epochs
        if self.loss.lower() == "ctc":
            self.criterion = F.CTCLoss()
        elif self.loss.lower() == "cross entropy":
            self.criterion = F.CrossEntropyLoss()
        else:
            raise ValueError(
                "Please give the name of a valid loss function."
                "For the moment we support only CTC loss and Cross Entropy loss."
            )
        if self.optimizer.lower() == "adam":
            self.optimizer_ = torch.optim.Adam(
                self.model.parameters(), lr=self.learning_rate
            )
        elif self.optimizer.lower() == "sgd":
            self.optimizer_ = torch.optim.SGD(
                self.model.parameters(), lr=self.learning_rate
            )
        else:
            raise ValueError(
                "Please give the name of a valid loss function."
                "For the moment we support only CTC loss and Cross Entropy loss."
            )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_, "min", patience=1, factor=0.7
        )
        self.save_epoch = save_epoch
        if load_from_checkpoint is True and self.path_to_model_checkpoint != "":
            self.model.load_state_dict(torch.load(self.path_to_model_checkpoint))
            print("Loading model checkpoint from " + self.path_to_model_checkpoint)
        self.model.to(self.device)

    def train(self, vocab, remove_pad: bool = True):
        """
        Trains the model
        :param vocab: Used vocabulary
        :param remove_pad: bool to specify if we remove padding when decoding
        :return: trained model and loss
        """
        self.model.train()
        self.lr_curve.append(self.learning_rate)
        train_set = DataLoader(
            dataset=self.training_set,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )
        print("Training the model for {0} epochs \n".format(self.epochs))
        for i in range(self.epochs):
            start_time = datetime.now().timestamp()
            free_gpu_cache()
            print("Training epoch {0} / {1}".format(i + 1, self.epochs))
            epochLoss = 0
            for src, trg in tqdm(train_set, desc="Training"):
                self.optimizer_.zero_grad()
                src, trg = src.to(self.device), trg.to(self.device)
                predictions = self.model(src, trg)
                src = src.detach()
                del src
                predictions = predictions.view(-1, predictions.size(-1))
                trg = trg.view(-1)
                loss = self.criterion(predictions, trg)
                trg = trg.detach()
                del trg
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer_.step()
                epochLoss += float(loss) / self.batch_size
                free_gpu_cache()
            self.total_loss.append(epochLoss)
            final = datetime.now().timestamp() - start_time
            print("Runtime from this epoch is {0} minutes".format(round(final/60, 2)))
            print("loss from epoch {0} is {1}".format(i + 1, epochLoss))
            if self.save_epoch is True:
                timestr = time.strftime("%Y%m%d-%H%M%S")
                timestr = str("temp_model_" + timestr + "_epoch_" + str(i + 1))
                os.mkdir(timestr)
                print("Saving the model in directory " + timestr)
                if torch.cuda.device_count() > 1:
                    torch.save(self.model.module.state_dict(), str(timestr) + "/transformer.pt")
                else:
                    torch.save(self.model.state_dict(), str(timestr) + "/transformer.pt")
            if self.evaluate_during_training is True:
                free_gpu_cache()
                if (i + 1) % self.eval_freq == 0:
                    self.eval_epoch.append(i + 1)
                    l, w = self.evaluate(vocab=vocab, remove_pad=remove_pad)
                    temp = copy.deepcopy(self.learning_rate)
                    free_gpu_cache()
                    self.scheduler.step(l)
                    if temp != self.scheduler.optimizer.param_groups[0]["lr"]:
                        self.lr_curve.append(
                            self.scheduler.optimizer.param_groups[0]["lr"]
                        )
                        print(
                            "New learning rate is: "
                            + str(self.scheduler.optimizer.param_groups[0]["lr"])
                        )
                    print(
                        "Evaluation occured on this epoch. Current loss: {0}, current WER: {1}".format(
                            l, w
                        )
                    )
                    print("Best model so far is from epoch {0} with a wer of {1}".format((np.argmin(self.wer) + 1) * self.eval_freq,
                                                                                  round(np.min(self.wer) * 100, 2)))
                    if len(self.total_eval_loss) > 5:
                        if l > self.total_eval_loss[-1]:
                            print("Early stopping because of possible overfiting")
                            free_gpu_cache()
                            return
                free_gpu_cache()
                self.model.train()

    def evaluate(self, vocab, remove_pad: bool = True):
        """
        Evaluates the model
        :param vocab: Used vocabulary
        :param remove_pad: bool to specify if we remove padding when decoding
        :return: Evaluation loss and wer
        """
        soft = torch.nn.Softmax(dim=1)
        ground, pred = [], []
        loss = 0
        self.model.eval()
        free_gpu_cache()
        val_set = DataLoader(
            dataset=self.validation_set,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )
        with torch.no_grad():
            for src, trg in tqdm(val_set, "Evaluating"):
                free_gpu_cache()
                src, trg = src.to(self.device), trg.to(self.device)
                predictions = self.model(src, trg)
                src = src.detach
                del src
                trg_wer, predictions_wer = copy.copy(trg), copy.copy(predictions)
                predictions = predictions.view(-1, predictions.size(-1))
                trg = trg.view(-1)
                loss += self.criterion(predictions, trg)
                trg = trg.detach()
                del trg
                free_gpu_cache()
                for i in range(trg_wer.shape[0]):
                    i_trg = trg_wer[i]
                    i_pred = predictions_wer[i]
                    i_pred = soft(i_pred)
                    i_pred = torch.argmax(i_pred, dim=-1)
                    seq_gt, seq_pred = seq_to_text(
                        ground_truth=i_trg,
                        prediction=i_pred,
                        vocab=vocab,
                        remove_padding=remove_pad,
                        padding_character=self.padding_character
                    )
                    ground.append(seq_gt)
                    pred.append(seq_pred)
                del trg_wer
                del predictions_wer
                free_gpu_cache()
        self.total_eval_loss.append(float(loss) / self.batch_size)
        w = compute_wer(ground_truth=ground, prediction=pred)
        self.wer.append(w)
        free_gpu_cache()
        return float(loss) / self.batch_size, w

    def model_infos(self):
        """
        Prints informations of a model
        :return:
        """
        return summary(model=self.model)
