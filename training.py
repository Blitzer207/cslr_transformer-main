"""
This file contains all the settings for training.
"""
import argparse
import numpy as np
import os
from utils.trainer import Trainer
from models.aslr import Transformer as T
import torch
from datetime import datetime
import time
from utils import plots
from utils.datasets import VideoDataset

parser = argparse.ArgumentParser(description="Arguments for the training script")
parser.add_argument(
    "--multisigner",
    type=bool,
    help="Boolean to specify if we use multisigner data or signer independent data.",
    default=True,
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--epochs",
    type=int,
    default=200,
    help="Integer to specify number of epochs",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=4,
    help="Integer to specify training batch size",
)
parser.add_argument(
    "--seed",
    type=int,
    help="Integer for reproducibility",
    default=2022
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda",
    help="String to specify device to train on"
)
parser.add_argument(
    "--optimizer",
    type=str,
    help="String to specify which optimizer should be used",
    default="sgd"
)
parser.add_argument(
    "--evaluation_frequency",
    type=int,
    default=5,
    help="After how many percentage of total number of epoch we should evaluate on the dev set",
)
parser.add_argument(
    "--evaluate_during_training",
    type=bool,
    help="Boolean to specify if we evaluate while training. For this, Please give a evaluation_frequency",
    default=True,
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--save_per_epoch",
    type=bool,
    help="Boolean to specify if we save the model after each epoch",
    default=True,
    action=argparse.BooleanOptionalAction,
)
args = parser.parse_args()
if args.device.lower() != "cuda" and args.device.lower() != "cpu":
    raise ValueError("Please provide normal device")
if args.optimizer.lower() != "sgd" and args.optimizer.lower() != "adam":
    raise ValueError("Please provide valid optimizer")
if 0 > args.evaluation_frequency > 100:
    raise ValueError("Please provide a valid evaluation_frequency. between 1 and 100")
if __name__ == "__main__":
    np.random.RandomState(args.seed)
    torch.manual_seed(seed=args.seed)
    to_use = "multi" if args.multisigner else "si5"
    trg_vocab = 1297 if args.multisigner else 1137
    PADDING = 1296 if args.multisigner else 1136
    training_features = np.load(to_use + "_training_features.npy", allow_pickle=True)
    training_label = np.load(to_use + "_training_label.npy", allow_pickle=True)
    evaluation_features = np.load(to_use + "_validation_features.npy", allow_pickle=True)
    evaluation_label = np.load(to_use + "_validation_label.npy", allow_pickle=True)
    torch.manual_seed(seed=args.seed)
    train_data = VideoDataset(feature_tab=training_features, label=training_label)
    val_data = VideoDataset(feature_tab=evaluation_features, label=evaluation_label)
    model = T(
        device=torch.device(str(args.device).lower()),
        embed_dim=512,
        seq_length_en=300,
        seq_length_dec=300,
        src_vocab_size=300,
        target_vocab_size=trg_vocab,
        num_layers=8,
        expansion_factor=4,
        n_heads=8,
        dropout_rate=0.2,
    )
    if torch.cuda.device_count() > 1 and args.device.lower() == "cuda":
        # You may need to adapt this to your GPU environment
        dev_arr = np.arange(torch.cuda.device_count())
        visible_device=""
        for dev in dev_arr:
            visible_device = visible_device + str(dev) + ","
        visible_device = visible_device[:-1]
        # os.environ["CUDA_VISIBLE_DEVICES"] = visible_device
        print("We will be training on {0} GPUs with IDs" + visible_device)
        model = torch.nn.DataParallel(model, device_ids=[0,1])
    vocab = np.load("full_gloss_" + to_use + ".npy", allow_pickle=True)
    vocab = dict(enumerate(vocab.flatten()))
    vocab = vocab[0]
    print("Training begins with following arguments ")
    print("Dataset: " + to_use + ", number of epochs: ", args.epochs, ", batch size: ", args.batch_size, ", device: "
          + args.device + ", seed: ", args.seed, ", optimizer: " + args.optimizer, ", evaluation frequency: "
          , args.evaluation_frequency)
    trainer = Trainer(
        model=model,
        training_set=train_data,
        validation_set=val_data,
        epochs=args.epochs,
        loss="cross entropy",
        optimizer=str(args.optimizer).lower(),
        evaluate_during_training=args.evaluate_during_training,
        learning_rate=0.09,
        batch_size=args.batch_size,
        device=torch.device(str(args.device).lower()),
        save_epoch=args.save_per_epoch,
        padding_character=PADDING
    )
    trainer.model_infos()
    now = datetime.now()
    print("Beginning time : ", now)
    trainer.train(vocab=vocab, remove_pad=True)
    now = datetime.now()
    print("Ending time : ", now)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    timestr = str("results_full_training" + timestr)
    os.mkdir(timestr)
    print("Training done. Plotting results and saving in directory" + timestr)

    x = trainer.eval_epoch
    training_loss = trainer.total_loss
    np.save("training_loss", arr=np.array(training_loss), allow_pickle=True)
    plots.plot_learning_curve(tab=training_loss, name=str(timestr) + "/learning_loss", log_scale=False)

    evaluation_loss = trainer.total_eval_loss
    np.save("evaluation_loss", arr=np.array(evaluation_loss), allow_pickle=True)
    plots.plot_evaluation_curve(x=x, tab=evaluation_loss, name=str(timestr) + "/evaluation_loss")

    wer_evolution = trainer.wer
    np.save("wer_evolution", arr=np.array(wer_evolution), allow_pickle=True)
    plots.wer_evolution(x=x, tab=wer_evolution, name=str(timestr) + "/wer_evolution")

    l_evolution = trainer.lr_curve
    np.save("learning_rate_evolution", arr=np.array(l_evolution), allow_pickle=True)
    plots.plot_learning_rate_evolution(tab=l_evolution, name=str(timestr) + "/lr_evolution")

    print("Saving the model")
    if torch.cuda.device_count() > 1:
        torch.save(trainer.model.module.state_dict(), str(timestr) + "/transformer.pt")
    else:
        torch.save(trainer.model.state_dict(), str(timestr) + "/transformer.pt")
