"""
This script contains different functions to plot the model's performances after training.
"""
import matplotlib.pyplot as plt
from matplotlib import pyplot
import numpy as np
import seaborn as sns
import math


def plot_frame_diff(input_array: np.ndarray, full_body: bool):
    """
    Plots the difference in input between frames based on the L2 Norm
    :param full_body: specify if the difference is for the full body or just for the hands.
    :param input_array: feature ND array.
    :return: plot
    """

    if full_body is False:
        for row in input_array:
            row[0:126] = 0.0
    fig = plt.figure(
        figsize=[6.4, 4.8], facecolor="skyblue", edgecolor="black", dpi=100
    )
    x = np.arange(start=0, stop=input_array.shape[0], step=1, dtype=int)
    y = np.zeros(shape=(input_array.shape[0]), dtype=float)
    for i in range(input_array.shape[0]):
        if i == 0 or i == input_array.shape[0] - 1:
            y[i] = 0.0
        else:
            next = i + 1
            y[i] = np.linalg.norm(input_array[next] - input_array[i])
    plt.step(x, y, color="black")
    plt.xlabel("Time / Frames")
    plt.ylabel("L2 Norm")
    if full_body is False:
        plt.title("Difference between frames (just hands)")
    else:
        plt.title("Difference between frames (full body)")
    plt.show()
    plt.clf()


def plot_learning_rate_evolution(tab: list, name: str):
    """
    Plots the curve of the evolution of the learning rate during training.
    :param tab: list of losses for each epoch.
    :param name: name to save the plot
    :return:
    """
    y = np.array(tab)
    x = np.arange(start=0, stop=y.shape[0], step=1, dtype=int)
    sns.set_theme(style="darkgrid")
    sns.lineplot(x, y)
    plt.xlabel("Update")
    plt.ylabel("Learning rate")
    plt.title("Learning rate evolution")
    plt.savefig(name + ".png")
    plt.clf()

    return


def plot_all(train: list, evall: list, wer: list, name: str):
    """
    Plots the learning curve during training
    :param train: training loss.
    :param evall: evaluation loss.
    :param wer: wer evolution.
    :param name: name to save the plot
    :return:
    """

    fig, axes = pyplot.subplots(3)

    y_train = np.array(train)
    x_train = np.arange(start=0, stop=y_train.shape[0], step=1, dtype=int)

    y_eval = np.array(evall)
    x_eval = np.arange(start=0, stop=y_eval.shape[0], step=1, dtype=int)

    y_wer = np.array(wer)
    x_wer = np.arange(start=0, stop=y_wer.shape[0], step=1, dtype=int)

    axes[0].plot(x_train, y_train, color="green", label="Training learning curve")
    axes[1].plot(x_eval, y_eval, color="blue", label="Evaluation learning curve")
    axes[2].plot(x_wer, y_wer, color="red", label="WER evolution")

    # position at which legend to be added
    axes[0].legend(loc="best")
    axes[1].legend(loc="best")
    axes[2].legend(loc="best")

    plt.subplots_adjust(
        left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4
    )
    plt.savefig(name + ".png")
    plt.clf()

    return


def plot_learning_curve(tab: list, name: str, log_scale: bool = True):
    """
    Plots the learning curve during training
    :param log_scale: Specify if we print progress on log scale.
    :param tab: list of losses for each epoch.
    :param name: name to save the plot
    :return:
    """
    com = ""
    y = np.array(tab)
    x = np.arange(start=0, stop=y.shape[0], step=1, dtype=int)
    sns.set_theme(style="darkgrid")
    if log_scale is True:
        com = "Epochs (on log scale)"
        sns.lineplot(x, y).set(xscale="log")
    else:
        sns.lineplot(x, y)
        com = "Epochs"
    plt.xlabel(com)
    plt.ylabel("Loss")
    plt.title("Training Learning Curve")
    plt.savefig(name + ".png")
    plt.clf()

    return


def wer_evolution(x: list, tab: list, name: str):
    """
    Plots the wer evolution during training
    :param x: epoch on which evaluation occures.
    :param tab: list of wer values for each evaluation.
    :param name: name to save the plot
    :return:
    """
    y = np.array(tab)
    x = np.array(x)
    sns.set_theme(style="darkgrid")
    sns.lineplot(x, y)
    plt.xlabel("Epochs")
    plt.ylabel("WER")
    plt.title("WER evolution during training")
    plt.savefig(name + ".png")
    plt.clf()

    return


def plot_evaluation_curve(x: list, tab: list, name: str):
    """
    Plots the evaluation loss during training
    :param x: epoch on which evaluation occures.
    :param tab: list of losses for each evaluation.
    :param name: name to save the plot
    :return:
    """
    y = np.array(tab)
    x = np.array(x)
    sns.set_theme(style="darkgrid")
    sns.lineplot(x, y)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Evaluation Learning Curve")
    plt.savefig(name + ".png")
    plt.clf()

    return
