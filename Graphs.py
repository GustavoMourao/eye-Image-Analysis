import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


class Graphs:
    """
    Class responsible for plot analysis graphs.
    """
    def __init__(self, name="graphs"):
        """
        Initialize class.
        """
        self.name = name

    def show_train_validation(self, epochs, model_out):
        """
        Args:
        ---------
            epochs: number of train epochs
            model_out: object of trained model

        Return:
        ---------
            graph with train/validation loss and accuracy
        """
        N = np.arange(0, epochs)
        model_info = pd.DataFrame()
        model_info['N'] = N
        model_info['loss'] = model_out.history["loss"]
        model_info['val_loss'] = model_out.history["val_loss"]
        model_info['accuracy'] = model_out.history["accuracy"]
        model_info['val_accuracy'] = model_out.history["val_accuracy"]

        # loss graphs
        sns.lineplot(
            x=model_info['N'],
            y=model_info['loss'],
            data=model_info
        )

        sns.lineplot(
            x=model_info['N'],
            y=model_info['val_loss'],
            data=model_info
        )

        plt.title(
            "Loss variation",
            fontsize=20
        )
        plt.xlabel(
            "Epochs",
            fontsize=15
        )
        plt.ylabel(
            "Loss",
            fontsize=15
        )

        plt.savefig('model1_loss')
        plt.show()

        # accuracy graphs
        sns.lineplot(
            x=model_info['N'],
            y=model_info['accuracy'],
            data=model_info
        )

        sns.lineplot(
            x=model_info['N'],
            y=model_info['val_accuracy'],
            data=model_info
        )

        plt.title(
            "Accuracy variation",
            fontsize=20
        )
        plt.xlabel(
            "Epochs",
            fontsize=15
        )
        plt.ylabel(
            "Accuracy",
            fontsize=15
        )

        plt.savefig('model1_accuracy')
        plt.show()
        # TODO: named legends
