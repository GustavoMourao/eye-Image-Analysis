import matplotlib.pyplot as plt
import numpy as np


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
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(N, model_out.history["loss"], label="train_loss")
        plt.plot(N, model_out.history["val_loss"], label="val_loss")
        plt.plot(N, model_out.history["accuracy"], label="train_acc")
        plt.plot(N, model_out.history["val_accuracy"], label="val_acc")
        plt.title("Training Loss and Accuracy on Dataset")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        plt.savefig('model1')
        plt.plot()
