import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix


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
#         model_info['accuracy'] = model_out.history["accuracy"]
#         model_info['val_accuracy'] = model_out.history["val_accuracy"]
        model_info['acc'] = model_out.history["acc"]
        model_info['val_acc'] = model_out.history["val_acc"]

        # loss graphs
        fig1 = plt.figure(figsize=(12, 12))
        ax = fig1.add_subplot(1, 1, 1)
        sns.lineplot(
            x=model_info['N'],
            y=model_info['loss'],
            ax=ax,
            data=model_info,
            legend='full'
        )

        sns.lineplot(
            x=model_info['N'],
            y=model_info['val_loss'],
            ax=ax,
            data=model_info,
            legend='full'
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
        fig1.legend(
            labels=['loss', 'val_loss']
        )
        plt.savefig('model1_loss')
        # plt.show()

        # accuracy graphs
        fig1 = plt.figure(figsize=(12, 12))
        ax = fig1.add_subplot(1, 1, 1)
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
        fig1.legend(
            labels=['accuracy', 'val_accuracy']
        )

#         plt.savefig('model1_accuracy')
        plt.show()

    def show_confusion_matrix(self, y_true, y_pred, classes):
        """
        """
        self.__plot_confusion_matrix(
            y_true,
            y_pred,
            classes,
            normalize=True
        )

        return 0

    def __plot_confusion_matrix(
        self,
        y_true,
        y_pred,
        classes,
        normalize=False,
        title=None,
        cmap=plt.cm.Blues
    ):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

        cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        classes = classes[unique_labels(y_true, y_pred)]
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        plt.show()
        return ax
