from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2, l1
import matplotlib.pyplot as plt
import numpy as np


class Interpreter:
    """
    """
    def __init__(self, batch_size, image_shape, epochs=50):
        """
        Get raw image

        Args:
        ---------
            batch_size:
        """
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )
        self.test_datagen = ImageDataGenerator(
            rescale=1/255
        )

    def split_data(self):
        """
        Splits data into train, test and validation data, according
        to data path images

        Return:
        ---------
            train_images: train images
            test_images: test images
            validation_images: validation images
        """
        train_images = self.train_datagen.flow_from_directory(
            directory='./Data/train',
            batch_size=self.batch_size,
            target_size=(150, 150),
            class_mode='binary'
        )

        test_images = self.test_datagen.flow_from_directory(
            directory='./Data/test',
            batch_size=self.batch_size,
            target_size=(150, 150),
            class_mode='binary'
        )

        validation_images = self.test_datagen.flow_from_directory(
            directory='./Data/valid',
            batch_size=self.batch_size,
            target_size=(150, 150),
            class_mode='binary'
        )

        return train_images, test_images, validation_images

    def train_model(self, train_images, test_images, validation_images):
        """
        Train simple CNN model
        # TODO: Evaluates VGG16 model

        Args:
        ---------
            'image': image to apply processing step
        """
        model = Sequential()
        model.add(
            Conv2D(
                32,
                (3, 3),
                input_shape=self.image_shape
                )
        )
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(
            64,
            bias_regularizer=l2(0.01),
            activation_regularizer=l2(0.02)
            ))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.compile(
            loss='binary_crossentropy',
            optimizer='rmsprop',
            metrics=['accuracy']
        )

        model_out = model.fit_generator(
                train_images,
                steps_per_epoch=2000 // self.batch_size,
                epochs=self.epochs,
                validation_data=validation_images,
                validation_steps=800 // self.batch_size
        )

        model.save_weights('model_2.h5')

        N = np.arange(0, self.epochs)
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(N, model_out.history["loss"], label="train_loss")
        plt.plot(N, model_out.history["val_loss"], label="val_loss")
        plt.plot(N, model_out.history["acc"], label="train_acc")
        plt.plot(N, model_out.history["val_acc"], label="val_acc")
        plt.title("Training Loss and Accuracy on Dataset")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        plt.savefig('model1')
