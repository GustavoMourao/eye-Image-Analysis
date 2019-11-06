from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2, l1
import matplotlib.pyplot as plt
import numpy as np
from keras.optimizers import SGD
import keras
from WindowsOpt import WindowOptimizer, initialize_window_setting
from keras import regularizers
from keras.layers import GlobalAveragePooling2D
from keras.models import Model


class Interpreter:
    """
    """
    def __init__(self, batch_size, image_shape, epochs=2):
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

    def windown_optimizer(self, train_images, test_images, validation_images):
        """
        """
        # For multi-channel WSOlayer
        nch_window = 2
        act_window = "sigmoid"
        upbound_window = 255.0
        init_windows = "ich_init"

        optimizer = SGD(lr=0.0001, decay=0, momentum=0.9, nesterov=True)
        input_shape = self.image_shape
        input_tensor = keras.layers.Input(shape=input_shape, name="input")

        # Define a window setting optimization layer
        x = WindowOptimizer(
            nch_window=nch_window,
            act_window=act_window,
            upbound_window=upbound_window,
            kernel_initializer="he_normal",
            kernel_regularizer=regularizers.l2(0.5 * 1e-5)
            )(input_tensor)

        # ... add some your layer here
        x = Conv2D(
            32,
            (3, 3),
            activation=None,
            padding="same",
            name="conv1"
        )(x)
        x = Activation("relu", name="conv1_relu")(x)
        x = MaxPooling2D((7, 7), strides=(3, 3), name="pool1")(x)
        x = Conv2D(
            256,
            (3, 3),
            activation=None,
            padding="same",
            name="conv2"
        )(x)
        x = Activation("relu", name="conv2_relu")(x)
        x = MaxPooling2D((7, 7), strides=(3, 3), name="pool2")(x)
        x = GlobalAveragePooling2D(name="gap")(x)
        outputs = Dense(1, activation='sigmoid', name="fc")(x)

        model = Model(inputs=input_tensor, outputs=outputs, name="main_model")

        # Initialize parameters of window setting opt module
        model = initialize_window_setting(
            model,
            act_window=act_window,
            init_windows=init_windows,
            upbound_window=upbound_window
        )

        # Compile and check parameters
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=["accuracy"]
        )
        model.summary()

        model_out = model.fit_generator(
            train_images,
            steps_per_epoch=2000 // self.batch_size,
            epochs=self.epochs,
            validation_data=validation_images,
            validation_steps=800 // self.batch_size
        )

        # TODO: Put this in another method!
        model.save_weights('model_2.h5')

        N = np.arange(0, self.epochs)
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
            # bias_regularizer=l2(0.01),
            activity_regularizer=l2(0.02)
            ))
        model.add(Activation('relu'))
        # model.add(Dropout(0.2))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.compile(
            loss='binary_crossentropy',
            optimizer='rmsprop',
            metrics=['accuracy']
        )
        model.summary()

        model_out = model.fit_generator(
            train_images,
            steps_per_epoch=2000 // self.batch_size,
            epochs=self.epochs,
            validation_data=validation_images,
            validation_steps=800 // self.batch_size
        )

        # TODO: Put this in another method!
        model.save_weights('model_2.h5')

        N = np.arange(0, self.epochs)
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
