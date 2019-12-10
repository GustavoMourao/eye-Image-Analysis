from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2, l1
import matplotlib.pyplot as plt
import numpy as np
from keras.optimizers import SGD, Adadelta, Nadam
import keras
from keras import regularizers
from keras.layers import GlobalAveragePooling2D
from keras.models import Model
from WindowOpt.functions import *
from WindowOpt.WindowsOpt import *
from Graphs import Graphs
import efficientnet.keras as efn
from keras import backend as K
import tensorflow as tf
from keras.layers.normalization import BatchNormalization
from sklearn.metrics import accuracy_score


class Interpreter:
    """
    Class responsible to split raw data into train, validation and test.
    Besides of that, allows to train two different CNN topologies.
    """
    def __init__(
        self,
        batch_size,
        image_shape,
        epochs=40,
        target_size=(128, 128)
    ):
        """
        Get raw image

        Args:
        ---------
            batch_size:
            image_shape:
            epochs:
            target_size:
        """
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.epochs = epochs
        self.target_size = target_size
        self.train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )
        self.test_datagen = ImageDataGenerator(
            rescale=1/255
        )

    def get_info_images(self, imagepath):
        """
        """
        return self.train_datagen.flow_from_directory(
            directory=imagepath,
            batch_size=self.batch_size,
            target_size=(225, 225),
            class_mode='binary',
            color_mode='grayscale'
        )

    def split_data(self):
        """
        Splits data into train, test and validation data, according
        to data path images.

        Return:
        ---------
            train_images: train images
            test_images: test images
            validation_images: validation images
        """
        train_images = self.train_datagen.flow_from_directory(
            directory='./Data/train',
            batch_size=self.batch_size,
            target_size=self.target_size,
            class_mode='binary',
            color_mode='grayscale'
        )

        validation_images = self.test_datagen.flow_from_directory(
            directory='./Data/valid',
            batch_size=self.batch_size,
            target_size=self.target_size,
            class_mode='binary',
            color_mode='grayscale'
        )

        test_images = self.test_datagen.flow_from_directory(
            directory='./Data/test',
            batch_size=self.batch_size,
            target_size=self.target_size,
            class_mode='binary',
            color_mode='grayscale'
        )

        return train_images, validation_images, test_images

    def windown_optimizer(self, train_images, test_images, validation_images):
        """
        Train CNN based on window optimization.
        Reference:
        Lee, Hyunkwang, Myeongchan Kim, and Synho Do. "Practical window
        setting optimization for medical image deep learning."
        arXiv preprint arXiv:1812.00572 (2018).

        Args:
        ---------
            train_image: train set of data
            test_image: test set of data
            validation_images: validation set of data

        Return:
        ---------
            loss and accuracy graph; model
        """
        # For multi-channel WSOlayer
        nch_window = 1
        act_window = "sigmoid"
        upbound_window = 255.0
        init_windows = "stone_init"

        optimizer = SGD(lr=0.0001, decay=0, momentum=0.9, nesterov=False)
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

        # Add some
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

        # Double check initialized parameters for WSO
        names = [weight.name for layer in model.layers for weight in layer.weights]
        weights = model.get_weights()

        for name, weight in zip(names, weights):
            if "window_conv" in name:
                if "kernel:0" in name:
                    ws = weight
                if "bias:0" in name:
                    bs = weight

        print("window optimization modeul set up (initialized with {} settings)".format(init_windows))
        print("(WL, WW)={}".format(dict_window_settings[init_windows]))
        print("Loaded parameter : w={} b={}".format(ws[0, 0, 0, :], bs)) # check result
        print("Expected paramter(brain) : w=[0.11074668] b=[-5.5373344]")
        print("Expected paramter(subdural) : w=[0.08518976] b=[-4.259488]")

        model_out = model.fit_generator(
            train_images,
            steps_per_epoch=2000 // self.batch_size,
            epochs=self.epochs,
            validation_data=validation_images,
            validation_steps=800 // self.batch_size
        )

        model.summary()

        model_out = model.fit_generator(
            train_images,
            steps_per_epoch=2000 // self.batch_size,
            epochs=self.epochs,
            validation_data=validation_images,
            validation_steps=800 // self.batch_size
        )

        # Serialize model to json.
        model_json = model.to_json()
        with open("model_simple.json", "w") as json_file:
            json_file.write(model_json)

        # Serialize model to hdf5.
        model.save_weights('model_simple.h5')
        print('Saved model')

        graphs = Graphs()
        graphs.show_train_validation(
            self.epochs,
            model_out
        )

    def train_model(
        self,
        train_images,
        test_images,
        validation_images,
        optimizer_test,
        num_mid_kernel
    ):
        """
        Train simple CNN model

        Args:
        ---------
            train_image: train set of data
            test_image: test set of data
            validation_images: validation set of data

        Return:
        ---------
            loss and accuracy graph; model
        """
        # Optimizers
        if optimizer_test == 'SGD':
            optimizer = SGD(lr=0.0001, decay=0, momentum=0.9, nesterov=True)
        if optimizer_test == 'Ada':
            optimizer = Adadelta(lr=0.0001)
        if optimizer_test == 'Nadam':
            optimizer = Nadam(lr=0.0001)

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

        # Convolutional Kernel middle layer.
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(num_mid_kernel, (5, 5)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(BatchNormalization())
        # model.add(Conv2D(num_mid_kernel * 10, (5, 5)))
        # model.add(Activation('relu'))

        model.add(Flatten())
        model.add(Dense(
            64,
            bias_regularizer=l2(0.01),
            # activity_regularizer=l2(0.02)
            ))

        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        # TODO: EVALUATES IF relu RETURN BETTER RESULTS (I BELIEVE YES...)
        model.add(Activation('relu'))

        model.compile(
            loss='binary_crossentropy',
            # optimizer='rmsprop',
            optimizer=optimizer,
            metrics=['accuracy']
        )
        model.summary()

        model_out = model.fit_generator(
            train_images,
            # steps_per_epoch=2000 // self.batch_size,
            steps_per_epoch=len(train_images.classes) // self.batch_size,
            epochs=self.epochs,
            validation_data=validation_images,
            # validation_steps=800 // self.batch_size
            validation_steps=len(validation_images.classes) // self.batch_size
        )

        print('----------------')
        score = model.evaluate_generator(
            validation_images,
            len(validation_images.classes) // self.batch_size
        )
        print(score)

        scores = model.predict_generator(
            test_images,
            len(validation_images.classes) // self.batch_size
        )
        print(scores)
        print('----------------')

        # Serialize model to json.
        model_json = model.to_json()
        with open("model_simple.json", "w") as json_file:
            json_file.write(model_json)

        # Serialize model to hdf5.
        model.save_weights('model_simple.h5')
        print('Saved model')

        graphs = Graphs()
        graphs.show_train_validation(
            self.epochs,
            model_out
        )

        # Test:
        # pred = model.predict(
        #     # test_images
        #     train_images
        # )
        pred = model.predict_generator(
            test_images
            # len(test_images.classes) // self.batch_size
        )
        pred[pred <= 0.5] = 0
        pred[pred > 0.5] = 1

        print(accuracy_score(
            test_images.classes,
            # train_images.classes,
            pred
        ))

        graphs = Graphs()
        graphs.show_confusion_matrix(
            test_images.classes,
            # train_images.classes,
            pred,
            np.array(['glaucoma', 'healthy'])
        )

        return model_out

    def train_efficient_net(
        self,
        train_images,
        test_images,
        validation_images,
        optimizer_test
    ):
        """
        Refs.:
        https://www.kaggle.com/ateplyuk/keras-starter-efficientnet
        https://www.kaggle.com/krishnakatyal/keras-efficientnet-b3?utm_medium=email&utm_source=intercom&utm_campaign=competition-recaps-rsna-2019
        """
        # --- Try 1.
        # Optimizers
        if optimizer_test == 'SGD':
            optimizer = SGD(lr=0.0001, decay=0, momentum=0.9, nesterov=True)
        if optimizer_test == 'Ada':
            optimizer = Adadelta(lr=0.0001)
        if optimizer_test == 'Nadam':
            optimizer = Nadam(lr=0.0001)

        # eff_net = efn.EfficientNetB3(
        #     weights='imagenet',
        #     include_top=False,
        #     pooling='avg',
        #     input_shape=self.image_shape
        # )

        # x = eff_net.output
        # # x = Flatten()(x)
        # x = Dense(1024, activation="relu")(x)
        # x = Dropout(0.5)(x)
        # predictions = Dense(
        #     1,
        #     activation="sigmoid"
        # )(x)
        # model = Model(
        #     input=eff_net.input,
        #     output=predictions
        # )
        # model.compile(
        #     optimizer=optimizer,
        #     # optimizers.rmsprop(lr=0.0001, decay=1e-6),
        #     loss='binary_crossentropy',
        #     metrics=['accuracy']
        # )

        # model.summary()

        # model_out = model.fit_generator(
        #     train_images,
        #     steps_per_epoch=2000 // self.batch_size,
        #     epochs=self.epochs,
        #     validation_data=validation_images,
        #     validation_steps=800 // self.batch_size
        # )

        # --- Try 2.
        model = self.__create_eff_model()

        # Full Training Model
        for base_layer in model.layers[:-1]:
            base_layer.trainable = True
        TRAIN_STEPS = int(len(train_images) / 6)
        LR = 0.00011

        if self.epochs != 0:
            # Load Model Weights
            model.load_weights('model.h5')

        model.compile(
            optimizer=Adam(learning_rate=LR),
            loss='binary_crossentropy',
            metrics=['acc', tf.keras.metrics.AUC()]
        )
        model.summary()

        # Train Model
        model_out = model.fit_generator(
            generator=train_images,
            validation_data=validation_images,
            steps_per_epoch=TRAIN_STEPS,
            epochs=self.epochs,
            callbacks=[self.__model_checkpoint_full('model.h5')],
            verbose=1
        )

        # Serialize model to json.
        model_json = model.to_json()
        with open("model_simple.json", "w") as json_file:
            json_file.write(model_json)

        # Serialize model to hdf5.
        model.save_weights('model_simple.h5')
        print('Saved model')

        graphs = Graphs()
        graphs.show_train_validation(
            self.epochs,
            model_out
        )

    def __create_eff_model(self):
        """
        """
        K.clear_session()
        base_model = efn.EfficientNetB3(
            weights='imagenet',
            include_top=False,
            pooling='avg',
            input_shape=self.image_shape
        )

        x = base_model.output
        y_pred = Dense(
            6,
            activation='sigmoid'
        )(x)

        return Model(
            inputs=base_model.input,
            outputs=y_pred
        )

    def __model_checkpoint_full(self, model_name):
        """
        """
        return ModelCheckpoint(
            model_name,
            monitor = 'val_loss',
            verbose = 1,
            save_best_only = False,
            save_weights_only = True,
            mode = 'min',
            period = 1
        )
