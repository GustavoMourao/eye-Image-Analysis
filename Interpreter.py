from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2, l1
import numpy as np
from keras.optimizers import SGD, Adadelta, Nadam
import keras
from keras import regularizers
from keras.layers import GlobalAveragePooling2D
from keras.models import Model
# from WindowOpt.functions import *
# from WindowOpt.WindowsOpt import *
from Graphs import Graphs
# import efficientnet.keras as efn
# import tensorflow as tf
from keras.layers.normalization import BatchNormalization
from sklearn.metrics import accuracy_score
from efficientnet_keras_transfer_learning.efficientnet import EfficientNetB0 as Net0
from efficientnet_keras_transfer_learning.efficientnet import EfficientNetB1 as Net1
from efficientnet_keras_transfer_learning.efficientnet import EfficientNetB2 as Net2
from efficientnet_keras_transfer_learning.efficientnet import EfficientNetB3 as Net3
from efficientnet_keras_transfer_learning.efficientnet import EfficientNetB4 as Net4
from efficientnet_keras_transfer_learning.efficientnet import EfficientNetB5 as Net5
from efficientnet_keras_transfer_learning.efficientnet import EfficientNetB6 as Net6
from efficientnet_keras_transfer_learning.efficientnet import EfficientNetB7 as Net7
from efficientnet_keras_transfer_learning.efficientnet import center_crop_and_resize, preprocess_input
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model



# Model parameter (Resnet case)
# ----------------------------------------------------------------------------
#           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
# Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
#           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
# ----------------------------------------------------------------------------
# ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
# ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
# ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
# ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
# ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
# ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
# ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
# ---------------------------------------------------------------------------

class Interpreter:
    """
    Class responsible to split raw data into train, validation and test.
    Besides of that, allows to train different CNN topologies.
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
        Get train image information.

        Args:
        ---------
            imagepath: path image

        Return:
        ---------
            image information
        """
        return self.train_datagen.flow_from_directory(
            directory=imagepath,
            batch_size=self.batch_size,
            target_size=(225, 225),
            class_mode='binary'
            # color_mode='grayscale'
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
            class_mode='binary'
            # color_mode='grayscale'
        )

        validation_images = self.test_datagen.flow_from_directory(
            directory='./Data/valid',
            batch_size=self.batch_size,
            target_size=self.target_size,
            class_mode='binary'
            # color_mode='grayscale'
        )

        test_images = self.test_datagen.flow_from_directory(
            directory='./Data/test',
            batch_size=self.batch_size,
            target_size=self.target_size,
            class_mode='binary'
            # color_mode='grayscale'
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

        return model, model_out

    def train_model(
        self,
        train_images,
        validation_images,
        optimizer_test,
        num_mid_kernel
    ):
        """
        Train simple CNN model.

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

        model.add(Conv2D(16, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Convolutional Kernel middle layer.
        model.add(
          Conv2D(
            24,
            (3, 3)
            )
        )
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(24, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(40, (5, 5)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(40, (5, 5)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(80, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(80, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(80, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(112, (5, 5)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(112, (5, 5)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(192, (5, 5)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(192, (5, 5)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(192, (5, 5)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(192, (5, 5)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(320, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(
            64,
            bias_regularizer=l2(0.01),
            # activity_regularizer=l2(0.02)
            ))

        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.compile(
            loss='binary_crossentropy',
            # optimizer='rmsprop',
            optimizer=optimizer,
            metrics=['accuracy']
        )
        model.summary()

        model_out = model.fit_generator(
            train_images,
            steps_per_epoch=len(train_images.classes) // self.batch_size,
            epochs=self.epochs,
            validation_data=validation_images,
            validation_steps=len(validation_images.classes) // self.batch_size
        )

        return model, model_out

    def train_efficient_net(
        self,
        train_images,
        test_images,
        validation_images,
        pretrained=True,
        topology=0
    ):
        """
        Train last layer of EfficientNet. It is possible to choose between
        EfficientNet-B0 to B7.

        Args:
        ---------
            train_image: train set of data
            test_image: test set of data
            validation_images: validation set of data
            pretrained: [True] - Pretrained weights / [False] - Train weights
            topology: (0) - EfficientNetB0; (1) - EfficientNetB1; ...

        Return:
        ---------
            loss and accuracy graph; model
        """
        conv_base = self.__get_eff_model(pretrained, topology)

        model = models.Sequential()
        model.add(conv_base)
        model.summary()

        model.add(layers.GlobalMaxPooling2D(name="gap"))
        # model.add(layers.Flatten(name="flatten"))
        model.add(layers.Dropout(
            0.2,
            name="dropout_out"
        ))
        model.summary()

        # model.add(layers.Dense(256, activation='relu', name="fc1"))
        model.add(layers.Dense(
            2,
            activation='softmax',
            name="fc_out"
        ))

        model.summary()

        print('This is the number of trainable layers '
              'before freezing the conv base:', len(model.trainable_weights))

        conv_base.trainable = False

        print('This is the number of trainable layers '
              'after freezing the conv base:', len(model.trainable_weights))

        model.compile(
            # loss='categorical_crossentropy',
            loss='sparse_categorical_crossentropy',
            optimizer=optimizers.RMSprop(lr=2e-5),
            metrics=['acc']
        )

        model_out = model.fit_generator(
            train_images,
            steps_per_epoch=len(train_images.classes) // self.batch_size,
            epochs=self.epochs,
            validation_data=validation_images,
            validation_steps=len(validation_images.classes) // self.batch_size,
            verbose=1,
            use_multiprocessing=True,
            workers=2
        )

        return model, model_out

    def resnet_layer(self,
                     inputs,
                     num_filters=16,
                     kernel_size=3,
                     strides=1,
                     activation='relu',
                     batch_normalization=True,
                     conv_first=True):
        """2D Convolution-Batch Normalization-Activation stack builder

        # Arguments
            inputs (tensor): input tensor from input image or previous layer
            num_filters (int): Conv2D number of filters
            kernel_size (int): Conv2D square kernel dimensions
            strides (int): Conv2D square stride dimensions
            activation (string): activation name
            batch_normalization (bool): whether to include batch normalization
            conv_first (bool): conv-bn-activation (True) or
                bn-activation-conv (False)

        # Returns
            x (tensor): tensor as input to the next layer
        """
        conv = Conv2D(num_filters,
                      kernel_size=kernel_size,
                      strides=strides,
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(1e-4))

        x = inputs
        if conv_first:
            x = conv(x)
            if batch_normalization:
                x = BatchNormalization()(x)
            if activation is not None:
                x = Activation(activation)(x)
        else:
            if batch_normalization:
                x = BatchNormalization()(x)
            if activation is not None:
                x = Activation(activation)(x)
            x = conv(x)
        return x

    def resnet_v1(
        self,
        input_shape,
        depth,
        train_images,
        test_images,
        validation_images,
        num_classes=1,
    ):
        """ResNet Version 1 Model builder [a]

        Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
        Last ReLU is after the shortcut connection.
        At the beginning of each stage, the feature map size is halved
        (downsampled)
        by a convolutional layer with strides=2, while the number of filters is
        doubled. Within each stage, the layers have the same number filters
        and the
        same number of filters.
        Features maps sizes:
        stage 0: 32x32, 16
        stage 1: 16x16, 32
        stage 2:  8x8,  64
        The Number of parameters is approx the same as Table 6 of [a]:
        ResNet20 0.27M
        ResNet32 0.46M
        ResNet44 0.66M
        ResNet56 0.85M
        ResNet110 1.7M

        # Arguments
            input_shape (tensor): shape of input image tensor
            depth (int): number of core convolutional layers
            num_classes (int): number of classes (CIFAR10 has 10)

        # Returns
            model (Model): Keras model instance
        """
        if (depth - 2) % 6 != 0:
            raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
        # Start model definition.
        num_filters = 16
        num_res_blocks = int((depth - 2) / 6)

        inputs = Input(shape=input_shape)
        x = self.resnet_layer(inputs=inputs)
        # Instantiate the stack of residual units
        for stack in range(3):
            for res_block in range(num_res_blocks):
                strides = 1
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    strides = 2  # downsample
                y = self.resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 strides=strides)
                y = self.resnet_layer(inputs=y,
                                 num_filters=num_filters,
                                 activation=None)
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    # linear projection residual shortcut connection to match
                    # changed dims
                    x = self.resnet_layer(inputs=x,
                                     num_filters=num_filters,
                                     kernel_size=1,
                                     strides=strides,
                                     activation=None,
                                     batch_normalization=False)
                x = keras.layers.add([x, y])
                x = Activation('relu')(x)
            num_filters *= 2

        # Add classifier on top.
        # v1 does not use BN after last shortcut connection-ReLU
        x = AveragePooling2D(pool_size=8)(x)
        y = Flatten()(x)
        outputs = Dense(num_classes,
                        activation='softmax',
                        kernel_initializer='he_normal')(y)

        # Instantiate model.
        model = Model(inputs=inputs, outputs=outputs)

        model.compile(
            # loss='categorical_crossentropy',
            loss='sparse_categorical_crossentropy',
            optimizer=optimizers.RMSprop(lr=2e-5),
            metrics=['acc']
        )
        model.summary()

        model_out = model.fit_generator(
            train_images,
            steps_per_epoch=len(train_images.classes) // self.batch_size,
            epochs=self.epochs,
            validation_data=validation_images,
            validation_steps=len(validation_images.classes) // self.batch_size
        )

        return model, model_out

    def model_evaluation_test(
        self,
        test_images,
        validation_images,
        model,
        model_out
    ):
        """
        Get score from test data from trained model.

        Args:
        ---------
        test_images: set of test data
        validation_images: set of validation data
        model: model
        model_out: trained model

        Return:
        ---------
            shows confusion matrix and saves model case
            accuracy higher than 0.6
        """
        graphs = Graphs()
        graphs.show_train_validation(
            self.epochs,
            model_out
        )

        pred = model.predict_generator(
            test_images
        )
        pred[pred <= 0.5] = 1
        pred[pred > 0.5] = 0

        graphs = Graphs()
        graphs.show_confusion_matrix(
            test_images.classes,
            pred,
            np.array(['glaucoma', 'healthy'])
        )

        print('Accuracy: \n')
        print(accuracy_score(
            test_images.classes,
            pred
        ))

        # Saves model case accuracy higher than 0.6
        if (accuracy_score(
            test_images.classes,
            pred
        ) > 0.6):

            # Serialize model to json.
            model_json = model.to_json()
            with open("model_simple.json", "w") as json_file:
                json_file.write(model_json)

            # Serialize model to hdf5.
            model.save_weights('model_simple.h5')
            print('Saved model')

    def __get_eff_model(self, pretrained, topology):
        """
        Return topology of efficientnet. Options: EfficentNetB0 - B7.

        Args:
        ---------
            topology: (0) - EfficientNetB0; (1) - EfficientNetB1; ...

        Return:
        ---------
            choosed model
        """
        if pretrained:
            load_weight = 'imagenet'
        else:
            load_weight = None

        if topology == 0:
            return Net0(
                weights=load_weight,
                include_top=False,
                input_shape=self.image_shape
            )

        if topology == 1:
            return Net1(
                weights=load_weight,
                include_top=False,
                input_shape=self.image_shape
            )

        if topology == 2:
            return Net2(
                weights=load_weight,
                include_top=False,
                input_shape=self.image_shape
            )

        if topology == 3:
            return Net3(
                weights=load_weight,
                include_top=False,
                input_shape=self.image_shape
            )

        if topology == 4:
            return Net4(
                weights=load_weight,
                include_top=False,
                input_shape=self.image_shape
            )

        if topology == 5:
            return Net5(
                weights=load_weight,
                include_top=False,
                input_shape=self.image_shape
            )

        if topology == 6:
            return Net6(
                weights=load_weight,
                include_top=False,
                input_shape=self.image_shape
            )

        if topology == 7:
            return Net7(
                weights=load_weight,
                include_top=False,
                input_shape=self.image_shape
            )
