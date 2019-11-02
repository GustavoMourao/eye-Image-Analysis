from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, MaxPool2D, Dropout
from keras.datasets import mnist
import matplotlib.pyplot as plt
import cv2
from keras import backend as K


if __name__ == "__main__":
    """
    Main
    """
    # Parameters configuration.
    batch_size_num = 16
    image_shape = (150,150,3)

    # This is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    train_images = train_datagen.flow_from_directory(
        directory='./Data/train',
        batch_size=batch_size_num,
        target_size=(150, 150),
        class_mode='binary'
    )

    images = ImageDataGenerator(rescale=1/255)
    test_images = images.flow_from_directory(
        directory='./Data/test',
        batch_size=batch_size_num,
        target_size=(150, 150),
        class_mode='binary'
    )

    validation_images = images.flow_from_directory(
        directory='./Data/valid',
        batch_size=batch_size_num,
        target_size=(150, 150),
        class_mode='binary'
    )

    # TODO: Find ideal hiperparameters to CNN.
    # Set up model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=image_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy']
    )

    model.fit_generator(
            train_images,
            steps_per_epoch=2000 // batch_size_num,
            epochs=50,
            validation_data=validation_images,
            validation_steps=800 // batch_size_num
    )

    model.save_weights('first_try.h5')  # always save your weights after training or during training
    print('end..')

    # 3.2. Evaluates ideal hiperparameters!
