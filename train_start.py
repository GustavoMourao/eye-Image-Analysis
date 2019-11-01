from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.datasets import mnist
import matplotlib.pyplot as plt
import cv2

if __name__ == "__main__":
    """
    Main
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(type(x_train))
    # plt.imshow(x_train[0])
    # plt.show()

    num_classes = 2
    image_shape = (150, 150, 3)
    images = ImageDataGenerator(rescale=1/255)
    raw_set_data = images.flow_from_directory(
        directory='./Database/',
        batch_size=32,
        target_size=image_shape
    )

    # TODO: separates into train/validation/test/
    train_images = images.flow_from_directory(
        directory='./Data/train',
        batch_size=32,
        target_size=image_shape
    )

    test_images = images.flow_from_directory(
        directory='./Data/test',
        batch_size=32,
        target_size=image_shape
    )

    validation_images = images.flow_from_directory(
        directory='./Data/valid',
        batch_size=32,
        target_size=image_shape
    )

    # TODO: Creates labels for each class.
    # References:
    # https://kylewbanks.com/blog/loading-unlabeled-images-with-imagedatagenerator-flowfromdirectory-keras
    # https://github.com/keras-team/keras/issues/3946

    # TODO: apply data augmentation
    # TODO: Find ideal hiperparameters to CNN.
    model = Sequential()
    model.add(Conv2D(32,
                     kernel_size=(5, 5),
                     strides=(1, 1),
                     activation='relu',
                     input_shape=(image_shape)
                     )
              )
    model.add(MaxPooling2D(pool_size=(2, 2),
                           strides=(2, 2)
                           )
              )
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    print(model.summary())

    model.compile(

        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['acc']
    )

    # models.fit(
    # )

    # 3.2. Evaluates ideal hiperparameters!
