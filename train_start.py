from Interpreter import Interpreter


if __name__ == "__main__":
    """
    Main
    """
    BATCH_SIZE = 16
    IMAGE_SHAPE = (150, 150, 3)
    inter = Interpreter(
        BATCH_SIZE,
        IMAGE_SHAPE
    )

    train_images, test_images, validation_images = inter.split_data()

    inter.windown_optimizer(
        train_images,
        test_images,
        validation_images
    )

    inter.train_model(
        train_images,
        test_images,
        validation_images
    )

    # 3.2. Evaluates ideal hiperparameters!

    # 4. Implements with offline images:
    # https://androidkt.com/how-to-predict-images-using-trained-keras-model/
