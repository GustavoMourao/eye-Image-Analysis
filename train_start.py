from Interpreter import Interpreter
import numpy as np


if __name__ == "__main__":
    """
    Get raw data and apply CNN model.
    """
    BATCH_SIZE = 32
    IMAGE_SHAPE = (225, 225, 1)
    inter = Interpreter(
        BATCH_SIZE,
        IMAGE_SHAPE
    )

    train_images, test_images, validation_images = inter.split_data()

    # Window Optimization.
    # inter.windown_optimizer(
    #     train_images,
    #     test_images,
    #     validation_images
    # )

    # Traditional method.
    inter.train_model(
        train_images,
        test_images,
        validation_images
    )

    # TODO
    # 3. Evaluates ideal hiperparameters!
