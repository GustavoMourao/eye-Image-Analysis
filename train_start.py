from Interpreter import Interpreter


if __name__ == "__main__":
    """
    Get raw data and apply CNN model.
    """
    TARGET_SIZE = (128, 128)
    BATCH_SIZE = 32
    EPOCHS = 100
    IMAGE_SHAPE = (128, 128, 1)
    IMAGE_SHAPE_EFFI = (128, 128, 3)
    inter = Interpreter(
        BATCH_SIZE,
        IMAGE_SHAPE,
        EPOCHS,
        TARGET_SIZE
    )

    train_images, validation_images, test_images = inter.split_data()

    # Traditional method.
    inter.train_model(
        train_images,
        test_images,
        validation_images,
        'SGD',
        2048
    )

    # inter = Interpreter(
    #     BATCH_SIZE,
    #     IMAGE_SHAPE_EFFI,
    #     EPOCHS,
    #     TARGET_SIZE
    # )

    # train_images, validation_images, test_images = inter.split_data()    

    # # Eff. net
    # inter.train_efficient_net(
    #     train_images,
    #     test_images,
    #     validation_images,
    #     'Nadam'
    # )

    # Window Optimization.
    # inter.windown_optimizer(
    #     train_images,
    #     test_images,
    #     validation_images
    # )
