from Interpreter import Interpreter


if __name__ == "__main__":
    """
    Get raw data and apply CNN model.
    """
    TARGET_SIZE = (256, 256)
    BATCH_SIZE = 32
    EPOCHS = 2
    IMAGE_SHAPE = (256, 256, 3)
    inter = Interpreter(
        BATCH_SIZE,
        IMAGE_SHAPE,
        EPOCHS,
        TARGET_SIZE
    )

    train_images, validation_images, test_images = inter.split_data()

    # Traditional method.
    model, model_out = inter.train_model(
        train_images,
        validation_images,
        'SGD',
        256
    )

    # Get score of test data from trained model.
    inter.model_evaluation_test(
        test_images,
        validation_images,
        model,
        model_out
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
