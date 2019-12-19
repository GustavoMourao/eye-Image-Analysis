from Interpreter import Interpreter


if __name__ == "__main__":
    """
    Get raw data and apply CNN model.
    """
    TARGET_SIZE = (512, 512)
    BATCH_SIZE = 8
    EPOCHS = 100
    IMAGE_SHAPE_EFFI = (512, 512, 3)

    inter = Interpreter(
        BATCH_SIZE,
        IMAGE_SHAPE_EFFI,
        EPOCHS,
        TARGET_SIZE
    )

    train_images, validation_images, test_images = inter.split_data()

    # Resnet model:
    # https://keras.io/examples/cifar10_resnet/
    n = 3
    depth = n * 6 + 2
    model, model_out = inter.resnet_v1(
        input_shape=IMAGE_SHAPE_EFFI,
        depth=depth,
        train_images=train_images,
        test_images=test_images,
        validation_images=validation_images
    )

    # Get score of test data from trained model.
    inter.model_evaluation_test(
        test_images,
        validation_images,
        model,
        model_out
    )

    # # Traditional method.
    # model, model_out = inter.train_model(
    #     train_images,
    #     validation_images,
    #     'Nadam',
    #     256
    # )

    # # Get score of test data from trained model.
    # inter.model_evaluation_test(
    #     test_images,
    #     validation_images,
    #     model,
    #     model_out
    # )

    # # Eff. net
    # model, model_out = inter.train_efficient_net(
    #     train_images,
    #     test_images,
    #     validation_images,
    #     0
    # )

    # inter.model_evaluation_test(
    #     test_images,
    #     validation_images,
    #     model,
    #     model_out
    # )

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
