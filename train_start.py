from Interpreter import Interpreter


if __name__ == "__main__":
    """
    Main
    """
    batch_size_num = 16
    image_shape = (150, 150, 3)
    inter = Interpreter(
        batch_size_num,
        image_shape
    )

    train_images, test_images, validation_images = inter.split_data()
    inter.train_model(
        train_images,
        test_images,
        validation_images
    )

    # 3.2. Evaluates ideal hiperparameters!
