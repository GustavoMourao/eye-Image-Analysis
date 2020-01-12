from Interpreter import Interpreter
import unittest
from sklearn.metrics import average_precision_score


class TestInterpreter(unittest.TestCase):
    """
    Implements train CNN methods.
    """
    def test_train_model(self):
        """
        """
        # Arrange
        # In this step: arrange many CNN topologies; get alll data
        TARGET_SIZE = (128, 128)
        BATCH_SIZE = 32
        EPOCHS = 70
        IMAGE_SHAPE = (128, 128, 1)
        inter = Interpreter(
            BATCH_SIZE,
            IMAGE_SHAPE,
            EPOCHS,
            TARGET_SIZE
        )
        train_images, validation_images, test_images =\
            inter.split_data()

        # Act
        # In this step train each model
        model = inter.train_model(
            train_images,
            test_images,
            validation_images,
            'Nadam',
            128
        )

        pred = model.predict(
            test_images
        )
        pred[pred < 0.5] = 0
        pred[pred >= 0.5] = 1

        avrg_prec = average_precision_score(
            test_images.classes,
            pred
        )

        # Assert
        # In this step verify if each model has accuracy
        assert avrg_prec > 70/100


if __name__ == '__main__':
    """
    Call unit test.
    """
    unittest.main()
