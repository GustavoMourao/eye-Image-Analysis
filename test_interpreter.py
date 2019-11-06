import Interpreter as Interpreter
import unittest


class TestInterpreter(unittest.TestCase):
    """
    """
    def test_train_model(self):
        """
        """
        # Arrange
        BATCH_SIZE = 32
        IMAGE_SHAPE = (150, 150, 3)
        inter = Interpreter(
            BATCH_SIZE,
            IMAGE_SHAPE
        )

        train_images, test_images, validation_images = inter.split_data()
        inter.train_model(
            train_images,
            test_images,
            validation_images
        )
        # Act

        # Assert
        assert True


if __name__ == '__main__':
    """
    Call unit test.
    """
    unittest.main()
