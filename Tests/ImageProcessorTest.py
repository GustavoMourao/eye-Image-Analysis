import unittest
from Processor import Processor
import cv2
import os


class ProcessorTest(unittest.TestCase):
# class TestProcessor():
    """
    """
    def test_get_global_snr(self):
        """
        Unit test that verify if image, after eliminates noise,
        has SNR above the defined threshold.

        Args:
        ---------
            image: raw image (ELIMINATES THIS INPUT)
        """
        # TODO: Evaluates if this is the right way to pass these parameter.
        # TODO: Insert regions on unit test: 
        # Act, test, ...
        os.chdir('healthy')
        image = cv2.imread('01_h.jpg', 0)

        processor = Processor(image)
        global_snr = processor.get_global_snr(image)

        assert global_snr > -20

    def test_noise_estimation(self):
        """
        Unit test that verify if noise estimation of filtered image
        is above defined threshold (SEARCH A BASELINE!!!!!).

        Args:
        ---------
            image: raw image
        """
        # os.chdir('healthy')
        image = cv2.imread('01_h.jpg', 0)

        processor = Processor(image)
        sigma_noise = processor.noise_estimation()

        assert sigma_noise < 2


if __name__ == '__main__':
    """
    TODO: Remove after validated!
    """
    # proc = TestProcessor()
    # proc.test_noise_estimation()
    unittest.main()
