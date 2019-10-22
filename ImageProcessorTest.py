import unittest
from Processor import Processor
import cv2
import os


# class ProcessorTest(unittest.TestCase):
class ProcessorTest():
    """
    """
    def test_get_global_snr(self):
        """
        Unit test that verify if image, after eliminates noise,
        has SNR above the defined threshold.

        Args:
        ---------
            image: raw image
        """
        # TODO: Evaluates if this is the right way to pass these parameter.
        # TODO: Insert regions on unit test: 
        # Act, test, ...
        os.chdir('healthy')
        image = cv2.imread('01_h.jpg', 0)

        processor = Processor(image)
        global_snr = processor.get_global_snr()

        assert global_snr > -20


if __name__ == '__main__':
    """
    TODO: Remove after validated!
    """
    proc = ProcessorTest()
    proc.test_get_global_snr()
    # unittest.main()
