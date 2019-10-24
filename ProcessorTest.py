import unittest
import cv2
import os
from Processor import Processor
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise


class ProcessorTest(unittest.TestCase):
    """
    Class responsible to call unit test methods of Processor class.
    """
    #def test_get_global_snr(self):
    #    """
    #    Unit test that verify if image, after eliminates noise,
    #    has SNR above the defined threshold. Higher values of SNR
    #    indicates that the 
    #    """
    #    # Arrange
    #    # TODO: Verify for each image or for randomly images?
    #    # 1. Get reference image (base line image)
    #    # 2. Get raw image
    #    # 3. Denoise image
    #    image = cv2.imread('02_h.jpg', 0)
    #
    #    # Act
    #    processor = Processor(image)
    #    global_snr = processor.get_global_snr(image)
    #
    #    # Assert
    #    assert global_snr > -20
    #
    #def test_noise_estimation(self):
    #    """
    #    Unit test that verify if noise estimation of filtered image
    #    is above defined threshold.
    #    """
    #    # Arrange
    #    # os.chdir('healthy')
    #    image = cv2.imread('02_h.jpg', 0)
    #
    #    # Act
    #    processor = Processor(image)
    #    sigma_noise = processor.noise_estimation()
    #
    #    # Assert
    #    assert sigma_noise > 0.1

    def test_filter(self):
        """
        Unit test that verify if noise estimation of filtered image
        is above defined threshold.
        """
        # Arrange
        image_ref = cv2.imread('02_h.jpg', 0)
        noise_image = random_noise(image_ref, mode='s&p', amount=0.3)

        # Act
        processor = Processor(noise_image)
        filtered_img_mean = processor.filter_mean()       
        filtered_img_mdan = processor.filter_median()
        # BUG: It is broken over here!
        filtered_img_wner = processor.filter_wiener()

        snr_filter_mean = processor.get_global_snr(filtered_img_mean)
        snr_filter_mdan = processor.get_global_snr(filtered_img_mdan)
        snr_filter_wner = processor.get_global_snr(filtered_img_wner)

        # Assert
        assert snr_filter_mean > 0
        assert snr_filter_mdan > 0
        assert snr_filter_wner > 0


if __name__ == '__main__':
    """
    Call unit test methods.
    """
    os.chdir('healthy')
    proc = ProcessorTest()
    proc.test_filter()
    unittest.main()
