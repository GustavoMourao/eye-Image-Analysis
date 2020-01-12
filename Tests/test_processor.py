import unittest
import cv2
import os
from Processor import Processor
from skimage.util import random_noise
import numpy as np
import matplotlib.pylab as plt


class TestProcessor(unittest.TestCase):
    """
    Class responsible to call unit test methods of Processor class.
    """
    def test_filter(self):
        """
        Unit test that verify if noise estimation of filtered image
        is above defined threshold.
        """
        # Arrange: Get figure and add noise (salt and pepper).
        image_ref = cv2.imread('02_h.jpg', 0)
        noise_image = random_noise(image_ref, mode='s&p', amount=0.3)

        # Act.
        processor = Processor(noise_image)
        filtered_img_mean = processor.filter_mean(noise_image)
        filtered_img_mdan = processor.filter_median(noise_image)
        filtered_img_wner = processor.filter_wiener(noise_image)

        plt.imshow(filtered_img_wner)
        plt.show()

        psnr_filter_mean = processor.get_global_psnr(filtered_img_mean)
        psnr_filter_mdan = processor.get_global_psnr(filtered_img_mdan)
        psnr_filter_wner = processor.get_global_psnr(filtered_img_wner)

        print('SNR after Mean filter: ', psnr_filter_mean)
        print('SNR after Median filter: ', psnr_filter_mdan)
        print('SNR after Wiener filter: ', psnr_filter_wner)

        # Assert.
        assert psnr_filter_mean > 0
        assert psnr_filter_mdan > 0
        assert psnr_filter_wner > 0

    def test_histogram_equalizer(self):
        """
        Unit test that verify if noise estimation of filtered image
        is above defined threshold.
        """
        # Arrange.
        image_ref = cv2.imread('02_h.jpg', 0)

        # Act: Get histogram of image without equalization.
        processor = Processor(image_ref)
        hist_imag_ref = processor.get_histogram(image_ref)
        mean_hist_imag_ref = np.mean(
            hist_imag_ref / np.max(hist_imag_ref)
        )
        std_hist_imag_ref = np.std(
            hist_imag_ref / np.max(hist_imag_ref)
        )
        hist_imag_ref_equl = processor.get_histogram_equalized(image_ref)
        mean_hist_imag_ref_equl = np.mean(
            hist_imag_ref_equl / np.max(hist_imag_ref_equl)
        )
        std_hist_imag_ref_equl = np.std(
            hist_imag_ref_equl / np.max(hist_imag_ref_equl)
        )

        print(
            'Histogram statistics of raw image: [mean] = ',
            mean_hist_imag_ref,
            '[STD] = ',
            std_hist_imag_ref
        )
        print(
            'Histogram statistics of equalized image: [mean] = ',
            mean_hist_imag_ref_equl,
            '[STD] = ',
            std_hist_imag_ref_equl
        )

        # Assert: evaluate variance since is a measure of image constrast
        assert mean_hist_imag_ref_equl < mean_hist_imag_ref
        assert std_hist_imag_ref_equl < std_hist_imag_ref


if __name__ == '__main__':
    """
    Call unit test methods.
    """
    os.chdir('healthy')
    unittest.main()
