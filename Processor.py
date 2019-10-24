from skimage.restoration import estimate_sigma
import cv2
from skimage.metrics import peak_signal_noise_ratio
from skimage import filters
from skimage import restoration
import numpy as np


class Processor:
    """
    Class that implements classical signal processing image methods, as
    noise estimation and signal-to-noise ratio based on clean image.
    """
    def __init__(self, noisy_image):
        """
        Get raw image on cv2 processing format.

        Args:
        ---------
            noisy_image: noisy image (without processing steps)
        """
        self.noisy_image = noisy_image

    def get_global_psnr(self, filtered_image):
        """
        Compute the peak signal to noise ratio (PSNR) for an image.

        Obs:
            Wrapper of skimage.metrics

        Args:
        ---------
            filtered_image: reference image

        Return:
        ---------
            psnr: peak signal to noise ratio
        """
        return peak_signal_noise_ratio(
            filtered_image,
            self.noisy_image
        )

    def noise_estimation(self, image):
        """
        Estimates the power spectrum of noise image.
        Based on Robust wavelet-based estimator of the (Gaussian)
        noise standard deviation.

        Obs:
            Wrapper of skimage.restoration

        Args:
        ---------
            image: image to apply processing step
        Return:
        ---------
            estimated_sigma: estimated noise standard deviation
        """
        return estimate_sigma(
            image,
            multichannel=True,
            average_sigmas=True
        )

    def filter_mean(self, image):
        """
        Filter image based on mean mask (linear). Convolves image with
        normalized box filter.

        Obs:
            Wrapper of opencv

        Args:
        ---------
            image: image to apply processing step

        Return:
        ---------
            image_denoised: denoised image
        """
        return cv2.boxFilter(
            image,
            -1,
            (3, 3)
        )

    def filter_median(self, image):
        """
        Filter image based on median mask (non-linear). Each output is computed
        as the median value of the oinút samples under the analyzed window.

        Obs:
            Wrapper of skimage

        Args:
        ---------
            image: image to apply processing step

        Return:
        ---------
            image_denoised: denoised image
        """
        return filters.median(image)

    def filter_wiener(self, image):
        """
        Filter image based on Wiener mask (non-linear).

        Obs:
            Wrapper of skimage

        Args:
        ---------
            image: image to apply processing step

        Return:
        ---------
            image_denoised: denoised image
        """
        return restoration.wiener(
            image,
            np.ones((5, 5)) / 25,
            1100
        )

    def get_histogram(self, image):
        """
        Get histogram of image (tipically before filter signal).

        Obs:
            Wrapper of cv2

        Args:
        ---------
            image: image to apply processing step

        Return:
        ---------
            image_hist: image histogram
        """
        return cv2.calcHist(
            image,
            [0],
            None,
            [256],
            [10, 256]
        )

    def get_histogram_equalized(self, image):
        """
        Histogram equalization of image (tipically before filter signal).

        Obs:
            Wrapper of cv2

        Args:
        ---------
            image: image to apply processing step

        Return:
        ---------
            image_equalized: histogram image equalized
        """
        # img_eql = cv2.equalizeHist(image)
        return cv2.calcHist(
            # img_eql,
            cv2.equalizeHist(image),
            [0],
            None,
            [256],
            [10, 256]
        )
