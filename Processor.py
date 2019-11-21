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
            'noisy_image': noisy image (unprocessed)
        """
        self.noisy_image = noisy_image
        self.psf = np.ones((5, 5)) / 25
        self.balance = 1100
        self.hist_channel = [0]
        self.hist_size = [256]
        self.hist_ranges = [10, 256]

    def get_global_psnr(self, filtered_image):
        """
        Compute the peak signal to noise ratio (PSNR) for an image.
        This objective metric is calculated based on
        20*log10(MAXi / sqrt(MSE)), where MAXi is the max pixel value
        of the image and MSE is the mean square error between image and noise.
        Args:
        ---------
            'filtered_image': reference image (usually without noise)
        Return:
        ---------
            'psnr': peak signal to noise ratio
        Notes
        ---------
            Wrapper of skimage.metrics
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
        Args:
        ---------
            'image': image to apply processing step
        Return:
        ---------
            'estimated_sigma': estimated noise standard deviation
        Notes
        ---------
            Wrapper of skimage.restoration
        """
        return estimate_sigma(
            image,
            multichannel=True,
            average_sigmas=True
        )

    def filter_mean(self, image, ksize=(3, 3)):
        """
        Filter image based on mean mask (linear). Convolves image with
        normalized box filter.
        Args:
        ---------
            'image': image to apply processing step
            'ksize': kernel size of mask
        Return:
        ---------
            'image_denoised': denoised image
        Notes
        ---------
            Wrapper of opencv
        """
        return cv2.boxFilter(
            image,
            -1,
            ksize
        )

    def filter_median(self, image):
        """
        Filter image based on median mask (non-linear). Each output is computed
        as the median value of the oinút samples under the analyzed window.
        Args:
        ---------
            'image': image to apply processing step
        Return:
        ---------
            'image_denoised': denoised image
        Notes
        ---------
            Wrapper of skimage
        """
        return filters.median(image)

    def filter_wiener(self, image):
        """
        Filter image based on Wiener mask (non-linear).
        Args:
        ---------
            'image': image to apply processing step
        Return:
        ---------
            'image_denoised': denoised image
        Notes
        ---------
            Wrapper of skimage
        """

        # BUG: broken here for dicom images!

        return restoration.wiener(
            image,
            self.psf,
            self.balance
        )

    def get_histogram(self, image):
        """
        Get histogram of image (tipically before filter signal).
        Args:
        ---------
            'image': image to apply processing step
        Return:
        ---------
            'image_hist': image histogram
        Notes
        ---------
            Wrapper of cv2
        """
        return cv2.calcHist(
            image,
            self.hist_channel,
            None,
            self.hist_size,
            self.hist_ranges
        )

    def get_histogram_equalized(self, image):
        """
        Histogram equalization of image (tipically before filter signal).
        Args:
        ---------
            'image': image to apply processing step
        Return:
        ---------
            'image_equalized': histogram image equalized
        Notes
        ---------
            Wrapper of cv2
        """
        # In case of colorful image.
        if image.shape[2] == 3:

            r, g, b = cv2.split(image)
            output1_r = cv2.equalizeHist(r)
            output1_g = cv2.equalizeHist(g)
            output1_b = cv2.equalizeHist(b)

            image = cv2.merge((
                output1_r,
                output1_g,
                output1_b
            ))

            return image

        else:
            return cv2.calcHist(
                cv2.equalizeHist(image),
                self.hist_channel,
                None,
                self.hist_size,
                self.hist_ranges
            )
