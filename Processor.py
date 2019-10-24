from skimage.restoration import estimate_sigma
from scipy import fftpack
import numpy as np
from scipy import ndimage
import cv2
from skimage.metrics import peak_signal_noise_ratio


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
            noisy_image: noisy image
        """
        self.noisy_image = noisy_image

    def noise_estimation(self):
        """
        Estimates the power spectrum of noise image.
        Based on Robust wavelet-based estimator of the (Gaussian) noise standard deviation.
        (Wrapper of skimage.restoration)

        Return:
        ---------
            estimated_sigma: estimated noise standard deviation(s)        
        """
        return estimate_sigma(
            self.noisy_image,
            multichannel=True,
            average_sigmas=True
        )

    def get_global_snr(self, clean_image):
        """
        Compute the peak signal to noise ratio (PSNR) for an image.
        (Wrapper of skimage.metrics)

        Args:
        ---------
            clean_image: reference image

        Return:
        ---------
            psnr: peak signal to noise ratio
        """
        return peak_signal_noise_ratio(clean_image, self.noisy_image)
