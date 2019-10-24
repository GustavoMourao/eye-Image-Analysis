from skimage.restoration import estimate_sigma
import cv2
from skimage.metrics import peak_signal_noise_ratio
from skimage import filters


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

        Obs:
            Wrapper of skimage.restoration

        Return:
        ---------
            estimated_sigma: estimated noise standard deviation(s)
        """
        return estimate_sigma(
            self.noisy_image,
            multichannel=True,
            average_sigmas=True
        )

    def get_global_snr(self, filtered_image):
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
        return peak_signal_noise_ratio(filtered_image, self.noisy_image)

    def filter_mean(self):
        """
        Filter image based on mean mask (linear). Convolves image with
        normalized box filter.

        Obs:
            Wrapper of opencv

        Return:
        ---------
            image_denoised: denoised image
        """
        return cv2.boxFilter(self.noisy_image, -1, (3,3))

    def filter_median(self):
        """
        Filter image based on median mask (non-linear). Each output is computed
        as the median value of the oinút samples under the analyzed window.

        Obs:
            Wrapper of skimage

        Return:
        ---------
            image_denoised: denoised image
        """
        return filters.median(self.noisy_image)

    def filter_wiener(self):
        """
        Filter image based on Wiener mask (non-linear).

        Obs:
            Wrapper of skimage

        Return:
        ---------
            image_denoised: denoised image
        """
        return filters.wiener(self.noisy_image, None)
