from skimage.restoration import estimate_sigma
from scipy import fftpack
import numpy as np
from scipy import ndimage
import cv2


class Processor:
    """
    """
    def __init__(self, image):
        """
        Get raw image on cv2 processing format.
        """
        self.image = image
        self.d_theta = 30
        self.r_min = 20
        self.r_max = 250

    def noise_estimation(self):
        """
        Estimates the power spectrum of noise image.
        """
        return estimate_sigma(
            self.image,
            multichannel=True,
            average_sigmas=True
        )

    def get_global_snr(self):
        """
        Get signal-to-noise ratio of the image in dB.
        """
        noise_power = self.noise_estimation()
        fft_image = fftpack.fft2(self.image)
        fft_image_shift = fftpack.fftshift(fft_image)
        psd2D = np.abs(fft_image_shift)
        # TODO: Find a way to estimate the psd or to establishment a ideal value of \sigma!

        angF, psd1DD = self.get_rpsd(
            psd2D,
            self.d_theta,
            self.r_min,
            self.r_min
        )

        psd1DD = np.mean(psd1DD)
        mu_signal = np.mean(psd1DD)
        snr = 20*np.log10(mu_signal/noise_power)

        return snr

    def get_rpsd(self, psd2D, dTheta, rMin, rMax):
        """
        Reference: https://gist.github.com/TangibitStudios
        TODO: Evaluates if there are other methods
        """
        h  = psd2D.shape[0]
        w  = psd2D.shape[1]
        wc = w//2
        hc = h//2
    
        # note that displaying PSD as image inverts Y axis
        # create an array of integer angular slices of dTheta
        Y, X  = np.ogrid[0:h, 0:w]
        theta = np.rad2deg(np.arctan2(-(Y-hc), (X-wc)))
        theta = np.mod(theta + dTheta/2 + 360, 360)
        theta = dTheta * (theta//dTheta)
        theta = theta.astype(np.int)
    
        # mask below rMin and above rMax by setting to -100
        R     = np.hypot(-(Y-hc), (X-wc))
        mask  = np.logical_and(R > rMin, R < rMax)
        theta = theta + 100
        theta = np.multiply(mask, theta)
        theta = theta - 100
    
        # SUM all psd2D pixels with label 'theta' for 0<=theta❤60 between rMin and rMax
        angF  = np.arange(0, 360, int(dTheta))
        psd1D = ndimage.sum(psd2D, theta, index=angF)
    
        # normalize each sector to the total sector power
        pwrTotal = np.sum(psd1D)
        psd1D    = psd1D/pwrTotal
    
        return angF, psd1D
