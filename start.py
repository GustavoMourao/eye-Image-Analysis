import os
import cv2
import matplotlib.pylab as plt
from skimage.restoration import estimate_sigma
import random
import numpy as np
import medpy
from scipy import fftpack
from scipy import ndimage
import Processor


def GetRPSD(psd2D, dTheta, rMin, rMax):
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


def azimuthalAverage(image, center=None):
    """
    Get from: https://github.com/mkolopanis/python/blob/master/radialProfile.py
    Calculate the azimuthally averaged radial profile.
    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fracitonal pixels).
    
    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if not center:
        center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]       # location of changed radius
    nr = rind[1:] - rind[:-1]        # number of radius bin
    
    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr

    return radial_prof

def sp_noise(image,prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


def estimate_noise_power(image):
    """
    """
    return medpy.filter.noise.immerkaer(image)


def estimate_noise(image):
    """
    """
    # It seems that this value significates the variance.
    # If so, converts it to power in dB
    # To estimates the SNR in terms of noise variance, try:
    # SNR[dB] = 20*math.log((1-noise_variance)/noise_variance)
    # SNR = mean(signal mean or expected value)/standard deviation of the noise
    sigma_noise = estimate_sigma(image, multichannel=True, average_sigmas=True)

    # TODO: Estimates image power spectrum
    # 1. Take FFT image
    fft_image = fftpack.fft2(image)

    # 2. Shift the FFT
    fft_image_shift = fftpack.fftshift(fft_image)

    # 3. Calculates the power spectrum
    psd2D = np.abs(fft_image_shift)

    # Converts this to one dimensional power spectrum. To do that, apply radial profile
    # and azimuthal average
    angF, psd1DD = GetRPSD(psd2D, 30, 20, 250)
    psd1DD = np.mean(psd1DD)

    psd1D = azimuthalAverage(psd2D)
    psd1D = psd1D/np.max(psd1D)

    # mu_signal = np.abs(np.mean(psd1D) - sigma_noise)
    # mu_signal = np.mean(psd1D)
    mu_signal = np.mean(psd1DD)
    snr = mu_signal/sigma_noise
    snd_db = 20*np.log10(snr)
    # snd_db = 20*np.log10(sigma_noise)
    return snd_db


if __name__=='__main__':
    """
    """
    # Get raw paramaters (exploratory analisys).
    os.chdir('healthy')
    img = cv2.imread('01_h.jpg', 0)


    # Test: add noise to image
    img_noised = sp_noise(img, 0.6)
    img_noised_power = estimate_noise(img_noised)
    img_power = estimate_noise(img)
    # img_power_2 = estimate_noise_power(img)

    plt.imshow(img_noised)
    plt.show()

    # 1. Histogram equalization.
    img_eql = cv2.equalizeHist(img)
    img_eql_hist = cv2.calcHist(img_eql,[0], None, [256], [0,256])

    # Test: Wiener estimation

    # Median filter apply.
    # TODO: First: estimates noise power
    noise_before = estimate_noise(img)
    # TODO: Inicialmente ler a relação sinal-ruído da imagem.
    # TODO: Após pŕocesso de filtragem a imagem deve apresentar uma SND > X dB
    img_median_filt = cv2.medianBlur(img, 5)
    noise_after = estimate_noise(img_median_filt)
    
    plt.imshow(img_median_filt)
    # plt.plot(img_median_filt)
    plt.show()
    print('-')
