import os
import cv2
import matplotlib.pylab as plt
from skimage.restoration import estimate_sigma
import random
import numpy as np


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


def esttimate_noise(image):
    """
    """
    # It seems that this value significates the variance.
    # If so, converts it to power in dB
    # To estimates the SNR in terms of noise variance, try:
    # SNR[dB] = 20*math.log((1-noise_variance)/noise_variance)
    return estimate_sigma(image, multichannel=True, average_sigmas=True)


if __name__=='__main__':
    """
    """
    # Get raw paramaters (exploratory analisys).
    os.chdir('healthy')
    img = cv2.imread('01_h.jpg', 0)

    # Test: add noise to image
    img_noised = sp_noise(img, 0.03)
    img_noised_power = esttimate_noise(img_noised)
    img_power = esttimate_noise(img)

    plt.imshow(img_noised)
    plt.show()

    # 1. Histogram equalization.
    img_eql = cv2.equalizeHist(img)
    img_eql_hist = cv2.calcHist(img_eql,[0], None, [256], [0,256])

    # Median filter apply.
    # TODO: First: estimates noise power
    noise_before = esttimate_noise(img)
    # TODO: Inicialmente ler a relação sinal-ruído da imagem.
    # TODO: Após pŕocesso de filtragem a imagem deve apresentar uma SND > X dB
    img_median_filt = cv2.medianBlur(img, 5)
    noise_after = esttimate_noise(img_median_filt)
    
    plt.imshow(img_median_filt)
    # plt.plot(img_median_filt)
    plt.show()
    print('-')
