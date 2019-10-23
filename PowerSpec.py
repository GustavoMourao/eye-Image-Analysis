"""
Power Spectra

These tools have been reimplemented and may be more up-to-date in turbustat:
https://github.com/Astroua/TurbuStat
"""

import numpy
try:
    import matplotlib.pyplot as pyplot
    pyplotOK = True
except ImportError:
    pyplotOK = False
from correlate2d import correlate2d
from AG_image_tools.radialprofile import azimuthalAverageBins,radialAverageBins

def hanning2d(M, N):
    """
    A 2D hanning window, as per IDL's hanning function.  See numpy.hanning for the 1d description
    """

    if N <= 1:
        return numpy.hanning(M)
    elif M <= 1:
        return numpy.hanning(N) # scalar unity; don't window if dims are too small
    else:
        return numpy.outer(numpy.hanning(M),numpy.hanning(N))

def power_spectrum(*args,**kwargs):
    """
    Thin wrapper of PSD2.  Returns the 1D power spectrum in stead of the 2D Power Spectral Density
    """
    kwargs['oned']=True
    return PSD2(*args,**kwargs)

def PSD2(image, image2=None, oned=False, 
        fft_pad=False, real=False, imag=False,
        binsize=1.0, radbins=1, azbins=1, radial=False, hanning=False, 
        wavnum_scale=False, twopi_scale=False, **kwargs):
    """
    Two-dimensional Power Spectral Density.
    NAN values are treated as zero.

    image2 - can specify a second image if you want to see the cross-power-spectrum instead of the 
        power spectrum.
    oned - return radial profile of 2D PSD (i.e. mean power as a function of spatial frequency)
           freq,zz = PSD2(image); plot(freq,zz) is a power spectrum
    fft_pad - Add zeros to the edge of the image before FFTing for a speed
        boost?  (the edge padding will be removed afterwards)
    real - Only compute the real part of the PSD (Default is absolute value)
    imag - Only compute the complex part of the PSD (Default is absolute value)
    hanning - Multiply the image to be PSD'd by a 2D Hanning window before performing the FTs.  
        Reduces edge effects.  This idea courtesy Paul Ricchiazzia (May 1993), author of the
        IDL astrolib psd.pro
    wavnum_scale - multiply the FFT^2 by the wavenumber when computing the PSD?
    twopi_scale - multiply the FFT^2 by 2pi?
    azbins - Number of azimuthal (angular) bins to include.  Default is 1, or
        all 360 degrees.  If azbins>1, the data will be split into [azbins]
        equally sized pie pieces.  Azbins can also be a numpy array.  See
        AG_image_tools.azimuthalAverageBins for details
        
    
    radial - An option to return the *azimuthal* power spectrum (i.e., the spectral power as a function 
        of angle).  Not commonly used.
    radbins - number of radial bins (you can compute the azimuthal power spectrum in different annuli)
    """
    
    # prevent modification of input image (i.e., the next two lines of active code)
    image = image.copy()

    # remove NANs (but not inf's)
    image[image!=image] = 0

    if hanning:
        image = hanning2d(*image.shape) * image

    if image2 is None:
        image2 = image
    else:
        image2 = image2.copy()
        image2[image2!=image2] = 0
        if hanning:
            image2 = hanning2d(*image2.shape) * image2

    if real:
        psd2 = numpy.real( correlate2d(image,image2,return_fft=True,fft_pad=fft_pad) ) 
    elif imag:
        psd2 = numpy.imag( correlate2d(image,image2,return_fft=True,fft_pad=fft_pad) ) 
    else: # default is absolute value
        psd2 = numpy.abs( correlate2d(image,image2,return_fft=True,fft_pad=fft_pad) ) 
    # normalization is approximately (numpy.abs(image).sum()*numpy.abs(image2).sum())

    if wavnum_scale:
        wx = numpy.concatenate([ numpy.arange(image.shape[0]/2,dtype='float') , image.shape[0]/2 - numpy.arange(image.shape[0]/2,dtype='float') -1 ]) / (image.shape[0]/2.)
        wy = numpy.concatenate([ numpy.arange(image.shape[1]/2,dtype='float') , image.shape[1]/2 - numpy.arange(image.shape[1]/2,dtype='float') -1 ]) / (image.shape[1]/2.)
        wx/=wx.max()
        wy/=wy.max()
        wavnum = numpy.sqrt( numpy.outer(wx,numpy.ones(wx.shape))**2 + numpy.outer(numpy.ones(wy.shape),wx)**2 )
        psd2 *= wavnum

    if twopi_scale:
        psd2 *= numpy.pi * 2

    if radial:
        azbins,az,zz = radialAverageBins(psd2,radbins=radbins, interpnan=True, binsize=binsize, **kwargs)
        if len(zz) == 1:
            return az,zz[0]
        else:
            return az,zz

    if oned:
        return pspec(psd2, azbins=azbins, binsize=binsize, **kwargs)

    # else...
    return psd2

def pspec(psd2, return_index=True, wavenumber=False, return_stddev=False, azbins=1, binsize=1.0, view=False, **kwargs):
    """
    Create a Power Spectrum (radial profile of a PSD) from a Power Spectral Density image

    return_index - if true, the first return item will be the indexes
    wavenumber - if one dimensional and return_index set, will return a normalized wavenumber instead
    view - Plot the PSD (in logspace)?
    """
    #freq = 1 + numpy.arange( numpy.floor( numpy.sqrt((image.shape[0]/2)**2+(image.shape[1]/2)**2) ) ) 

    azbins,(freq,zz) = azimuthalAverageBins(psd2,azbins=azbins,interpnan=True, binsize=binsize, **kwargs)
    if len(zz) == 1: zz=zz[0]
    # the "Frequency" is the spatial frequency f = 1/x for the standard numpy fft, which follows the convention
    # A_k =  \sum_{m=0}^{n-1} a_m \exp\left\{-2\pi i{mk \over n}\right\}
    # or 
    # F_f = Sum( a_m e^(-2 pi i f x_m)  over the range m,m_max where a_m are the values of the pixels, x_m are the
    # indices of the pixels, and f is the spatial frequency
    freq = freq.astype('float')  # there was a +1.0 here before, presumably to deal with div-by-0, but that shouldn't happen and shouldn't have been "accounted for" anyway

    if return_index:
        if wavenumber:
            fftwavenum = (numpy.fft.fftfreq(zz.size*2)[:zz.size])
            return_vals = list((fftwavenum,zz))
            #return_vals = list((len(freq)/freq,zz))
        else:
            return_vals = list((freq/len(freq),zz))
    else:
        return_vals = list(zz)
    if return_stddev:
        azbinsS,(freqstd,zzstd) = azimuthalAverageBins(psd2,azbins=azbins,stddev=True,interpnan=True, binsize=binsize, **kwargs)
        return_vals.append(zzstd)
    
    if view and pyplotOK:
        pyplot.loglog(freq,zz)
        pyplot.xlabel("Spatial Frequency")
        pyplot.ylabel("Spectral Power")

    return return_vals
