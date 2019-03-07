import numpy as np
import util
import speclite.filters
import matplotlib.pyplot as plt
from compute_epm import get_bbflux
from photutil import ABspectrum

def simulate_bb(n=100,seed=1234,filt='R'):
    rst=np.random.seed(seed=seed)
    temp=np.random.uniform(2000.,17000.,n)
    mags=np.random.uniform(12,18,n)
    #temp=np.sort(temp)
    rotse_resp=util.rotse_response()
    band_resp=speclite.filters.load_filter('bessell-'+filt)
    wave=np.arange(2900.,11000.,100.)

    rotsem=np.zeros(len(temp))
    bandm=np.zeros(len(temp))
    rotsefl=np.zeros(len(temp))
    bandfl=np.zeros(len(temp))

    for ii in range(len(temp)):
        bb1,ebb1=get_bbflux(wave,temp[ii],500.)
        #print bb1
        #print wave
        #rotsefl[ii]=rotse_resp.convolve_with_array(wavelength=wave,values=bb1,interpolate=True)
        #bandfl[ii]=band_resp.convolve_with_array(wavelength=wave,values=bb1)
        rotsem[ii]=rotse_resp.get_ab_magnitude(wavelength=wave,spectrum=bb1)
        bandm[ii]=band_resp.get_ab_magnitude(wavelength=wave,spectrum=bb1)
        #bandem=band_resp.get_ab_magnitude(wavelength=wave,spectrum=ebb1)
        scale=mags[ii]-bandm[ii]
        bandm[ii]=mags[ii]
        rotsem[ii]=rotsem[ii]+scale

        #- evaluate errors (purely photo-statistics)
        thisfilt=speclite.filters.load_filter('bessell-'+filt)
        wavelength=thisfilt.wavelength
        rspec=ABspectrum(wavelength,magnitude=rotsem[ii])
        bandspec=ABspectrum(wavelength,magnitude=bandm[ii])
        filtconv=speclite.filters.FilterConvolution(thisfilt,wavelength,photon_weighted=False)
        rotsefl[ii]=filtconv(rspec)
        bandfl[ii]=filtconv(bandspec)

    #rotseem = rotseefl* 2.5/(rotsefl*np.log(10.0))
    #bandem = bandefl*2.5/(bandfl*np.log(10.0))
    return temp,rotsem, bandm #, rotseem, bandm, bandem

