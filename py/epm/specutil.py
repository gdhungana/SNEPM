import numpy as np
import speclite.filters
import glob
from astropy.time import Time
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcess
from epm.util import unred


def spec_k_correction(z,wave,flux,band='V'):
    """
    use spectra to estimate the k correction for the given band
    spectra should be in rest frame

    TODO: estimate error
    Only in bessell filters for now
    """
    filt=speclite.filters.load_filter('bessell-'+band)
    obsmag=filt.get_ab_magnitude(wavelength=wave,spectrum=flux)

    resmag=filt.get_ab_magnitude(wavelength=wave/(1+z),spectrum=flux)
    #print "observed synthetic mag", obsmag
    #print "rest frame mag", resmag

    return obsmag-resmag #- This has to be subtracted from observed magnitude

