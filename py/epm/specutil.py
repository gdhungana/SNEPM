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

def get_k_corrections(specpath,ebv=0,eebv=0,t0=0,zin=None,ezin=0,zout=None,ezout=0,band='V',snname='sn'):

    
    files=glob.glob(specpath+'/sn*')
    print "Specfiles are:", files
    
    #- get the epochs
    epochs=np.zeros(len(files))
    kcorr=np.zeros(len(files))
    ekcorr=np.zeros(len(files))
    for ii in range(len(files)):
        print files[ii]
        d=str.split(files[ii],'-')[1][:10]
        date=d[0:4]+'-'+d[4:6]+'-'+d[6:8]
        mjd=Time(date).mjd+float(d[8:])
        epochs[ii]=mjd-t0

        data=np.loadtxt(files[ii],unpack=True)
        w=data[0]
        flux=data[1]
        #- unreden:
        redcurve=unred(w,ebv)
        flux=flux*redcurve
        if zin is not None:
            rw=w/(1+zin) #- bring it to rest frame
            if np.logical_or((rw[0]>4700), (rw[-1] < 7000)):
                print("Skipping {} for not enough wavelength coverage for V band".format(files[ii]))
                print("wavelength is {},....... {}".format(rw[0],rw[-1]))
                continue
        if zout is not None:
            rw=rw*(1+zout) #- shifted to zout (observer frame)
            if np.logical_or((rw[0]>4700), (rw[-1] < 7000)):
                print("Skipping {} for not enough wavelength coverage for V band".format(files[ii]))
                print("wavelength is {},....... {}".format(rw[0],rw[-1]))
                continue

        kcorr[ii]=spec_k_correction(zout,rw,flux,band=band)
        dz=np.sqrt(ezin**2+ezout**2)
        ekcorr[ii]=spec_k_correction(dz,rw,flux,band=band)

        #- Additional error from ebv
        redcurvelo=unred(rw,ebv-eebv)
        redcurvehi=unred(rw,ebv+eebv)
        fluxlo=flux*redcurvelo
        fluxhi=flux*redcurvehi
        kcorrlo=spec_k_correction(zout,rw,fluxlo,band=band)
        kcorrhi=spec_k_correction(zout,rw,fluxhi,band=band)

        ekcorr[ii]=np.sqrt(ekcorr[ii]**2+((kcorrhi-kcorrlo)/2)**2)
        
    ekcorr=np.abs(ekcorr)
    kk=np.where(kcorr!=0.0)[0]
    kcorr=kcorr[kk]
    epochs=epochs[kk]
    ekcirr=ekcorr[kk]

    tt=np.argsort(epochs)
    if len(tt)>0:
        np.savetxt('kcorr_{}_z_{}.txt'.format(snname,zout),np.c_[epochs[tt],kcorr[tt],ekcorr[tt]],fmt='%.4f')

    return kcorr,ekcorr
