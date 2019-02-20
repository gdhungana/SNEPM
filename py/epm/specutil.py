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


def gaussprocess_sample(path,epoch=10.,snkcorrfile=None,snname='this-SN'):
    files=glob.glob(path+'/kcorr_*')
    ep=[]
    kcorr=[]
    ekcorr=[]
    for ff in files:
        thisep,thiskcorr,thisekcorr=np.loadtxt(ff,unpack=True)
        sel=np.where(thisep<100)
        thisep=thisep[sel]
        thiskcorr=thiskcorr[sel]
        thisekcorr=thisekcorr[sel]
        ep=np.concatenate([ep,thisep])
        kcorr=np.concatenate([kcorr,thiskcorr])
        ekcorr=np.concatenate([ekcorr,thisekcorr])

    tt=np.argsort(ep)
    ep=ep[tt]
    kcorr=kcorr[tt]
    ekcorr=np.sqrt(ekcorr[tt]**2+0.01**2)
    gp1=GaussianProcess(corr='squared_exponential',theta0=0.05, nugget= (ekcorr/kcorr)**2, random_state=0)
    gp1.fit(ep[:,None],kcorr)
    nep=np.arange(5,100)
    fit1,MSE1=gp1.predict(nep[:,None],eval_MSE=True)
    fiterr1=np.sqrt(MSE1)
    #- plot
    #plt.plot(ep,kcorr,'go',label='K-correction from spectra')
    plt.errorbar(ep,kcorr,yerr=ekcorr,color='g',marker='o',capsize=0.1,ls='None',label='K-correction from sample spectra')
    plt.plot(nep,fit1,'-',color='black',label='GP Regression model')
    plt.fill_between(nep,fit1-2*fiterr1,fit1+2*fiterr1,color='gray',alpha=0.3)

    if snkcorrfile is not None:
        snep,snkcorr,snekcorr=np.loadtxt(snkcorrfile,unpack=True)
        snekcorr=np.sqrt(snekcorr**2+0.01**2)
        plt.errorbar(snep,snkcorr,yerr=snekcorr,color='r',marker='o',capsize=0.,ls='None',label='K-correction from {} spectra'.format(snname))
    plt.legend(numpoints=1,loc=4,fontsize=22)
    plt.xlabel(r'${\rm Days\ since\ explosion}$',fontsize=22)
    plt.ylabel(r'$K-{\rm correction\ [V-mag]}$',fontsize=22)
    plt.tick_params(axis='both',labelsize=18)
    plt.ylim(-0.1,0.06)
    plt.tight_layout()
    plt.savefig('Kcorrection_{}.eps'.format(snname))
    plt.show()
    value,value_var=gp1.predict(epoch[:,None],eval_MSE=True)
    np.savetxt('kcorrection_at_rotseep.dat',np.c_[epoch,value,np.sqrt(value_var)],fmt='%.4f')
    
    return value, np.sqrt(value_var)
