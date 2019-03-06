import numpy as np
import speclite.filters
import matplotlib.pyplot as plt
import scipy.constants as const
import astropy.units as u


hc=const.h*1.0e7*const.c*1.0e2 #- ergs s
jy=1.0e-23 #- ergs/s/Hz/cm2
_abconst= 3631*jy*const.c*1.0e2 *1.0e8#- ergs/s/cm2/Hz * A/s

def get_extinction(ebv, method='sf', band='R',rv=3.1):

    if method == 'sf': #- s&f 2011 using NED - Lambda is diff.
                      #- http://www.astro.sunysb.edu/metchev/PHY517_AST443/extinction_lab.pdf
        av = ebv*rv
        extmap={
        'a_uvw2': av*2.941, #- Apj 345, 245 Eq 4a,4b for swift
        'a_uvm2': av*3.078,
        'a_uvw1': av*2.104,
        'a_U':av*1.580,
        'a_B':av*1.322,
        'a_V':av*1,
        'a_R':av*0.791,
        'a_I':av*0.548,
        'a_J':av*0.258,
        'a_H':av*0.163,
        'a_K':av*0.110,
        'a_L':av*0.055,
        'a_M':av*0.023}
        
    if 'a_'+band in extmap.keys():
        print 'Extinction, A_lambda for band:', band, '=:', extmap['a_'+band]
        return extmap['a_'+band]

def mag_flux_density(mag,emag,band,system='Vega'):

    wave,fwhm=effwavelength(band)
    #- Vega system
    if system=='Vega':
        #- Bessel UBVRI (Bessell 98.)
        if band == 'U':       
            f_lambda = 10**(-0.4*(mag+21.1-0.152))
        if band == 'B': 
            f_lambda = 10**(-0.4*(mag+21.1-0.602))
        if band == 'V': 
            f_lambda = 10**(-0.4*(mag+21.1+0.0))
        if band == 'R':       
            f_lambda = 10**(-0.4*(mag+21.1+0.555))
        if band == 'I':       
            f_lambda = 10**(-0.4*(mag+21.1+1.271))
        if band == 'J':      
            f_lambda = 10**(-0.4*(mag+21.1+2.655))
        if band == 'H':      
            f_lambda = 10**(-0.4*(mag+21.1+3.760))
        if band == 'K':      
            f_lambda = 10**(-0.4*(mag+21.1+4.906))
    #- AB system:
    if system == 'AB':
        """
        #- for swift filters:
        #- Poole et al. 2008, using this can also calibrate u to Johnson's U
        
        if band == 'uvw2':    
            f_lambda = 10**(-0.4*(mag+19.11))
        if band == 'uvw1':    
            f_lambda = 10**(-0.4*(mag+18.95))
            flux=f_lambda*fwhm
        """
        if band in ['U','B','V','R','I']:
            band='bessell-'+band
            #filt=speclite.filters.load_filter(band)
            #wavelength=filt.wavelength
            f_lambda=ABspectrum(wave,magnitude=mag) # f_lambda at filter effective wavelength
    ef_lambda=emag/2.5*f_lambda*np.log(10.0)
    return wave,f_lambda,ef_lambda

def mag_flux(mag,emag,band,system='Vega'): #- mag to flux conversion given a band

    filt=speclite.filters.load_filter('bessell-'+band)
    wavelength=filt.wavelength
    filtconv=speclite.filters.FilterConvolution(filt,wavelength,photon_weighted=False)

    if system=='Vega':    
        wave,f_lambda,ef_lambda=mag_flux_density(mag,emag,band,system=system)
        spec=np.ones_like(wavelength)*f_lambda #- f_lambda is averaged
        flux=filtconv(spec) #- integrated flux
         
    if system=='AB':
        wave=filt.effective_wavelength
        spec=ABspectrum(wavelength,magnitude=mag)
        flux=filtconv(spec) #- integrated flux

    eflux = emag/2.5*flux*np.log(10.0)

    return wave,flux,eflux

def effwavelength(band):
    wavemap={'U': 3618.4, 'B': 4442.25, 'V': 5536.15, 'R': 6648.33, 'I': 8086.40, 'J': 12200, 'H': 16300, 'K': 21900,
              'uvw2': 1991., 'uvw1': 2468.}
    fwhmmap={'U': 340., 'B': 780., 'V': 990., 'R': 1065., 'I': 2892}
               
    return wavemap[band],fwhmmap[band]

def count_bins(wave,band):
    #- number of bins in the spectrum within the filter range
    filt=speclite.filters.load_filter('bessell-'+band)
    wavelength=filt.wavelength
    bins=np.logical_and(wave>=np.min(wavelength),wave<=np.max(wavelength))
    #nbins=len(wave[bins])
    nbins=len(wavelength)
    return nbins

def ABspectrum(wavelength,magnitude=0.):
    """
    wavelength in Angstrom
    """
    #- https://en.wikipedia.org/wiki/AB_magnitude
    f_lambda= 10 ** (-0.4 * magnitude) * _abconst / wavelength ** 2
    return f_lambda



def find_ABmag(temp,band="bessell-R",normalization=1.):
    #- find the band magnitude of BB at the given temperature using Planck function

    filt=speclite.filters.load_filter(band)
    wave=filt.wavelength
    bb=planck(wave,temp,norm=normalization)
    mag=filt.get_ab_magnitude(wavelength=wave,spectrum=bbspec)
    return mag

def vega_to_AB(vegamag,band='R'):
    # Following Blanton 2007; http://www.astronomy.ohio-state.edu/~martini/usefuldata.html

    if band=='U':
        ABmag = vegamag+0.79
    if band=='B':
        ABmag = vegamag-0.09
    if band=='V':
        ABmag = vegamag+0.02
    if band=='R':
        ABmag = vegamag+0.21
    if band=='I':
        ABmag = vegamag+0.45
    if band=='J':
        ABmag = vegamag+0.91
    if band=='H':
        ABmag = vegamag+1.39
    if band=='K':
        ABmag = vegamag+1.85
    if band=='u':
        ABmag = vegamag+0.91
    if band=='g':
        ABmag = vegamag-0.08
    if band=='r':
        ABmag = vegamag+0.16
    if band=='i':
        ABmag = vegamag+0.37
    if band=='z':
        ABmag = vegamag+0.54

    return ABmag

def restframe_spec(wave,z):
    restwave=wave/(1+z)
    return restwave


def specphot(wave,spec,band,z=None):
    """
    find AB magnitude from the spectrum
    if z is given, find magnitude in rest frame
    """

    if z is not None:
        wave=wave/(1+z)
        
    filt=speclite.filters.load_filter(band)
    #wave=filt.wavelength
    #filtconv=speclite.filters.FilterConvolution(filt,wavelength,photon_weighted=False)
    #flux=filtconv(spec)
    mag=filt.get_ab_magnitude(wavelength=wave,spectrum=spec)
    return mag

def calibrotse2V(rmag,remag,temp):
    print rmag
    print temp
    model=-7.157+7.2346*(1-np.exp(-7.2579e-4*temp))
    rms=0.01
    mag=rmag-model
    emag=(remag**2+0.01**2)**0.5
    return mag,emag

def plot_lc(lcfile,A_v=0.,kcorrfile=None,snname='SN', mjd=0.,tempfile=None):
    #- plot light curves, mjd is not used in computation, only for legend 
    fig=plt.figure()
    ax=fig.add_subplot(111)

    #- load lc
    ep,mag,emag=np.loadtxt(lcfile,unpack=True)
    mag=mag-A_v #- extinction if not already done

    if kcorrfile is not None:
        kep,kcorr,ekcorr=np.loadtxt(kcorrfile,unpack=True)
        np.alltrue(ep==kep) #- make sure all epochs match
        mag=mag-kcorr
        emag=np.sqrt(emag**2+ekcorr**2)

    k=np.where(ep<400.)

    ep=ep[k]
    mag=mag[k]
    emag=emag[k]

    if tempfile is not None:
        from epm.temperature import sample_temp
        eptemp,meastemp,measetemp=np.loadtxt(tempfile,unpack=True)
        temp,etemp=sample_temp(ep,eptemp,meastemp,measetemp)
        mag,emag=calibrotse2V(mag,emag,temp)
    
    ylim=(np.max(mag)+1,np.min(mag)-1)
    ax.errorbar(ep,mag,yerr=emag,color='black',ls='None', marker='o', capsize=0)
    from matplotlib.font_manager import FontProperties
    font=FontProperties()
    font.set_style('italic')
    font.set_weight('bold')
    ax.text(0.7,0.95,snname,ha='left', va='top', transform=ax.transAxes,fontproperties=font, fontsize=22,alpha=2)
    ax.set_ylabel(r'${\rm V\ Magnitude}$',fontsize=24)
    ax.set_xlabel(r'${\rm Phase\ since\ MJD\ %.1f\ [days]} $'%mjd,fontsize=24)
    ax.axvline(0,linestyle='dashed',lw=2.)
    ax.set_ylim(ylim)
    ax.set_xlim(-5,np.max(ep)+5)
    ax.tick_params(axis='both',labelsize=16)
    plt.tight_layout()
    plt.savefig('sn{}_lightcurve.eps'.format(snname[-4:]))
    plt.show()
    return


