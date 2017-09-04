import numpy as np
from sklearn.mixture import GMM
import matplotlib.pyplot as plt
from astroML.plotting import hist
from desispec.interpolation import resample_flux


def estimate_continuum(wave,flux,mask=True,minmask=None,maxmask=None,deg=1,flat=True,spline=False,nknots=None):
    """
    perform polynomial fit for local continuum subtraction
    """ 

    if flat:
        continuum=np.min(flux)+np.zeros(wave.shape[0])
        return continuum
    else:
        ww=wave
        ff=flux
        if mask is True:
            if minmask is not None and maxmask is not None:
                kk = np.where((wave > minmask) & (wave < maxmask))
                ww=np.delete(wave,kk)
                ff=np.delete(flux,kk)
        print "No of points for estimating continuum",ww.shape[0], 'of', wave.shape[0]
        if spline:
            from scipy.interpolate import UnivariateSpline as spl
            if nknots is None:
                nknots=len(ww)
            spfit=spl(ww,ff,s=nknots) #knots downsample by a factor of 10
            continuum=spfit(wave)
        else: #- poly fit
            coeff=np.polyfit(ww,ff,deg) #- first order, deg=1
            continuum = np.polyval(coeff,wave)
    return continuum

def sample_array(array,size,seed=0):
    """
    randomly sample array and return values and indices
    Note: no random sampling of indices can be done as that breaks up the density estimate
    """
    np.random.seed(seed)
    values=np.random.choice(array,size=size, replace = False)
    np.random.seed(seed)
    indices=np.random.choice(len(array),size=size,replace = False)
    
    if np.array_equal(values,array[indices]):
        
        return values, indices
    else:
        raise ValueError("No array matching!. Chose same seed")


def sample_below(xarray,yarray,size=20000,inter_mode='quadratic',seed=0):
    from scipy.interpolate import interp1d
    #- generate uniform random points
    np.random.seed(seed)
    xpts=np.random.uniform(np.min(xarray),np.max(xarray),size)
    ypts=np.random.uniform(np.min(yarray),np.max(yarray),size)
    
    #- model
    f = interp1d(xarray,yarray,kind=inter_mode)
    select=np.where(ypts < f(xpts))
    return xpts[select], ypts[select]

def downsample_spec(wave,flux,factor=2):
    downw=wave[::factor]
    downsp=resample_flux(downw,wave,flux)
    return downw,downsp

def gmm_model(wave,flux,line_rest=5169.,invert = False, locut=300,hicut=300,minmask=None,maxmask=None,vplot=False,z=0,index=None,flat=True,ncomp=10,deg=1,spline=False,nknots=None,shift=0):
    """ 
    wave should be rest frame
    invert=True for absorption
    hicut/locut: cut for fit region in wavelength(A)
    line_rest: rest wavelength of line to find the velocity.
    vplot: plot velocity plot after knowing which the min/max is, needs index
    index: which Gaussian is this?
    """

    #- set the data first for fitting:
    if invert:
        flux=flux-np.max(flux)*1.02
        flux=flux*-1
    kk = np.where(((line_rest*(1+z) - locut) <= wave) & (wave <= line_rest*(1+z)+hicut))
    wwave=wave[kk]
    fflux=flux[kk]
    continuum = estimate_continuum(wwave,fflux,minmask=minmask,maxmask=maxmask,deg=deg,flat=flat,spline=spline,nknots=nknots)

    fflux-= continuum
    #nflux=fflux[fflux>0]
    #nwave=wwave[fflux>0]
    normflux=fflux.clip(0)/np.trapz(fflux.clip(0),wwave)
    downw,downf=downsample_spec(wwave,normflux,factor=5)
    plt.show()
    
    #- sample the pdf
    #sel=np.where(wwave < line_rest*(1+z))[0]
    #xx, yy=sample_below(wwave[sel], normflux[sel])
    xx,yy=sample_below(wwave,normflux)
    xx=xx-shift
    N=np.arange(1,ncomp)
    models=[None for i in range(len(N))]
    plt.plot(wwave,normflux,'r')
    plt.plot(xx,yy, 'g.')
    plt.show()  
    #plt.plot(xx,yy,'g.')
    plt.plot(downw,downf,ls='steps')
    nn=len(downw)
    print nn
    plt.hist(xx,nn-1,normed=True,histtype='stepfilled',alpha=0.4)
    #if invert: 
    #    plt.invert_yaxis()
    plt.show()
    for i in range(len(N)):
        models[i]=GMM(N[i],covariance_type='full').fit(xx)
    
    #- Compute AIC and BIC
    AIC = [m.aic(xx)/100 for m in models]
    BIC = [m.bic(xx)/100 for m in models]

    #- Plot the results
    fig=plt.figure()
    ax0=fig.add_subplot(221)
    ax0.plot(wave,flux,'b')
    ax0.plot(wwave,flux[kk],'g', label = 'Region of interest')
    ax0.plot(wwave,continuum,color='r',ls='dashed', label='Continuum',linewidth=2)
    if invert:
        ax0.invert_yaxis()
    ax0.tick_params(axis='both',labelsize=14)
    ax0.set_xlabel('Rest Wavelength',fontsize=15)
    ax0.set_ylabel('Scaled flux',fontsize=15)
    ax0.legend(fontsize=15)
    
    #- plot best model
    ax1=fig.add_subplot(222)
    M_best=models[np.argmin(BIC)]
    x=np.sort(xx)
    logprob,resp=M_best.eval(x)
    pdf=np.exp(logprob)
    pdf_individual = resp*pdf[:,np.newaxis]
    ax1.plot(downw,downf,ls='steps',alpha=1)
    plt.hist(x,nn-1,normed=True,histtype='stepfilled',alpha=0.4)
    ax1.plot(x,pdf,'-k')
    ax1.plot(x,pdf_individual, '--k')
    ax1.set_xlabel('Rest Wavelength',fontsize=15)
    ax1.set_ylabel('Normalized flux',fontsize=15)
    ax1.axvline(line_rest*(1+z),ls='dashed',color='r',linewidth=2.)
    if invert:
        ax1.text(0.04, 0.1, "Best-fit Mixture",
            ha='left', va='top', transform=ax1.transAxes,fontsize=15)
        ax1.invert_yaxis()
    else: 
        ax1.text(0.04, 0.96, "Best-fit Mixture",
            ha='left', va='top', transform=ax1.transAxes,fontsize=15)
    ax1.tick_params(axis='both',labelsize=14)
    
    #ax1.xaxis.set_major_locator(plt.MultipleLocator(1))
    #ax1.xaxis.set_major_formatter(plt.FormatStrFormatter('%i'))
    #- plot velocity
    vthis=None
    dvthis=None
    if vplot:
        ax=fig.add_subplot(223)
        xrest=(z+1)*line_rest
        v=(x-xrest)/xrest * 2.99e5 #- km/s

        #- estimate the closest component to max pdf
        if index is None:
            pdf_ind_max=np.max(pdf_individual,axis=0)
            print pdf_ind_max
            pdfs_peakat=[]
            for ii in range(len(pdf_ind_max)):
                peak=np.where(pdf_individual[:,ii]==pdf_ind_max[ii])[0]
                pdfs_peakat.append(x[peak])
            pdf_maxat = x[pdf==np.max(pdf)].tolist()
            pdfs_peakat=np.concatenate(pdfs_peakat[:]).tolist()
            print "Pdf indviduals peakat ", pdfs_peakat
            print "Pdf max at ", pdf_maxat
            diff= np.array(pdf_maxat)-np.array(pdfs_peakat)
            print diff
            restdiff=np.array(xrest)-np.array(pdfs_peakat)
            print restdiff
            print diff[restdiff>0]
            index=np.argmin(np.abs(diff))
            #index=np.argmin((np.array(pdf_maxat)-np.array(pdfs_peakat))) #- blue shift 
            print "Estimated best index:", index
        ncounts=count_sample_points(M_best,index,wwave)
        print "Ncounts",ncounts
        xobs=M_best.means_[index][0]
        print "Mean at", xobs
        dx=np.sqrt(M_best.covars_[index][0])
        #vthis=(M_best.means_[index]-xrest)/xrest*2.998e5
        #dvthis=dx/xrest*2.998e5
        vthis=(xobs**2-xrest**2)/(xobs**2+xrest**2) *2.998e5  #- relativistic doppler velocity
        dvthis=2.*xobs*dx/(xobs**2+xrest**2)*(1.-2.*(xobs**2-xrest**2)/(xobs**2+xrest**2)) * 2.998e5
        dvthis/=np.sqrt(ncounts) # error on the mean
        #dvthis=np.sqrt(dvthis**2+(shift/2./line_rest*2.998e5)**2)
        ax.plot((downw-xrest)/xrest* 2.998e5,downf,ls='steps')
        ax.axvline(0,color='r',ls='dashed',linewidth=2.)
        ax.axvline(vthis,color='r',label="$\lambda %s : %0d \pm %0d km/s$"%(line_rest,vthis,dvthis),linewidth=2.)
        #ss=np.argmax(downf)
        #ax.axvline(((downw-xrest)/xrest* 2.998e5)[ss],color='b',linewidth=0.5)
        ax.set_xlabel('velocity [km/s]',fontsize=16)
        ax.set_ylabel('Normalized flux',fontsize=16)
        ax.plot((x-xrest)/xrest* 2.998e5,pdf_individual[:,index], '--k',label='Best Fit component PDF')
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.002))
        ax.tick_params(axis='both',labelsize=14)
        ax.legend(loc=4)
        print "Velocity of $\lambda %s : %0d \pm %0d km/s$"%(line_rest,vthis,dvthis)
        if invert:
            ax.invert_yaxis()
    # plot 2: AIC and BIC
    ax2 = fig.add_subplot(224)
    ax2.plot(N, AIC, '-k', label='AIC')
    ax2.plot(N, BIC, '--k', label='BIC')

    ax2.set_xlabel('n. components', fontsize=16)
    ax2.set_ylabel('information criterion/100',fontsize=16)
    ax2.yaxis.set_major_locator(plt.MultipleLocator(10))
    ax2.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax2.xaxis.set_major_formatter(plt.FormatStrFormatter('%i'))
    ax2.legend(loc=1)
    ax2.tick_params(axis='both',labelsize=14)
    plt.show()
    return M_best,vthis,dvthis

def count_sample_points(model,index,wave):
    """
    Find number of points below the model[index] probability inside 3 sigma in the sample
    """
    sigma=np.sqrt(model.covars_[index][0])
    mean=model.means_[index][0]
    print mean
    print sigma
    hilim=mean+3*sigma
    lowlim=mean-3*sigma
    valid=np.where((wave> lowlim) & (wave<hilim))[0]
    ncount=valid.shape[0]
    print "no. of effective sample points", ncount
    return ncount #- no of points under the pdf
    

def get_v50(line,epoch,velocity,evelocity):

    """
    Eg:
    In [1]: from epm.velocity import get_v50,vHbetatovFeII,vph50tovFe
    In [2]: v50_hbeta,ev50_hbeta=get_v50("Hbeta",18.,10350,500)
    In [3]: vFeII50,evFeII50=vHbetatovFeII(v50_hbeta,ev50_hbeta)
    In [4]: vFeII50,evFeII50
    Out[4]: (4853.111318324523, 476.51914312015322)
    In [5]: vph,evph=vph50tovFe(vFeII50,evFeII50,18.)
    In [6]: vph,evph
    Out[6]: (8786.350018947107, 977.80997015993751)
    In [7]: vFeII18,evFeII18=vHbetatovFeII(10350,500)
    #- cross check direct transform
    In [8]: vFeII18,evFeII18
    Out[8]: (8331.75, 405.81314973765944)

    """
    if line not in ['Hbeta','FeII5169','Halpha','HeI5876']:
        raise ValueError("Line can be either of Hbeta,FeII5169 or Halpha")
    #- measure the velocity at 50 days from Faran et. al 2014: https://arxiv.org/pdf/1404.0378.pdf (fig 16)

    if line=="Hbeta":
        v50=velocity*(50./epoch)**(-0.529)
        evsq=(evelocity*(50./epoch)**(-0.529))**2+(0.027*velocity*np.log(50/epoch)*(50./epoch)**(-0.529))**2
        ev50=np.sqrt(evsq)
        v50,ev50=vHbetatovFeII(v50,ev50)

    if (line=='FeII5169') or (line=='HeI5876'):
        v50=velocity*(50./epoch)**(-0.581)
        evsq=(evelocity*(50./epoch)**(-0.581))**2+(0.034*velocity*np.log(50/epoch)*(50./epoch)**(-0.581))**2
        ev50=np.sqrt(evsq)
        
    if line=="Halpha":
        v50=velocity*(50./epoch)**(-0.412)
        evsq=(evelocity*(50./epoch)**(-0.412))**2+(0.020*velocity*np.log(50/epoch)*(50./epoch)**(-0.412))**2
        ev50=np.sqrt(evsq)
        v50,ev50=vHalphatovFeII(v50,ev50)
    
    return v50,ev50  #- Photospheric velocity for day 50

def vHbetatovFeII(vHbeta,evHbeta):
    vFeII=0.805*vHbeta # Following Faran 2014
    evFeII=np.sqrt((vHbeta*0.005)**2+evHbeta**2)
    return vFeII,evFeII

def vHalphatovFeII(vHalpha,evHalpha):
    vFeII = 0.855*vHalpha-1499.
    evFeII = (evHalpha**2+(vHalpha*0.006)**2+87**2)**0.5
    return vFeII,evFeII

def vph50tovFe(vph50,evph50,epoch):
    vFeII=vph50*(epoch/50.)**(-0.581)
    evFeII=evph50*(epoch/50.)**(-0.581)
    return vFeII,evFeII    

def sample_velocity(epochs,measep,measvel,measevel,line=['FeII5169'],save=False,plot=False,snname=None,mjd=None):

    if type(measep) in [float,np.float64]:
        measep=np.array([measep])
        measvel=np.array([measvel])
        measevel=np.array([measevel])

    measvel=np.sqrt(measvel**2)
    nspec=len(measep)
    nphot=len(epochs)
    vel=np.zeros((nphot,nspec))
    evel=np.zeros((nphot,nspec))
    
    if len(line)==1:
            
        newline=np.chararray(len(measep),itemsize=len(line[0]))
        newline[:]=line[0]
    else:
        newline=line

    for ii in range(nspec):
        v50,ev50=get_v50(newline[ii],measep[ii],measvel[ii],measevel[ii])
        print("Velocity at 50 day using line {}-{}: {} +/- {}".format(newline[ii],measep[ii],v50,ev50))
        vel[:,ii],evel[:,ii]=vph50tovFe(v50,ev50,epochs)

    #-Averaging
    meanvel=np.zeros(nphot) #- no of epoch
    meanevel=np.zeros(nphot)
        
    for jj in range(nphot):
        if nspec>1:
            """
            meanvel[jj]=np.average(vel[jj,:])
            meanevel[jj]=np.average(evel[jj,:])#**2+np.std(vel[jj,:])**2)**0.5
            """
            meanvel[jj]=np.average(vel[jj,:],weights=1/evel[jj,:]**2)
            #- variance: https://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy
            #- https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_variance
    
            variance=np.average((vel[jj,:]-meanvel[jj])**2, weights=1/evel[jj,:]**2)
            meanevel[jj]=np.sqrt(variance/nspec)
                   
        else:
            meanvel[jj]=vel[jj,:]
            meanevel[jj]=np.sqrt(evel[jj,:]**2)
     
    if save:
        print("Saving mean vel file")
        np.savetxt('final_meanvel.txt',np.c_[epochs,meanvel,meanevel],fmt='%.2f')
    
    if plot:
        from matplotlib.font_manager import FontProperties
        font=FontProperties()
        font.set_style('italic')
        font.set_weight('bold')
        fig=plt.figure()
        ax=fig.add_subplot(111)
        #- sample even more
        ep=np.linspace(5,70,26)
        
        textmap={'Hbeta':r'$H_{\beta}$','HeI5876':r'$HeI\ 5876$','FeII5169':r'$FeII\ 5169$','Halpha':r'$H_{\alpha}$'}
        
        ax.errorbar(measep,measvel/1.0e3,yerr=measevel/1.0e3,color='b',marker='o',capsize=0,ls='None',label='Measured  Vel.')
        ax.errorbar(epochs,meanvel/1.0e3,yerr=meanevel/1.0e3,color='r',marker='o',label='Derived Vel.',ls='None',capsize=0)
        for ii in range(len(measep)):
            ax.text(measep[ii]+1,0.3+measvel[ii]/1.0e3,textmap[newline[ii]],fontsize=18, alpha=2)
            print "New line:", newline[ii]
            
            nv50,env50=get_v50(newline[ii],measep[ii],measvel[ii],measevel[ii])
            nvel,nevel=vph50tovFe(nv50,env50,ep)
            if ii ==0:
                ax.plot(ep,nvel/1.0e3,'k--',lw=1,alpha=1,label='Derived Model')
            else:
                ax.plot(ep,nvel/1.0e3,'k--',lw=1,alpha=1,label='')

        #ax.fill_between(ep,(nT-neT)/1.0e3,(nT+neT)/1.0e3,alpha=0.3)
        ax.set_xlabel(r'${\rm Phase\ since\ MJD\ %7s\ [days]}$'%mjd,fontsize=20)
        ax.set_ylabel(r'${\rm Photospheric Vel. (v_{ph})\ [kkm/s]}$',fontsize=20)
        ax.tick_params(axis='both',labelsize=16)
        ax.legend(numpoints=1,fontsize=18)
        if snname is not None:
            ax.text(0.05,0.95,snname,ha='left', va='top', transform=ax.transAxes,fontproperties=font,fontsize=18, alpha=2)
        plt.tight_layout()
        print("Saving vel plot")
        plt.savefig("Vel_evolve.eps")
        plt.show()
        
    return meanvel,meanevel
