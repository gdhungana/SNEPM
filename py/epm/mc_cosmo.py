import numpy as np
from scipy import optimize as op
import emcee
import matplotlib.pyplot as plt
import corner
from epm.compute_epm import dist2distmodulus
#- use EMCEE package to perform cosmological workout: http://dfm.io/emcee/current/user/line/

def read_data():
    from epm.compute_epm import dist2distmodulus
    import numpy as np
    from epm.util import helio_to_cmb

    data=np.loadtxt('distance.txt',unpack=True)
    z=data[2]
    d=data[4]
    derr=data[5]

    ra=data[0]
    dec=data[1]
    zcmb=np.zeros_like(z)
    for ii in range(len(zcmb)):
        zcmb[ii]=helio_to_cmb(z[ii],ra[ii],dec[ii])

    dm,edm=dist2distmodulus(d,derr)
    return zcmb,dm,edm

class cosmoH0(object):
    #- solve for omega matter

    def __init__(self,seed=1234):
        self.name='CH0'
        self.seed=seed

    def model(self,z,H0):
        omegaM=0.3
        omegaL=0.7
        from compute_epm import cosmo
        dl=np.zeros_like(z)
        for ii in range(len(z)):
            dl[ii]= cosmo(z[ii],omegaM,omegaL,H0) #- Luminosity distance in Mpc
        dm = 5.*np.log10(dl)+25. #- dist in mega parsec
        return dm

    def lnlike(self,theta,z,dm,edm):
        H0,lnsig=theta
        thismodel=self.model(z,H0)
        inv_sigma2=1.0/(edm**2+np.exp(2*lnsig)) #- should not multiply by model as not a fraction.
        return -0.5*(np.sum((dm-thismodel)**2*inv_sigma2 - np.log(inv_sigma2)))

    def optimize_like(self,z,dm,edm):
        #- first get the ls solution for H0, and lnsig
        popt,pcov=op.curve_fit(self.model,self.z,self.dm)
        h0=popt[0]
        sig=0.3
        self.theta=[h0,np.log(sig)]
        nll=lambda *args: -self.lnlike(*args)
        result=op.minimize(nll,self.theta,args=(z,dm,edm))
        return result

    def lnprior(self,theta):
        H0, lnsig = theta
        if 50 < H0 < 85 and -10. < lnsig < 2.0:
            return 0.0
        return -np.inf

    def lnprob(self,theta,z,dm,edm):
        lp = self.lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.lnlike(theta,z,dm,edm)

    def run_emcee(self,z,dm,edm):
        self.z=z
        self.dm=dm
        self.edm=edm

        rst=np.random.RandomState(self.seed)
        ndim, nwalkers = 2, 500
        self.result=self.optimize_like(z,dm,edm)
        pos = [self.result["x"] + 1.0e-4*rst.randn(ndim) for i in range(nwalkers)]
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob, args=(self.z, self.dm, self.edm))
        sampler.run_mcmc(pos, 500)
        self.samples=sampler.chain[:,50:,:].reshape((-1,ndim))

    def plot_samples(self):
        fig2=plt.figure()
        ax=fig2.add_subplot(111)
        zz=np.array([0.0001,0.1])
        for h0,lnsig in self.samples[np.random.randint(len(self.samples),size=100)]:
            ax.plot(zz,self.model(zz,h0),color="gray",alpha=0.2)

        ml_h0=self.result['x'][0]
        ml_lnsig=self.result['x'][1]
        ax.plot(zz,self.model(zz,ml_h0),color="k",lw=2, alpha=0.8)
        ax.errorbar(self.z,self.dm,yerr=self.edm,fmt="ro",ls='None',capsize=0)
        ax.set_ylim(28,40)
        ax.set_xlim(0.001,0.1)
        ax.set_xscale('log')
        ax.set_xlabel(r"$z_{CMB}$",fontsize=20)
        ax.set_ylabel(r"$\mu$",fontsize=20)
        fig2.savefig("mu_vs_z_sample_{}.eps".format(self.name))
        plt.show()

    def plot_corner(self):
        ml_h0=self.result['x'][0]
        ml_lnsig=self.result['x'][1]
        reshape1=self.samples.T
        #newformat=np.array((reshape1[0],np.exp(reshape1[1]))).T
        fig1 = corner.corner(self.samples, truths=[ml_h0,ml_lnsig],labels=["$H_{0}$", "$\ln \sigma$"],quantiles=[0.16,0.5,0.84],show_titles=True,title_kwargs={"fontsize": 18}) 
        fig1.savefig("mu_z_corner_param_{}.eps".format(self.name))
        plt.show()


class cosmoOm(object):
    #- get Omegam, sigma
    def __init__(self,seed=1234):
        self.name='COm'
        self.seed=seed

    def model(self,z,omegaM):
        H0=70.
        omegaL=1-omegaM
        from compute_epm import cosmo
        dl=np.zeros_like(z)
        for ii in range(len(z)):
            dl[ii]= cosmo(z[ii],omegaM,omegaL,H0) #- Luminosity distance in Mpc
        dm = 5.*np.log10(dl)+25. #- dist in mega parsec
        return dm

    def lnlike(self,theta,z,dm,edm):
        omegaM,lnsig=theta
        thismodel=self.model(z,omegaM)
        inv_sigma2=1.0/(edm**2+np.exp(2*lnsig))
        return -0.5*(np.sum((dm-thismodel)**2*inv_sigma2 - np.log(inv_sigma2)))

    def optimize_like(self,z,dm,edm):
        #- first get the ls solution for H0, and lnsig
        popt,pcov=op.curve_fit(self.model,self.z,self.dm)
        om=popt[0]
        sig=0.3
        self.theta=[om,np.log(sig)]
        nll=lambda *args: -self.lnlike(*args)
        result=op.minimize(nll,self.theta,args=(z,dm,edm))
        return result

    def lnprior(self,theta):
        Om, lnsig = theta
        if 0.01 < Om < 0.9 and -10. < lnsig < 2.0:
            return 0.0
        return -np.inf

    def lnprob(self,theta,z,dm,edm):
        lp = self.lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.lnlike(theta,z,dm,edm)

    def run_emcee(self,z,dm,edm):
        self.z=z
        self.dm=dm
        self.edm=edm

        rst=np.random.RandomState(self.seed)
        ndim, nwalkers = 2, 500
        self.result=self.optimize_like(z,dm,edm)
        pos = [self.result["x"] + 1.0e-4*rst.randn(ndim) for i in range(nwalkers)]
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob, args=(self.z, self.dm, self.edm))
        sampler.run_mcmc(pos, 500)
        self.samples=sampler.chain[:,50:,:].reshape((-1,ndim))

    def plot_samples(self):
        fig2=plt.figure()
        ax=fig2.add_subplot(111)
        zz=np.array([0.0001,0.1])
        for om,lnsig in self.samples[np.random.randint(len(self.samples),size=100)]:
            ax.plot(zz,self.model(zz,om),color="gray",alpha=0.2)

        ml_om=self.result['x'][0]
        ml_lnsig=self.result['x'][1]
        ax.plot(zz,self.model(zz,ml_om),color="k",lw=2, alpha=0.8)
        ax.errorbar(self.z,self.dm,yerr=self.edm,fmt="ro",ls='None',capsize=0)
        ax.set_ylim(28,40)
        ax.set_xlim(0.001,0.1)
        ax.set_xscale('log')
        ax.set_xlabel(r"$z_{CMB}$",fontsize=20)
        ax.set_ylabel(r"$\mu$",fontsize=20)
        fig2.savefig("mu_vs_z_sample_{}.eps".format(self.name))
        plt.show()

    def plot_corner(self):
        ml_om=self.result['x'][0]
        ml_lnsig=self.result['x'][1]
        reshape1=self.samples.T
        #newformat=np.array((reshape1[0],np.exp(reshape1[1]))).T
        fig1 = corner.corner(self.samples, truths=[ml_om,ml_lnsig],labels=["$\Omega_{m}$", "$\ln \sigma$"],quantiles=[0.16,0.5,0.84],show_titles=True,title_kwargs={"fontsize": 18}) 
        fig1.savefig("mu_z_corner_param_{}.eps".format(self.name))
        plt.show()
        

class cosmoH0_Om(object):
    #- solve for omega matter

    def __init__(self,seed=1234):
        self.name='CH0_Om'
        self.seed=seed

    def model(self,z,H0,omegaM):
        omegaL=1.-omegaM
        from compute_epm import cosmo
        dl=np.zeros_like(z)
        for ii in range(len(z)):
            dl[ii]= cosmo(z[ii],omegaM,omegaL,H0) #- Luminosity distance in Mpc
        dm = 5.*np.log10(dl)+25. #- dist in mega parsec
        return dm

    def lnlike(self,theta,z,dm,edm):
        H0,Om,lnsig=theta
        thismodel=self.model(z,H0,Om)
        inv_sigma2=1.0/(edm**2+np.exp(2*lnsig))
        return -0.5*(np.sum((dm-thismodel)**2*inv_sigma2 - np.log(inv_sigma2)))

    def optimize_like(self,z,dm,edm):
        #- first get the ls solution for H0, and lnsig
        popt,pcov=op.curve_fit(self.model,self.z,self.dm)
        h0=popt[0]
        om=popt[1]
        sig=0.3
        self.theta=[h0,om,np.log(sig)]
        nll=lambda *args: -self.lnlike(*args)
        result=op.minimize(nll,self.theta,args=(z,dm,edm))
        return result

    def lnprior(self,theta):
        H0, Om, lnsig = theta
        if 50 < H0 < 85 and 0.01 < Om < 0.9 and -10. < lnsig < 2.0:
            return 0.0
        return -np.inf

    def lnprob(self,theta,z,dm,edm):
        lp = self.lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.lnlike(theta,z,dm,edm)

    def run_emcee(self,z,dm,edm):
        self.z=z
        self.dm=dm
        self.edm=edm

        rst=np.random.RandomState(self.seed)
        ndim, nwalkers = 3, 500
        self.result=self.optimize_like(z,dm,edm)
        pos = [self.result["x"] + 1.0e-4*rst.randn(ndim) for i in range(nwalkers)]
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob, args=(self.z, self.dm, self.edm))
        sampler.run_mcmc(pos, 500)
        self.samples=sampler.chain[:,50:,:].reshape((-1,ndim))

    def plot_samples(self):
        fig2=plt.figure()
        ax=fig2.add_subplot(111)
        zz=np.array([0.0001,0.1])
        for h0,om,lnsig in self.samples[np.random.randint(len(self.samples),size=100)]:
            ax.plot(zz,self.model(zz,h0,om),color="gray",alpha=0.2)

        ml_h0=self.result['x'][0]
        ml_om=self.result['x'][1]
        ml_lnsig=self.result['x'][2]
        ax.plot(zz,self.model(zz,ml_h0,ml_om),color="k",lw=2, alpha=0.8)
        ax.errorbar(self.z,self.dm,yerr=self.edm,fmt="ro",ls='None',capsize=0)
        ax.set_ylim(28,40)
        ax.set_xlim(0.001,0.1)
        ax.set_xscale('log')
        ax.set_xlabel(r"$z_{CMB}$",fontsize=20)
        ax.set_ylabel(r"$\mu$",fontsize=20)
        fig2.savefig("mu_vs_z_sample_{}.eps".format(self.name))
        plt.show()

    def plot_corner(self):
        ml_h0=self.result['x'][0]
        ml_om=self.result['x'][1]
        ml_lnsig=self.result['x'][2]
        reshape1=self.samples.T
        #newformat=np.array((reshape1[0],np.exp(reshape1[1]))).T
        fig1 = corner.corner(self.samples, truths=[ml_h0,ml_om,ml_lnsig],labels=["$H_{0}$", "$\Omega_{M}$", "$\ln\ \sigma$"],quantiles=[0.16,0.5,0.84],show_titles=True,title_kwargs={"fontsize": 18}) 
        fig1.savefig("mu_z_corner_param_{}.eps".format(self.name))
        plt.show()

"""
def lnlike(theta,x,y,yerr): #- Log likelihood. for a linear y=mx+b
    m,b,lnf=theta #- fraction f
    model=m*x+b
    inv_sigma2=1.0/(yerr**2+model**2*np.exp(2*lnf))
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))


def optimize_like(x,y,yerr):
    from scipy import optimize as op
    #- first get the ls solution for m,b,lnf
    fit=lambda x,m,b: m*x+b
    popt,pcov=op.curve_fit(fit,x,y)
    m=popt[0]
    b=popt[1]
    f=0.4
    nll=lambda *args: -lnlike(*args)
    result=op.minimize(nll,[m,b,np.log(f)],args=(x,y,yerr))
    m_ml,b_ml,lnf_ml=result["x"]
    return result
    
def lnprior(theta):
    m, b, lnf = theta
    if 35.0 < m < 45 and 30 < b < 40.0 and -10.0 < lnf < 1.0:
        return 0.0
    return -np.inf

def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)


def run_emcee(x,y,yerr,save=True):
    import corner
    import emcee
    import matplotlib.pyplot as plt
    ndim, nwalkers = 3, 5000
    result=optimize_like(np.log10(x),y,yerr)
    print(result["x"])
    ml_m=result['x'][0]
    ml_b=result['x'][1]
    ml_lnf=result['x'][2]
    rst=np.random.RandomState(1234)
    pos = [result["x"] + 1.0e-1*rst.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr))
    sampler.run_mcmc(pos, 500)
    samples=sampler.chain[:,50:,:].reshape((-1,ndim))
    if save:
        fig1 = corner.corner(samples, truths=[ml_m,ml_b,ml_lnf],labels=["$m$", "$b$", "$\ln\,f$"],quantiles=[0.16,0.5,0.84],show_titles=True,title_kwargs={"fontsize": 14}) 
        fig1.savefig("mu_z_corner_param.eps")
        zz=np.array([0.0001,0.1])
        fig2=plt.figure()
        ax=fig2.add_subplot(111)
        for m,b,lnf in samples[np.random.randint(len(samples),size=100)]:
            ax.plot(zz,m*np.log10(zz)+b,color="gray",alpha=0.2)
        ax.plot(zz,ml_m*np.log10(zz)+ml_b,color="k",lw=2, alpha=0.8)
        ax.errorbar(x,y,yerr=yerr,fmt="ro",ls='None')
        #ax.set_yscale("log")
        ax.set_ylim(28,40)
        ax.set_xlim(0.001,0.1)
        ax.set_xscale('log')
        fig2.savefig("mu_vs_z_sample.eps")
    return samples
"""
