import numpy as np
from scipy import optimize as op
import emcee
import matplotlib.pyplot as plt
import corner
#- use EMCEE package to perform cosmological workout: http://dfm.io/emcee/current/user/line/

class line1(object):

    def __init__(self,seed=1234):
        self.name='line1'
        self.seed=seed

    def lnlike(self,theta,x,y,yerr):
        m,b,lnf=theta
        model=m*x+b
        inv_sigma2=1.0/(yerr**2+model**2*np.exp(2*lnf))
        return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))

    def optimize_like(self,x,y,yerr):
        #- first get the ls solution for m,b,lnf
        fit=lambda t,m,b: m*t+b
        popt,pcov=op.curve_fit(fit,x,y)
        m=popt[0]
        b=popt[1]
        f=0.4
        self.theta=[m,b,np.log(f)]
        nll=lambda *args: -self.lnlike(*args)
        result=op.minimize(nll,self.theta,args=(x,y,yerr))
        return result

    def lnprior(self,theta):
        m, b, lnf = theta
        if -5. < m < 0.5 and 0.0 < b < 10.0 and -10.0 < lnf < 1.0:
            return 0.0
        return -np.inf

    def lnprob(self,theta,x,y,yerr):
        lp = self.lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.lnlike(theta,x,y,yerr)

    def run_emcee(self,x,y,yerr):
        self.x=x
        self.y=y
        self.yerr= yerr

        rst=np.random.RandomState(self.seed)
        ndim, nwalkers = 3, 1000
        self.result=self.optimize_like(x,y,yerr)
        pos = [self.result["x"] + 1.0e-4*rst.randn(ndim) for i in range(nwalkers)]
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob,args=(x, y, yerr))
        sampler.run_mcmc(pos, 500)
        self.samples=sampler.chain[:,50:,:].reshape((-1,ndim))

    def plot_samples(self):
        #zz=np.array([0.0001,0.1])
        fig2=plt.figure()
        ax=fig2.add_subplot(111)
        for m,b,lnf in self.samples[np.random.randint(len(self.samples),size=100)]:
            ax.plot(self.x,m*self.x+b,color="gray",alpha=0.2)

        ml_m=self.result['x'][0]
        ml_b=self.result['x'][1]
        ml_lnf=self.result['x'][2]
        ax.plot(self.x,ml_m*self.x+ml_b,color="k",lw=2, alpha=0.8)
        ax.errorbar(self.x,self.y,yerr=self.yerr,fmt=".r",ls='None')
        #ax.set_ylim(28,40)
        #ax.set_xlim(0.001,0.1)
        #ax.set_xscale('log')
        ax.set_xlabel(r"$z_{CMB}$",fontsize=20)
        ax.set_ylabel(r"$\mu$",fontsize=20)
        fig2.savefig("mu_vs_z_sample_{}.eps".format(self.name))
        plt.show()

    def plot_corner(self):
        ml_m=self.result['x'][0]
        ml_b=self.result['x'][1]
        ml_lnf=self.result['x'][2]
        fig1 = corner.corner(self.samples, truths=[ml_m,ml_b,ml_lnf],labels=["$m$", "$b$", "$\ln\,f$"],quantiles=[0.16,0.5,0.84],show_titles=True,title_kwargs={"fontsize": 14}) 
        fig1.savefig("mu_z_corner_param_{}.eps".format(self.name))
        plt.show()


class line2(object):
    """
    y=m*x model (no intercept)
    """
    def __init__(self,x,y,yerr):
        self.name='line1'
        self.x=x
        self.y=y
        self.yerr=yerr
        self.seed=1234

    def lnlike(self,theta):
        self.theta=theta
        m,lnf=self.theta
        model=m*self.x
        inv_sigma2=1.0/(self.yerr**2+model**2*np.exp(2*lnf))
        return -0.5*(np.sum((self.y-model)**2*inv_sigma2 - np.log(inv_sigma2)))

    def optimize_like(self):
        #- first get the ls solution for m,lnf
        fit=lambda x,m: m*x
        popt,pcov=op.curve_fit(fit,self.x,self.y)
        m=popt[0]
        f=0.4
        nll=lambda *args: -self.lnlike(*args)
        self.result=op.minimize(nll,[m,np.log(f)]) #,args=(self.x,self.y,self.yerr))
        self.maxlike=self.result["x"]
        return self.result

    def lnprior(self):
        m, lnf = self.theta
        if 35.0 < m < 45 and -10.0 < lnf < 1.0:
            return 0.0
        return -np.inf

    def lnprob(self,theta):
        lp = self.lnprior()
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.lnlike(theta)

    def run_emcee(self):
        rst=np.random.RandomState(self.seed)
        ndim, nwalkers = 2, 5000
        self.optimize_like()
        pos = [self.result["x"] + 1.0e-1*rst.randn(ndim) for i in range(nwalkers)]
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob)#, args=(self.x, self.y, self.yerr))
        sampler.run_mcmc(pos, 500)
        self.samples=sampler.chain[:,50:,:].reshape((-1,ndim))

    def plot_samples(self):
        zz=np.array([0.0001,0.1])
        fig2=plt.figure()
        ax=fig2.add_subplot(111)
        for m,lnf in self.samples[np.random.randint(len(self.samples),size=100)]:
            ax.plot(zz,m*np.log10(zz),color="gray",alpha=0.2)

        ml_m=self.result['x'][0]
        ml_lnf=self.result['x'][1]
        ax.plot(zz,ml_m*np.log10(zz),color="k",lw=2, alpha=0.8)
        ax.errorbar(10**(self.x),self.y,yerr=self.yerr,fmt="ro",ls='None')
        ax.set_ylim(28,40)
        ax.set_xlim(0.001,0.1)
        ax.set_xscale('log')
        ax.set_xlabel(r"$z_{CMB}$",fontsize=20)
        ax.set_ylabel(r"$\mu$",fontsize=20)
        fig2.savefig("mu_vs_z_sample_{}.eps".format(self.name))
        plt.show()

    def plot_corner(self):
        ml_m=self.result['x'][0]
        ml_lnf=self.result['x'][1]
        fig1 = corner.corner(self.samples, truths=[ml_m,ml_lnf],labels=["$m$", "$b$", "$\ln\,f$"],quantiles=[0.16,0.5,0.84],show_titles=True,title_kwargs={"fontsize": 14}) 
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
