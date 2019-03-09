import numpy as np
from scipy import optimize as op
import emcee
import matplotlib.pyplot as plt
import corner
from epm.compute_epm import dist2distmodulus,cosmo

def make_data():
    m_true = -0.9594
    b_true = 4.294
    f_true = 0.534

    # Generate some synthetic data from the model.
    N = 50
    x = np.sort(10*np.random.rand(N))
    yerr = 0.1+0.5*np.random.rand(N)
    y = m_true*x+b_true
    y += np.abs(f_true*y) * np.random.randn(N)
    y += yerr * np.random.randn(N)

    return x,y,yerr

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
    if -5. < m < 0.5 and 0.0 < b < 10.0 and -10.0 < lnf < 1.0:
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
    ndim, nwalkers = 3, 1000
    result=optimize_like(x,y,yerr)
    print(result["x"])
    ml_m=result['x'][0]
    ml_b=result['x'][1]
    ml_lnf=result['x'][2]
    rst=np.random.RandomState(1234)
    pos = [result["x"] + 1.0e-4*rst.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr))
    sampler.run_mcmc(pos, 500)
    samples=sampler.chain[:,50:,:].reshape((-1,ndim))
    if save:
        fig1 = corner.corner(samples, truths=[ml_m,ml_b,ml_lnf],labels=["$m$", "$b$", "$\ln\,f$"],quantiles=[0.16,0.5,0.84],show_titles=True,title_kwargs={"fontsize": 14}) 
        fig1.savefig("test_param.eps")
        fig2=plt.figure()
        ax=fig2.add_subplot(111)
        for m,b,lnf in samples[np.random.randint(len(samples),size=100)]:
            ax.plot(x,m*x+b,color="gray",alpha=0.2)
        ax.plot(x,ml_m*x+ml_b,color="k",lw=2, alpha=0.8)
        ax.errorbar(x,y,yerr=yerr,fmt=".r",ls='None',capsize=0)
        #ax.set_yscale("log")
        #ax.set_ylim(28,40)
        #ax.set_xlim(0.001,0.1)
        #ax.set_xscale('log')
        fig2.savefig("test_sample.eps")
    return samples

