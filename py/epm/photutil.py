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
