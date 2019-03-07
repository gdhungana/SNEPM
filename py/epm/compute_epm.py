import numpy as np
from util import dilution_factor
import photutil
import temperature
import speclite.filters
import scipy.constants as const

h=const.h *1.e7 #- ergs sec (easy in astropy 1.3.2 but using this unless upgraded)
c=const.c*1.e2 # cm/s
kB=const.k*1.e7 # erg deg-1

def get_bbflux(wave,temp,etemp):
    #- same as temperature.planck but gives out error also. Planck is used as fit function. this is used for flux and error.
    wave=wave*1e-8
    ind=h * c/(wave * kB * temp)
    B_lamb= 2*h * c**2/(wave**5 * (np.exp(ind)-1.)) 
    B_lamb*=1.0e-8 #- B_lamb(T) in ergs s-1 cm-2 A-1 sr -1; one factor to put in per A

    dB_lamb= 2*h*c**2/wave**5 * (np.exp(ind)-1.)**-2 * (ind)*(1/temp) *etemp 
    dB_lamb*=1.0e-8 

    return B_lamb, dB_lamb

