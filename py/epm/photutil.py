import numpy as np
import speclite.filters
import matplotlib.pyplot as plt
import scipy.constants as const
import astropy.units as u


hc=const.h*1.0e7*const.c*1.0e2 #- ergs s
jy=1.0e-23 #- ergs/s/Hz/cm2
_abconst= 3631*jy*const.c*1.0e2 *1.0e8#- ergs/s/cm2/Hz * A/s


