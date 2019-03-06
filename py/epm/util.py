import numpy as np
import speclite
import math

def dilution_factor(temp,etemp=None,bandset='BVI'):
    #- Following Dessart and Hillier 2005; table 1
    #- temp (color temp) in K
    if bandset=='BVI':
        xi=0.63241-0.38375*(10000./temp)+0.28425*(10000./temp)**2
        if etemp is not None:
            exi=etemp*(0.38375*10000./temp**2 + 0.28425*10000.*2/temp**3)
        else: exi=0.
    else:
        raise ValueError("No other combinations implemented")
    return xi,exi

def filter_response(responsefile=None):
     
    #- load rotse and sdss filters
    
    filters=speclite.filters.load_filters(responsefile,'sdss2010-*')
    #rotse=speclite.filters.load_filters('./rotse_response.ecsv')       

    speclite.filters.plot_filters(filters, wavelength_limits=(3000,11000))
    #speclite.filters.plot_filters(sdss,wavelength_limits=(3000,11000))

    plt.savefig("filter_response.png")

def rotse_response():
    rotse_resp=speclite.filters.load_filter('../../data/rotse_response_normalized.ecsv')
    return rotse_resp

def radec_to_xyz(ra, dec):
    x = math.cos(np.deg2rad(dec)) * math.cos(np.deg2rad(ra))
    y = math.cos(np.deg2rad(dec)) * math.sin(np.deg2rad(ra))
    z = math.sin(np.deg2rad(dec))

    return np.array([x, y, z], dtype=np.float64)


def cmb_dz(ra, dec):
    """See http://arxiv.org/pdf/astro-ph/9609034
     CMBcoordsRA = 167.98750000 # J2000 Lineweaver
     CMBcoordsDEC = -7.22000000
    """

    # J2000 coords from NED
    CMB_DZ = 371000. / 299792458.
    CMB_RA = 168.01190437
    CMB_DEC = -6.98296811
    CMB_XYZ = radec_to_xyz(CMB_RA, CMB_DEC)

    coords_xyz = radec_to_xyz(ra, dec)
    
    dz = CMB_DZ * np.dot(CMB_XYZ, coords_xyz)

    return dz

def helio_to_cmb(z, ra, dec):
    """Convert from heliocentric redshift to CMB-frame redshift.
    
    Parameters
    ----------
    z : float
        Heliocentric redshift.
    ra, dec: float
        RA and Declination in degrees (J2000).
    """

    dz = -cmb_dz(ra, dec)
    one_plus_z_pec = math.sqrt((1. + dz) / (1. - dz))
    one_plus_z_CMB = (1. + z) / one_plus_z_pec

    return one_plus_z_CMB - 1.

def cmb_to_helio(z, ra, dec):
    """Convert from CMB-frame redshift to heliocentric redshift.
    
    Parameters
    ----------
    z : float
        CMB-frame redshift.
    ra, dec: float
        RA and Declination in degrees (J2000).
    """

    dz = -cmb_dz(ra, dec)
    one_plus_z_pec = math.sqrt((1. + dz) / (1. - dz))
    one_plus_z_helio = (1. + z) * one_plus_z_pec

    return one_plus_z_helio - 1.

def unred(wave, ebv, R_V=3.1, LMC2=False, AVGLMC=False):
	"""
         Imported directly from http://projects.ift.uam-csic.es/skies-universes/pySU/doc/_build/html/_modules/firefly_dust.html
	 Deredden a flux vector using the Fitzpatrick (1999) parameterization
 
	 Parameters
	 ----------
	 wave :   array
			  Wavelength in Angstrom
	 flux :   array
			  Calibrated flux vector, same number of elements as wave.
	 ebv  :   float, optional
			  Color excess E(B-V). If a negative ebv is supplied,
			  then fluxes will be reddened rather than dereddened.
			  The default is 3.1.
	 AVGLMC : boolean
			  If True, then the default fit parameters c1,c2,c3,c4,gamma,x0 
			  are set to the average values determined for reddening in the 
			  general Large Magellanic Cloud (LMC) field by
			  Misselt et al. (1999, ApJ, 515, 128). The default is
			  False.
	 LMC2 :   boolean
			  If True, the fit parameters are set to the values determined
			  for the LMC2 field (including 30 Dor) by Misselt et al.
			  Note that neither `AVGLMC` nor `LMC2` will alter the default value 
			  of R_V, which is poorly known for the LMC.
   
	 Returns
	 -------             
	 new_flux : array 
				Dereddened flux vector, same units and number of elements
				as input flux.
 
	 Notes
	 -----

	 .. note:: This function was ported from the IDL Astronomy User's Library.

	 :IDL - Documentation:
 
	  PURPOSE:
	   Deredden a flux vector using the Fitzpatrick (1999) parameterization
	  EXPLANATION:
	   The R-dependent Galactic extinction curve is that of Fitzpatrick & Massa 
	   (Fitzpatrick, 1999, PASP, 111, 63; astro-ph/9809387 ).    
	   Parameterization is valid from the IR to the far-UV (3.5 microns to 0.1 
	   microns).    UV extinction curve is extrapolated down to 912 Angstroms.

	  CALLING SEQUENCE:
		FM_UNRED, wave, flux, ebv, [ funred, R_V = , /LMC2, /AVGLMC, ExtCurve= 
						  gamma =, x0=, c1=, c2=, c3=, c4= ]
	  INPUT:
		 WAVE - wavelength vector (Angstroms)
		 FLUX - calibrated flux vector, same number of elements as WAVE
				  If only 3 parameters are supplied, then this vector will
				  updated on output to contain the dereddened flux.
		 EBV  - color excess E(B-V), scalar.  If a negative EBV is supplied,
				  then fluxes will be reddened rather than dereddened.

	  OUTPUT:
		 FUNRED - unreddened flux vector, same units and number of elements
				  as FLUX

	  OPTIONAL INPUT KEYWORDS
		  R_V - scalar specifying the ratio of total to selective extinction
				   R(V) = A(V) / E(B - V).    If not specified, then R = 3.1
				   Extreme values of R(V) range from 2.3 to 5.3

	   /AVGLMC - if set, then the default fit parameters c1,c2,c3,c4,gamma,x0 
				 are set to the average values determined for reddening in the 
				 general Large Magellanic Cloud (LMC) field by Misselt et al. 
				 (1999, ApJ, 515, 128)
		/LMC2 - if set, then the fit parameters are set to the values determined
				 for the LMC2 field (including 30 Dor) by Misselt et al.
				 Note that neither /AVGLMC or /LMC2 will alter the default value 
				 of R_V which is poorly known for the LMC. 
			
		 The following five input keyword parameters allow the user to customize
		 the adopted extinction curve.    For example, see Clayton et al. (2003,
		 ApJ, 588, 871) for examples of these parameters in different interstellar
		 environments.

		 x0 - Centroid of 2200 A bump in microns (default = 4.596)
		 gamma - Width of 2200 A bump in microns (default  =0.99)
		 c3 - Strength of the 2200 A bump (default = 3.23)
		 c4 - FUV curvature (default = 0.41)
		 c2 - Slope of the linear UV extinction component 
			  (default = -0.824 + 4.717/R)
		 c1 - Intercept of the linear UV extinction component 
			  (default = 2.030 - 3.007*c2
	"""
        import scipy.interpolate as interpolate

	x = 10000./ wave # Convert to inverse microns 
	curve = x*0.

	# Set some standard values:
	x0 = 4.596
	gamma =  0.99
	c3 =  3.23      
	c4 =  0.41    
	c2 = -0.824 + 4.717/R_V
	c1 =  2.030 - 3.007*c2

	if LMC2:
		x0    =  4.626
		gamma =  1.05   
		c4   =  0.42   
		c3    =  1.92      
		c2    = 1.31
		c1    =  -2.16
	elif AVGLMC:   
		x0 = 4.596  
		gamma = 0.91
		c4   =  0.64  
		c3    =  2.73      
		c2    = 1.11
		c1    =  -1.28

	# Compute UV portion of A(lambda)/E(B-V) curve using FM fitting function and 
	# R-dependent coefficients
	xcutuv = np.array([10000.0/2700.0])
	xspluv = 10000.0/np.array([2700.0,2600.0])

	iuv = np.where(x >= xcutuv)[0]
	N_UV = len(iuv)
	iopir = np.where(x < xcutuv)[0]
	Nopir = len(iopir)
	if (N_UV > 0): xuv = np.concatenate((xspluv,x[iuv]))
	else:  xuv = xspluv

	yuv = c1  + c2*xuv
	yuv = yuv + c3*xuv**2/((xuv**2-x0**2)**2 +(xuv*gamma)**2)
	yuv = yuv + c4*(0.5392*(np.maximum(xuv,5.9)-5.9)**2+0.05644*(np.maximum(xuv,5.9)-5.9)**3)
	yuv = yuv + R_V
	yspluv  = yuv[0:2]  # save spline points
 
	if (N_UV > 0): curve[iuv] = yuv[2::] # remove spline points

	# Compute optical portion of A(lambda)/E(B-V) curve
	# using cubic spline anchored in UV, optical, and IR
	xsplopir = np.concatenate(([0],10000.0/np.array([26500.0,12200.0,6000.0,5470.0,4670.0,4110.0])))
	ysplir   = np.array([0.0,0.26469,0.82925])*R_V/3.1 
	ysplop   = np.array((np.polyval([-4.22809e-01, 1.00270, 2.13572e-04][::-1],R_V ), 
			np.polyval([-5.13540e-02, 1.00216, -7.35778e-05][::-1],R_V ), 
			np.polyval([ 7.00127e-01, 1.00184, -3.32598e-05][::-1],R_V ), 
			np.polyval([ 1.19456, 1.01707, -5.46959e-03, 7.97809e-04, -4.45636e-05][::-1],R_V ) ))
	ysplopir = np.concatenate((ysplir,ysplop))

	if (Nopir > 0): 
	  tck = interpolate.splrep(np.concatenate((xsplopir,xspluv)),np.concatenate((ysplopir,yspluv)),s=0)
	  curve[iopir] = interpolate.splev(x[iopir], tck)

	#Now apply extinction correction to input flux vector
	curve *= ebv

	return 10.**(0.4*curve)
