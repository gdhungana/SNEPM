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


