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
