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

