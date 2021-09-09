import numpy as np
       
def AB_fun(Tsurf,pH2O,volatile_mass,AL,AH):
    if pH2O/1e5<1:
        TA = 1000.0
    else:
        TA = 1000 + 200*np.log10(pH2O/1e5)**2 ## Puriel plot approximate
    
    AB_atmo = 0.5*(AL-AH) * np.tanh((TA-Tsurf)/400.0) + 0.5*(AH+AL) # pluriel
    return AB_atmo


