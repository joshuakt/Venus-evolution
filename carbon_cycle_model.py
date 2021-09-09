import numpy as np
from numba import jit

## Calculate silicate weathering flux, in kg/s
@jit(nopython=True)    
def weathering_flux(t0,CO2_Pressure_surface,SurfT,Tp,water_frac,T_efold,H2O_Pressure_surface,g,y12,alpha_exp,supp_lim,ocean_pH,omega_ocean,CP,MMW):
    if (CO2_Pressure_surface < 0.0):  # no CO2 left to be weathered.
        return 0.0
    if (water_frac<0.9999999)and(SurfT<647)and(H2O_Pressure_surface>0):  #weathering only occurs if below critical point
        infront= 4000 #coefficient chosen to reproduce approximate Earth fluxes
        Supply_limit = supp_lim 
        Te = T_efold 
        alpha = alpha_exp
        Ocean_depth = (0.018/MMW)*(1-water_frac) * H2O_Pressure_surface / (g*1000) ## max ocean depth continents 11.4 * gEarth/gplanet (COwan and Abbot 2014)
        Max_depth = 11400 * (9.8 / g) 
    
    
        if Ocean_depth > Max_depth:
            Ocean_fraction = 1.0
            Land_fraction = 0.0
            LF = 0.0
        else:
            Ocean_fraction = (Ocean_depth/Max_depth)**0.25 ## Approximation to Earth hypsometric curve
            Land_fraction = 1 - Ocean_fraction
            Land_fraction0 = 1 - (2.5/11.4)**0.25 
            LF = Land_fraction / Land_fraction0
    
        Tdeep = SurfT + Ocean_depth * 5e-4 * g * SurfT / 4000.0 # adiabatic lapse rate, won't make much difference

        Crustal_production = CP/360.0 
        Seafloor_weathering =(infront/4.0) * 10**(-0.3*(ocean_pH-7.727)) * Crustal_production * np.exp(-(90000.0/8.314)*(1.0/Tdeep-1.0/285))

        Continental_weathering =infront * LF * ((CO2_Pressure_surface/1e5)/(350e-6) )**alpha * np.exp((SurfT-285)/Te)
        Weathering = (Continental_weathering + Seafloor_weathering) * (1-water_frac) #water frac to stop weathering as dry out
        
        if Weathering>Supply_limit: #Weathering flux never exceeds supply limit
            return Supply_limit     
    else:
        Weathering = 0.0

    return Weathering
    
