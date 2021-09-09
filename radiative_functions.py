##################################
import numpy as np
import pylab
from scipy.integrate import *
from scipy.interpolate import interp1d
from scipy import optimize
import pdb
import scipy.interpolate
from numba import jit
import time
#################################

   

### H2O and CO2 ranges for OLR grid
P_H2O_grid_new = np.logspace(1,9,34) 
P_CO2_grid_new = np.logspace(1,8,24)


########### OLR grid used for Tstrat sensitivity test ####
OLR_hybrid_FIX=np.load("OLR_200_FIX_flat.npy")
OLR_hybrid_FIX = np.log10(OLR_hybrid_FIX)
water_frac_multi_new=np.load("Atmo_frac_200_FIX_flat.npy")
fH2O_new = np.load("fH2O_200_FIX_flat.npy")
# dont use k=0 for anything, and dont use k=1 for OLR (ok for fH2O though)
T_surf_grid = np.linspace(274,4000,200)
Te_grid = np.linspace(150,350,8)
##############################################################

########### OLR grid used for nominal model ##############
OLR_hybrid_FIX=np.load("OLR_200_FIX_cold.npy")
OLR_hybrid_FIX = np.log10(OLR_hybrid_FIX)
water_frac_multi_new=np.load("Atmo_frac_200_FIX_cold.npy")
fH2O_new = np.load("fH2O_200_FIX_cold.npy")
T_surf_grid = np.linspace(250,4000,200)
Te_grid = np.linspace(180,350,10)
##########################################################

fH2O_new = np.log10(fH2O_new)

@jit(nopython=True)
def my_interp(Tsurf,Te,PH2O,PCO2):  
   
    if Tsurf<=np.min(T_surf_grid):
        Actual_Ts = np.min(T_surf_grid)
        Ts_index = 0
    elif Tsurf>=np.max(T_surf_grid):
        Actual_Ts = np.max(T_surf_grid)
        Ts_index = len(T_surf_grid) - 2 #98 
    else:
        for i in range(1,len(T_surf_grid)):
            if (T_surf_grid[i]>Tsurf)and(T_surf_grid[i-1]<=Tsurf):
                Ts_index = i-1
                Actual_Ts = Tsurf

    if Te<=np.min(Te_grid):
        Actual_Te = np.min(Te_grid)
        Te_index = 0
    elif Te>=np.max(Te_grid):
        Actual_Te = np.max(Te_grid)
        Te_index = len(Te_grid) - 2 #4
    else:
        for i in range(1,len(Te_grid)): ## Te already filtered, hopefully
            if (Te_grid[i]>Te)and(Te_grid[i-1]<=Te):
                Te_index = i-1
                Actual_Te = Te

    if PH2O <= np.min(P_H2O_grid_new):
        Actual_H2O = np.min(P_H2O_grid_new) 
        H2O_index = 0
    elif PH2O >= np.max(P_H2O_grid_new):
        Actual_H2O = np.max(P_H2O_grid_new) 
        H2O_index = len(P_H2O_grid_new) - 2 #32
    else:
        for i in range(1,len(P_H2O_grid_new)):
            if (P_H2O_grid_new[i]>PH2O)and(P_H2O_grid_new[i-1]<=PH2O):
                H2O_index = i-1
                Actual_H2O = PH2O

    if PCO2 <= np.min(P_CO2_grid_new):
        Actual_CO2 = np.min(P_CO2_grid_new) 
        CO2_index = 0
    elif PCO2 >= np.max(P_CO2_grid_new):
        Actual_CO2 = np.max(P_CO2_grid_new) 
        CO2_index = len(P_CO2_grid_new) - 2 #22
    else:
        for i in range(1,len(P_CO2_grid_new)):
            if (P_CO2_grid_new[i]>PCO2)and(P_CO2_grid_new[i-1]<=PCO2):
                CO2_index = i-1
                Actual_CO2 = PCO2
    intTs = T_surf_grid[1+Ts_index]-T_surf_grid[Ts_index]
    intTe = Te_grid[1+Te_index] - Te_grid[Te_index]
    intH2O = P_H2O_grid_new[1+H2O_index] - P_H2O_grid_new[H2O_index] 
    intCO2 = P_CO2_grid_new[1+CO2_index] - P_CO2_grid_new[CO2_index]

   
    delTs  =  Actual_Ts -   T_surf_grid[Ts_index]
    delTe  =  Actual_Te -   Te_grid[Te_index]
    delH2O =  Actual_H2O -   P_H2O_grid_new[H2O_index]     
    delCO2  =  Actual_CO2  -   P_CO2_grid_new[CO2_index]

    xd = delTs / intTs
    yd = delTe / intTe
    zd = delH2O / intH2O
    qd = delCO2 / intCO2

    C00 = OLR_hybrid_FIX[Ts_index,Te_index,H2O_index,CO2_index] * (1 - xd) + xd * OLR_hybrid_FIX[Ts_index+1,Te_index,H2O_index,CO2_index]
    C01 = OLR_hybrid_FIX[Ts_index,Te_index,H2O_index+1,CO2_index] * (1 - xd) + xd * OLR_hybrid_FIX[Ts_index+1,Te_index,H2O_index+1,CO2_index]
    C10 = OLR_hybrid_FIX[Ts_index,Te_index+1,H2O_index,CO2_index] * (1 - xd) + xd * OLR_hybrid_FIX[Ts_index+1,Te_index+1,H2O_index,CO2_index]
    C11 = OLR_hybrid_FIX[Ts_index,Te_index+1,H2O_index+1,CO2_index] * (1 - xd) + xd * OLR_hybrid_FIX[Ts_index+1,Te_index+1,H2O_index+1,CO2_index]

    C0 = C00 * (1 - yd) + yd * C10
    C1 = C01 * (1 - yd) + yd * C11

    C = C0*(1-zd) + C1*zd

    DC00 = OLR_hybrid_FIX[Ts_index,Te_index,H2O_index,CO2_index+1] * (1 - xd) + xd * OLR_hybrid_FIX[Ts_index+1,Te_index,H2O_index,CO2_index+1]
    DC01 = OLR_hybrid_FIX[Ts_index,Te_index,H2O_index+1,CO2_index+1] * (1 - xd) + xd * OLR_hybrid_FIX[Ts_index+1,Te_index,H2O_index+1,CO2_index+1]
    DC10 = OLR_hybrid_FIX[Ts_index,Te_index+1,H2O_index,CO2_index+1] * (1 - xd) + xd * OLR_hybrid_FIX[Ts_index+1,Te_index+1,H2O_index,CO2_index+1]
    DC11 = OLR_hybrid_FIX[Ts_index,Te_index+1,H2O_index+1,CO2_index+1] * (1 - xd) + xd * OLR_hybrid_FIX[Ts_index+1,Te_index+1,H2O_index+1,CO2_index+1]

    DC0 = DC00 * (1 - yd) + yd * DC10
    DC1 = DC01 * (1 - yd) + yd * DC11

    DC = DC0*(1-zd) + DC1*zd

    answer = C*(1-qd) + qd * DC
    if Tsurf < np.min(T_surf_grid):
         return (Tsurf/np.min(T_surf_grid))**4 * 10**answer # Extrapolation for cold surface temperatures (rarely used)
    elif Tsurf > np.max(T_surf_grid):
         return (Tsurf/np.max(T_surf_grid))**4 * 10**answer   # Extrapolation for hot surface temperatures (rarely used) 
    else:
        return 10**answer
        
@jit(nopython=True)
def my_water_frac(Tsurf,Te,PH2O,PCO2):  
    
    if Tsurf >=647: 
        return 1.0    
    if Tsurf<=np.min(T_surf_grid):
        Actual_Ts = np.min(T_surf_grid)
        Ts_index = 0
    elif Tsurf>=np.max(T_surf_grid):
        Actual_Ts = np.max(T_surf_grid)
        Ts_index = len(T_surf_grid) - 2 #98 
    else:
        for i in range(1,len(T_surf_grid)):
            if (T_surf_grid[i]>Tsurf)and(T_surf_grid[i-1]<=Tsurf):
                Ts_index = i-1
                Actual_Ts = Tsurf

    if Te<=np.min(Te_grid):
        Actual_Te = np.min(Te_grid)
        Te_index = 0
    elif Te>=np.max(Te_grid):
        Actual_Te = np.max(Te_grid)
        Te_index = len(Te_grid) - 2#4
    else:
        for i in range(1,len(Te_grid)): ## Te already filtered, hopefully
            if (Te_grid[i]>Te)and(Te_grid[i-1]<=Te):
                Te_index = i-1
                Actual_Te = Te

    if PH2O <= np.min(P_H2O_grid_new):
        Actual_H2O = np.min(P_H2O_grid_new) 
        H2O_index = 0
    elif PH2O >= np.max(P_H2O_grid_new):
        Actual_H2O = np.max(P_H2O_grid_new) 
        H2O_index = len(P_H2O_grid_new) - 2 #32
    else:
        for i in range(1,len(P_H2O_grid_new)):
            if (P_H2O_grid_new[i]>PH2O)and(P_H2O_grid_new[i-1]<=PH2O):
                H2O_index = i-1
                Actual_H2O = PH2O

    if PCO2 <= np.min(P_CO2_grid_new):
        Actual_CO2 = np.min(P_CO2_grid_new) 
        CO2_index = 0
    elif PCO2 >= np.max(P_CO2_grid_new):
        Actual_CO2 = np.max(P_CO2_grid_new) 
        CO2_index = len(P_CO2_grid_new) - 2 #22
    else:
        for i in range(1,len(P_CO2_grid_new)):
            if (P_CO2_grid_new[i]>PCO2)and(P_CO2_grid_new[i-1]<=PCO2):
                CO2_index = i-1
                Actual_CO2 = PCO2
    intTs = T_surf_grid[1+Ts_index]-T_surf_grid[Ts_index]
    intTe = Te_grid[1+Te_index] - Te_grid[Te_index]
    intH2O = P_H2O_grid_new[1+H2O_index] - P_H2O_grid_new[H2O_index] 
    intCO2 = P_CO2_grid_new[1+CO2_index] - P_CO2_grid_new[CO2_index]

   
    delTs  =  Actual_Ts -   T_surf_grid[Ts_index]
    delTe  =  Actual_Te -   Te_grid[Te_index]
    delH2O =  Actual_H2O -   P_H2O_grid_new[H2O_index]     
    delCO2  =  Actual_CO2  -   P_CO2_grid_new[CO2_index]

    xd = delTs / intTs
    yd = delTe / intTe
    zd = delH2O / intH2O
    qd = delCO2 / intCO2

    C00 = water_frac_multi_new[Ts_index,Te_index,H2O_index,CO2_index] * (1 - xd) + xd * water_frac_multi_new[Ts_index+1,Te_index,H2O_index,CO2_index]
    C01 = water_frac_multi_new[Ts_index,Te_index,H2O_index+1,CO2_index] * (1 - xd) + xd * water_frac_multi_new[Ts_index+1,Te_index,H2O_index+1,CO2_index]
    C10 = water_frac_multi_new[Ts_index,Te_index+1,H2O_index,CO2_index] * (1 - xd) + xd * water_frac_multi_new[Ts_index+1,Te_index+1,H2O_index,CO2_index]
    C11 = water_frac_multi_new[Ts_index,Te_index+1,H2O_index+1,CO2_index] * (1 - xd) + xd * water_frac_multi_new[Ts_index+1,Te_index+1,H2O_index+1,CO2_index]

    C0 = C00 * (1 - yd) + yd * C10
    C1 = C01 * (1 - yd) + yd * C11

    C = C0*(1-zd) + C1*zd

    DC00 = water_frac_multi_new[Ts_index,Te_index,H2O_index,CO2_index+1] * (1 - xd) + xd * water_frac_multi_new[Ts_index+1,Te_index,H2O_index,CO2_index+1]
    DC01 = water_frac_multi_new[Ts_index,Te_index,H2O_index+1,CO2_index+1] * (1 - xd) + xd * water_frac_multi_new[Ts_index+1,Te_index,H2O_index+1,CO2_index+1]
    DC10 = water_frac_multi_new[Ts_index,Te_index+1,H2O_index,CO2_index+1] * (1 - xd) + xd * water_frac_multi_new[Ts_index+1,Te_index+1,H2O_index,CO2_index+1]
    DC11 = water_frac_multi_new[Ts_index,Te_index+1,H2O_index+1,CO2_index+1] * (1 - xd) + xd * water_frac_multi_new[Ts_index+1,Te_index+1,H2O_index+1,CO2_index+1]

    DC0 = DC00 * (1 - yd) + yd * DC10
    DC1 = DC01 * (1 - yd) + yd * DC11

    DC = DC0*(1-zd) + DC1*zd

    answer = C*(1-qd) + qd * DC
    
    if answer <0:
        return 0.0
    elif answer>1:
        return 1.0
    
    return answer

@jit(nopython=True)
def my_fH2O(Tsurf,Te,PH2O,PCO2):  
        
    if Tsurf<=np.min(T_surf_grid):
        Actual_Ts = np.min(T_surf_grid)
        Ts_index = 0
    elif Tsurf>=np.max(T_surf_grid):
        Actual_Ts = np.max(T_surf_grid)
        Ts_index = len(T_surf_grid) - 2#98 
    else:
        for i in range(1,len(T_surf_grid)):
            if (T_surf_grid[i]>Tsurf)and(T_surf_grid[i-1]<=Tsurf):
                Ts_index = i-1
                Actual_Ts = Tsurf

    if Te<=np.min(Te_grid):
        Actual_Te = np.min(Te_grid)
        Te_index = 0
    elif Te>=np.max(Te_grid):
        Actual_Te = np.max(Te_grid)
        Te_index = len(Te_grid) - 2#4
    else:
        for i in range(1,len(Te_grid)): ## Te already filtered, hopefully
            if (Te_grid[i]>Te)and(Te_grid[i-1]<=Te):
                Te_index = i-1
                Actual_Te = Te

    if PH2O <= np.min(P_H2O_grid_new):
        Actual_H2O = np.min(P_H2O_grid_new) 
        H2O_index = 0
    elif PH2O >= np.max(P_H2O_grid_new):
        Actual_H2O = np.max(P_H2O_grid_new) 
        H2O_index = len(P_H2O_grid_new) - 2 #32
    else:
        for i in range(1,len(P_H2O_grid_new)):
            if (P_H2O_grid_new[i]>PH2O)and(P_H2O_grid_new[i-1]<=PH2O):
                H2O_index = i-1
                Actual_H2O = PH2O

    if PCO2 <= np.min(P_CO2_grid_new):
        Actual_CO2 = np.min(P_CO2_grid_new) 
        CO2_index = 0
    elif PCO2 >= np.max(P_CO2_grid_new):
        Actual_CO2 = np.max(P_CO2_grid_new) 
        CO2_index = len(P_CO2_grid_new) - 2 #22
    else:
        for i in range(1,len(P_CO2_grid_new)):
            if (P_CO2_grid_new[i]>PCO2)and(P_CO2_grid_new[i-1]<=PCO2):
                CO2_index = i-1
                Actual_CO2 = PCO2
    intTs = T_surf_grid[1+Ts_index]-T_surf_grid[Ts_index]
    intTe = Te_grid[1+Te_index] - Te_grid[Te_index]
    intH2O = P_H2O_grid_new[1+H2O_index] - P_H2O_grid_new[H2O_index] 
    intCO2 = P_CO2_grid_new[1+CO2_index] - P_CO2_grid_new[CO2_index]

   
    delTs  =  Actual_Ts -   T_surf_grid[Ts_index]
    delTe  =  Actual_Te -   Te_grid[Te_index]
    delH2O =  Actual_H2O -   P_H2O_grid_new[H2O_index]     
    delCO2  =  Actual_CO2  -   P_CO2_grid_new[CO2_index]

    xd = delTs / intTs
    yd = delTe / intTe
    zd = delH2O / intH2O
    qd = delCO2 / intCO2

    C00 = fH2O_new[Ts_index,Te_index,H2O_index,CO2_index] * (1 - xd) + xd * fH2O_new[Ts_index+1,Te_index,H2O_index,CO2_index]
    C01 = fH2O_new[Ts_index,Te_index,H2O_index+1,CO2_index] * (1 - xd) + xd * fH2O_new[Ts_index+1,Te_index,H2O_index+1,CO2_index]
    C10 = fH2O_new[Ts_index,Te_index+1,H2O_index,CO2_index] * (1 - xd) + xd * fH2O_new[Ts_index+1,Te_index+1,H2O_index,CO2_index]
    C11 = fH2O_new[Ts_index,Te_index+1,H2O_index+1,CO2_index] * (1 - xd) + xd * fH2O_new[Ts_index+1,Te_index+1,H2O_index+1,CO2_index]

    C0 = C00 * (1 - yd) + yd * C10
    C1 = C01 * (1 - yd) + yd * C11

    C = C0*(1-zd) + C1*zd

    DC00 = fH2O_new[Ts_index,Te_index,H2O_index,CO2_index+1] * (1 - xd) + xd * fH2O_new[Ts_index+1,Te_index,H2O_index,CO2_index+1]
    DC01 = fH2O_new[Ts_index,Te_index,H2O_index+1,CO2_index+1] * (1 - xd) + xd * fH2O_new[Ts_index+1,Te_index,H2O_index+1,CO2_index+1]
    DC10 = fH2O_new[Ts_index,Te_index+1,H2O_index,CO2_index+1] * (1 - xd) + xd * fH2O_new[Ts_index+1,Te_index+1,H2O_index,CO2_index+1]
    DC11 = fH2O_new[Ts_index,Te_index+1,H2O_index+1,CO2_index+1] * (1 - xd) + xd * fH2O_new[Ts_index+1,Te_index+1,H2O_index+1,CO2_index+1]

    DC0 = DC00 * (1 - yd) + yd * DC10
    DC1 = DC01 * (1 - yd) + yd * DC11

    DC = DC0*(1-zd) + DC1*zd

    answer = C*(1-qd) + qd * DC

    if Tsurf < np.min(T_surf_grid):  ## Extrapolation for low surface temperatures (rarely used)
        return (Tsurf/np.min(T_surf_grid))**3 * 10**answer 
    
    return 10**answer

## OLR correction after partitioning CO2 between the atmosphere and liquid water ocean
@jit(nopython=True) 
def correction(Tsurf,Te,PH2O,PCO2,rp,g,CO3,PN2,MMW,PO2):
    if Tsurf<274: # Carbonate equilibrium constants not valid below freezing
        T=274
    else:
        T=Tsurf
    pK1=17.788 - .073104 *T - .0051087*35 + 1.1463*10**-4*T**2
    pK2=20.919 - .064209 *T - .011887*35 + 8.7313*10**-5*T**2
    H_CO2=1.0/(0.018*10.0*np.exp(-6.8346 + 1.2817e4/T - 3.7668e6/T**2 + 2.997e8/T**3)   )
    # from https://srd.nist.gov/JPCRD/jpcrd427.pdf  with unit conversion

    atmo_fraction = my_water_frac(Tsurf,Te,PH2O,PCO2)
    if (atmo_fraction == 1.0)or(PCO2<0): # if no liquid water ocean, all CO2 in atmosphere and same result as before
        return [my_interp(Tsurf,Te,PH2O,PCO2),PCO2,0.0,0.0,0.0,0.0]
    else:
        atmoH2O =  atmo_fraction*PH2O
        Mass_oceans_crude = PH2O*(1.0 - atmo_fraction )* 4 *np.pi *rp**2 / g  ## This is an approximation
        PTOT = atmoH2O + PCO2 + PN2 + PO2
        mtot = (PTOT * 4*np.pi*rp**2) / g
        MCO2 = 0.044
        Mave = MMW
        total_mass_CO2 = mtot * MCO2 * PCO2 /(PTOT * Mave)
        cCon = total_mass_CO2/(MCO2*Mass_oceans_crude) # mol CO2/kg ocean
        if cCon < CO3: #quick fix to ensure no more CO3 than in total atmo-ocean system!
            CO3 = 0.9999*cCon
        #cCon = s * pCO2 + DIC, where pCO2 is in bar
        s = 1e5*(4.0 *np.pi * rp**2 / (MCO2*g) )* (MCO2 / Mave) / Mass_oceans_crude ## mol CO2/ kg atm / Pa, so *1e5 Pa/bar
        aa = CO3 * (1 + s/H_CO2)/(10**-pK2*10**-pK1)
        bb = CO3 / (10**-pK2)
        cc = CO3 - cCon
        rr1 = - (bb + np.sqrt (bb**2 - 4 * aa * cc) ) / (2*aa)
        rr2 = - (bb - np.sqrt (bb**2 - 4 * aa * cc) ) / (2*aa)
        if rr1 <= rr2:
            EM_H_o  = rr2
        else:
            EM_H_o = rr1
        ALK = CO3 * (2+EM_H_o/(10**-pK2))
        EM_pH_o=-np.log10(EM_H_o) ## Evolving model ocean pH
        EM_hco3_o=ALK-2*CO3 ## Evolving model ocean bicarbonate molality
        EM_co2aq_o=( EM_hco3_o*EM_H_o/(10**-pK1) ) ## Evolving model ocean aqueous CO2 molality
        EM_ppCO2_o = EM_co2aq_o /H_CO2 ## Evolving model atmospheric pCO2
        DIC_check = EM_hco3_o + CO3 + EM_co2aq_o
        true_OLR =  my_interp(Tsurf,Te,PH2O,EM_ppCO2_o*1e5)
        return [true_OLR,EM_ppCO2_o*1e5,EM_pH_o,ALK,Mass_oceans_crude,DIC_check]

