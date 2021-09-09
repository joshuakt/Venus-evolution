########################## 
## Load modules
import numpy as np
from scipy import optimize
import sys
import random
from numba import jit
from numba_nelder_mead import nelder_mead
from VolcGases import functions
################################

## The VolcGases outgassing model is described in Wogan et al. (2020; PSJ), and must be installed prior to running this code: https://github.com/Nicholaswogan/VolcGases

def buffer_fO2(T,Press,redox_buffer): # T in K, P in bar
    if redox_buffer == 'FMQ':
        [A,B,C] = [25738.0, 9.0, 0.092]
    elif redox_buffer == 'IW':
        [A,B,C] = [27215 ,6.57 ,0.0552]
    elif redox_buffer == 'MH':
        [A,B,C] = [25700.6,14.558,0.019] # from Frost
    else:
        print ('error, no such redox buffer')
        return -999
    return 10**(-A/T + B + C*(Press-1)/T)

@jit(nopython=True)
def get_fO2(XFe2O3_over_XFeO,P,T,Total_Fe): #Function for calculating oxygen fugacity given Fe3+/Fe2+ speciation
## Total_Fe is a mole fraction of iron minerals XFeO + XFeO1.5 = Total_Fe, and XFe2O3 = 0.5*XFeO1.5, so XFeO + 2XFe2O3 = Total_Fe
    XAl2O3 = 0.022423 
    XCaO = 0.0335 
    XNa2O = 0.0024 
    XK2O = 0.0001077 
    terms1 =  11492.0/T - 6.675 - 2.243*XAl2O3
    terms2 = 3.201*XCaO + 5.854 * XNa2O
    terms3 = 6.215*XK2O - 3.36 * (1 - 1673.0/T - np.log(T/1673.0))
    terms4 = -7.01e-7 * P/T - 1.54e-10 * P * (T - 1673)/T + 3.85e-17 * P**2 / T
    fO2 =  np.exp( (np.log(XFe2O3_over_XFeO) + 1.828 * Total_Fe -(terms1+terms2+terms3+terms4) )/0.196)
    return fO2 

@jit(nopython=True)
def outgas_flux_cal_fast(Temp,Pressure,mantle_ratio,mantle_mass,mantle_CO2_mass,mantle_H2O_mass,M_MELT,Total_Fe,F): #M_MELT is g/s melt production, Pressure in Pa, Temp in KS
    pO2 = get_fO2(mantle_ratio,Pressure,Temp,Total_Fe)
    if (mantle_CO2_mass<0.0)or(mantle_H2O_mass<0)or(Pressure<0):
        print ('Nothing in the mantle!')
        return [0.0,0.0,0.0,0.0,0.0,0.0]
    
    x = 0.01550152865954013
    M_H2O = 18.01528
    M_CO2 = 44.01

    XH2O_melt_max = x*M_H2O*0.499 # half of mol fraction allowed to be H2O
    XCO2_melt_max = x*M_CO2*0.499 # half of mol fraction allowed to be CO2

    XH2O_melt = (1- (1-F)**(1/0.01)) * (mantle_H2O_mass/mantle_mass)/F
    if 0.99*XH2O_melt_max<XH2O_melt:
        XH2O_melt = 0.99*XH2O_melt_max
    XCO2_melt = (1- (1-F)**(1/2e-3)) * (mantle_CO2_mass/mantle_mass)/F
    if 0.99*XCO2_melt_max<XCO2_melt:
        XCO2_melt = 0.99*XCO2_melt_max
    
    # do we make graphite? 
    graph_on = "y"
    if graph_on == "y":
        log10_K1 = 40.07639 - 2.53932e-2 * Temp + 5.27096e-6*Temp**2 + 0.0267 * (Pressure/1e5 - 1 )/Temp
        log10_K2 = - 6.24763 - 282.56/Temp - 0.119242 * (Pressure/1e5 - 1000)/Temp
        gXCO3_melt = ((10**log10_K1)*(10**log10_K2)*pO2)/(1+(10**log10_K1)*(10**log10_K2)*pO2) 
        gXCO2_melt = (44/36.594)*gXCO3_melt / (1 - (1 - 44/36.594)*gXCO3_melt)

        if gXCO2_melt < XCO2_melt:
            XCO2_melt = gXCO2_melt
        ## Derivation:
        ## XCO3 = (wcO2/44) / [(100-wCO2)/fwm + wCO2/44] Holloway et al. 1992
        ## XCO3 * [(100-wCO2)/fwm + wCO2/44] = (wcO2/44) 
        ## XCO3 * wCO2 [(100/wCO2 - 1)/fwm + 1/44] =  (wcO2/44) 
        ## [(100/wCO2 - 1)/fwm + 1/44]  = 1/(XCO3 * 44)
        ## 100/wCO2  = fwm * [1/(XCO3 * 44) - 1/44] + 1
        ## wCO2 =(44/fwm) * 100*XCO3 /  ( [1 - XCO3] + 44/fwm*XCO3) = (44/fwm) * 100*XCO3 / (1-XCO3*(1 - 44/fwm))
        ## wCO2 = (44/fwm) * XCO3 / (1-XCO3*(1 - 44/fwm)) 

    [PH2O, PH2, PCO2, PCO, PCH4, alphaG, xCO2, xH2O] = functions.solve_gases_jit(Temp,float(Pressure/1e5),pO2,XCO2_melt,XH2O_melt)
    [mH2O,mH2,mCO2,mCO,mCH4,mO2] =  [1e5*PH2O/Pressure, 1e5*PH2/Pressure, 1e5*PCO2/Pressure, 1e5*PCO/Pressure, 1e5*PCH4/Pressure,1e5*pO2/Pressure]

    if alphaG<0:
        print ('-ve alphaG, outgassing assumed to be zero!')
        return [0.0,0.0,0.0,0.0,0.0,0.0]
    xm = 1.55e-2 #mol magma / g magma
    q_H2O = mH2O * alphaG * xm / (1-alphaG) #mol gas/g magma 
    q_CO2 = mCO2 * alphaG * xm / (1-alphaG)
    q_H2 = mH2 * alphaG * xm / (1-alphaG)
    q_CO = mCO * alphaG * xm / (1-alphaG)
    q_CH4 = mCH4 * alphaG * xm / (1-alphaG)


    if alphaG>1.0: 
        F_H2O = 0.0
        F_CO2 = 0.0
        F_CO = 0.0 
        F_H2 = 0.0
        F_CH4 = 0.0 
        O2_consumption = 0.5*F_H2 + 0.5*F_CO + 2 * F_CH4
        print ('weird >1 alphaG')
        return [F_H2O,F_CO2,F_H2,F_CO,F_CH4,O2_consumption]

    F_H2O = M_MELT*q_H2O
    F_CO2 = M_MELT*q_CO2
    F_H2 = M_MELT*q_H2 # H2 + 0.5O2 = H2O
    F_CO = M_MELT*q_CO #CO + 0.5O2 = CO2 
    F_CH4 = M_MELT*q_CH4 # CH4 + 2O2 = CO2 + 2H2O
    O2_consumption = 0.5*F_H2 + 0.5*F_CO + 2 * F_CH4
    return [F_H2O,F_CO2,F_H2,F_CO,F_CH4,O2_consumption] # outputs mol/s fluxes of gases, and mol/s O2 consumption

## Units check:
## xm, constant = mol magma/ g magma (inverse molar mass)
## alpha, mol gas total / mol gas AND magma
## 1- alpha = mol gas AND magma/mol gas AND magma - mol gas total / mol gas AND magma = mol magma / mol gas AND magma
## qi, mol gas i per kg magma, qi = (alpha*xm / (1 - alpha) * (Pi/P)
## what is 1 - alpha = mol magma / mol gas AND magma
## qi units: (mol gas total / mol gas and magma total) * (mol magma)/ (g magma) / (mol magma / mol gas AND magma) 
## = (mol gas total / g magma) * Pi/P = mol gas i / g total
## outgassing flux, Fi = qi*QM QM is kg magma/s so mol gas i / kg magma * kg magma / s = mol gas i / s 
