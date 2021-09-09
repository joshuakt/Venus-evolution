import numpy as np
import pylab
from scipy.integrate import *
from scipy.interpolate import interp1d
from scipy import optimize
import pdb
import scipy.interpolate
from numba import jit

mH = 1.67e-27
G = 6.67e-11 
kB = 1.38e-23
mO = 16*mH

@jit(nopython=True)
def better_diffusion(fH2O,T,g,fCO2,fO2,fN2): # diffusion H2O thrugh CO2 and N2
    pO2 = fO2
    pN2 = fN2
    pCO2 = fCO2
    ## Units conversion 
    ## H2O_CO2: 9.24e-5 * T**1.5 / np.exp(307.9/T) = 9.24e-5/(10*1.38e-23) * T**0.5 / np.exp(307.9/T)
    ## H2O_N2: 0.187e-5 * T **2.072 =  0.187e-5/(10*1.38e-23)  * T**1.072
    ## H2O_O2: 0.189e-5 * T **2.072 =  0.189e-5/(10*1.38e-23)  * T**1.072
    bH2O_CO2 = 100 * 9.24e-5/(1.38e-22) * T**0.5 / np.exp(307.9/T) # conversion m
    bH2O_N2 = 100 * 0.187e-5/(1.38e-22)  * T**1.072 # conversion m 
    bH2O_O2 = 100 *  0.189e-5/(10*1.38e-23) * T**1.072 # conversion m 
    bH2O = (bH2O_CO2 * pCO2 + bH2O_N2 * pN2 + bH2O_O2*pO2) / (pCO2 + pN2 + pO2)
    Mn = (0.044 * pCO2 + 0.028 * pN2 + 0.032 * pO2)/(pCO2 + pN2 + pO2) #kg/mol
    
    Hn = (8.314*T)/(Mn*g)
    H_H2O = (8.314*T)/(0.018*g)
    phi = bH2O * fH2O * (1.0/Hn - 1.0/H_H2O)
    return phi/6.022e23 #mol H2O/m2/s I think

@jit(nopython=True)
def better_atomic_diffusion(fH2O,T,g,fCO2,fO2,fN2): # diffusion H thrugh CO2 and N2 and O
    pO = (2*fO2+fH2O)/(2*fO2 + fN2 + fCO2 + 3*fH2O)
    pN2 = fN2/(2*fO2 + fN2 + fCO2 + 3*fH2O)
    pCO2 = fCO2/(2*fO2 + fN2 + fCO2 + 3*fH2O)
    pH = 2*fH2O/(2*fO2 + fN2 + fCO2 + 3*fH2O)
    ## Units conversion 
    bH_CO2 = 100 * 8.4e17 * T**0.6
    bH_N2 = 100 * 6.5e17 * T**0.7
    bH_O = 100 * 4.8e17 * T**0.75
    bH = (bH_CO2 * pCO2 + bH_N2 * pN2 + bH_O*pO) / (pCO2 + pN2 + pO)
    Mn = (0.044 * pCO2 + 0.028 * pN2 + 0.016 * pO)/(pCO2 + pN2 + pO) #kg/mol
    Hn = (8.314*T)/(Mn*g)
    H_H = (8.314*T)/(0.001*g)
    phi = bH * fH2O * (1.0/Hn - 1.0/H_H)
    return phi/6.022e23 #mol H2O/m2/s I think

@jit(nopython=True)
def find_epsilon(T,RE,ME,FXUV, XO, XH, XC,epsilon_init,mixparam):

    Rp = RE * 6.371e6
    Mp = ME * 5.972e24
    g = G* Mp / (Rp**2)
    delPHI = G*Mp/Rp
    mass_loss = epsilon_init * FXUV  / (4.0*delPHI)       
    mi = mH
    mj = 16.0*mH
    
    bij = 100.0*4.8e17*(T**0.75) #100 * cm-1 s-1 bH,O
    fj = XO

    Fi_limit  =  g*(mj - mi) * bij   / ( kB * T * (1+fj))  
    true_flux = Fi_limit * mi
    ratio = true_flux/mass_loss   
    if ratio > 1.0: 
        return epsilon_init     
    else:     
        epsilon =  epsilon_init*true_flux/ mass_loss                                              
        return ((1-mixparam)*epsilon + mixparam*epsilon_init) # mix param = 0.9 -> only 10% energy about xj=1 goes into extra escape

@jit(nopython=True)                                                                                                                                                                                                              
def Odert_three(T,RE,ME,epsilon,FXUV, XO, XH, XC):

    Rp = RE * 6.371e6
    Mp = ME * 5.972e24
    g = G* Mp / (Rp**2)
    delPHI = G*Mp/Rp
    mass_loss = epsilon * FXUV  / (4.0*delPHI)
    
    # carbon
    #mk = 12.0*mH # C atoms
    mk = 44.0*mH # CO2 molecules
    fk = XC
    
    mi = mH
    mj = 16.0*mH
    
    bij = 100.0*4.8e17*(T**0.75) #100 * cm-1 s-1 bH,O
    fj = XO
    
    bik =100* 8.4e17*T**0.6 #k is CO2
    bjk = 100* 7.86e16 * T**0.776 # O in CO2
    
    ### carefully verified
    LHS = mass_loss  + mj * fj * g * (mj - mi) * bij / (kB*T * (1 + fj)) + (mk*fk*g * (mk - mi) * bik / (kB*T) )/ (1.0 + bik*fj/bjk) - mk*fk * bik *fj * g * (mj - mi) / ((1+bik*fj/bjk)* kB*T*(1+fj)) + mk*fk*bik*fj*g*(mj-mi)*bij / ((1+bik*fj/bjk )* bjk *kB*T* (1+fj))
    RHS = ( mi + mj*fj + mk*fk / (1.0 + bik*fj/bjk) +  mk*fk * (bik*fj/bjk) / (1.0 + bik*fj/bjk)) # verified
    Fi = LHS/RHS
    
    #Fi = (mass_loss + mj * fj *g * (mj - mi) * bij / (kB*T*(1+fi))) / (mi+mj*fj)
    xj = 1 -  g*(mj - mi) * bij / (Fi * kB * T * (1+fj)) 
    ### need better criteria for zeroing out k flux, recalculating xk
    if xj<0: 
        xj = 0.0

        # checked
        RHS = mass_loss  + bik*(mk*fk*g*(mk - mi)/ (kB*T))/ (1.0 + bik*fj/bjk)
        LHS = mi + mk*fk / (1.0 + bik*fj/bjk) + mk*fk*bik* fj / (bij*(1 + fj*bik/bjk))
        Fi = RHS/LHS #new Fi assuming zero j
        
        xk_top = 1 -  g*(mk - mi) * bik / (Fi * kB * T ) + (bik/bij)*fj * (1 - xj) + (bik/bjk) * fj *xj
        xk_bot = 1 + (bik/bjk) * fj
        xk = xk_top/ xk_bot
        
        if xk< 0.0: #no O escape no C escape
            xk = 0.0
            Fk = 0.0
            Fj = 0.0
            Fi = mass_loss / (mi)
        else: # no O escape, but yes C escape
            Fj = 0.0
            Fk = Fi * fk * xk

    else:
        xk_top = 1 -  g*(mk - mi) * bik / (Fi * kB * T ) + (bik/bij)*fj * (1 - xj) + (bik/bjk) * fj *xj
        xk_bot = 1 + (bik/bjk) * fj
        xk = xk_top / xk_bot
        if xk>0: # O escape and C escape
            Fj = Fi * fj *xj
            Fk = Fi * fk * xk
        else: # O escape, no C escape
            xk = 0.0
            Fk = 0.0
            Fi = mass_loss / (mi + mj*fj*xj)
            Fj = (mass_loss - mi*Fi)/mj

    F_down_O = 0.5* Fi - Fj + 2*Fk
    
    return [Fi*mi, Fj*mj ,F_down_O * mj, Fk*mk]


