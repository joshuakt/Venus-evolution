########################## 
import numpy as np
from scipy import optimize
import sys
import pdb
import random
from thermodynamics import *
from numba import jit
################################

## These functions are redundant: outgassing_module_fast.py is used instead
## This outgassing model is described in Wogan et al. (2020; PSJ). Standalone outgassing code along with detailed documentation and example calculations is available here: https://github.com/Nicholaswogan/VolcGases


@jit(nopython=True)
def get_fO2(XFe2O3_over_XFeO,P,T,Total_Fe): 
## Total_Fe is a mole fraction of iron minerals XFeO + XFeO1.5 = Total_Fe, and XFe2O3 = 0.5*XFeO1.5, xo XFeO + 2XFe2O3 = Total_Fe
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


def buffer_fO2(T,Press,redox_buffer): # T in K, P in bar
    if redox_buffer == 'FMQ':
        [A,B,C] = [25738.0, 9.0, 0.092]
    elif redox_buffer == 'IW':
        [A,B,C] = [27215 ,6.57 ,0.0552]
    elif redox_buffer == 'MH':
        [A,B,C] = [25700.6,14.558,0.019] 
    else:
        print ('error, no such redox buffer')
        return -999
    return 10**(-A/T + B + C*(Press-1)/T)

@jit(nopython=True) #updated from Nick's website 6/23
def system(y):
    ln_x_H2O,ln_x_CO2,ln_H2O,ln_CO2,alphaG,ln_H2,ln_CH4,ln_CO = y
    return (np.exp(ln_H2O)+np.exp(ln_CO2)+np.exp(ln_H2)+np.exp(ln_CH4)+np.exp(ln_CO)-P,\
            -ln_x_CO2+np.exp(ln_x_H2O)*d_H2O+a_CO2*ln_CO2+F1,\
            -ln_x_H2O+a_H2O*ln_H2O+F2,\
            -xH2Otot*P + (np.exp(ln_H2O)+np.exp(ln_H2)+2*np.exp(ln_CH4))*alphaG+(1-alphaG)*np.exp(ln_x_H2O)*P,\
            -xCO2tot*P + (np.exp(ln_CO2)+np.exp(ln_CO)+np.exp(ln_CH4))*alphaG+(1-alphaG)*np.exp(ln_x_CO2)*P,\
            np.log(C1)+ln_H2O-ln_H2,\
            np.log(C2)+ln_CO2-ln_CO,\
            np.log(C3)+ln_CO2+2*ln_H2O-ln_CH4)


@jit(nopython=True) #updated from Nick's website 6/23
def system1(y,d_H2O,a_H2O,C1,C2,C3,P,a_CO2,F1,F2,xH2Otot,xCO2tot):
    ln_x_H2O,ln_x_CO2,ln_H2O,ln_CO2,lnalphaG,ln_H2,ln_CH4,ln_CO = y
    return (np.exp(ln_H2O)+np.exp(ln_CO2)+np.exp(ln_H2)+np.exp(ln_CH4)+np.exp(ln_CO)-P,\
            -ln_x_CO2+np.exp(ln_x_H2O)*d_H2O+a_CO2*ln_CO2+F1,\
            -ln_x_H2O+a_H2O*ln_H2O+F2,\
            -xH2Otot*P + (np.exp(ln_H2O)+np.exp(ln_H2)+2*np.exp(ln_CH4))*np.exp(lnalphaG)+(1-np.exp(lnalphaG))*np.exp(ln_x_H2O)*P,\
            -xCO2tot*P + (np.exp(ln_CO2)+np.exp(ln_CO)+np.exp(ln_CH4))*np.exp(lnalphaG)+(1-np.exp(lnalphaG))*np.exp(ln_x_CO2)*P,\
            np.log(C1)+ln_H2O-ln_H2,\
            np.log(C2)+ln_CO2-ln_CO,\
            np.log(C3)+ln_CO2+2*ln_H2O-ln_CH4)

@jit(nopython=True) #updated from Nick's website 6/23
def equation(PCO2,F1,a_CO2,P,C1,C2,C3,xCO2tot,F2,a_H2O,xH2Otot):
    return ( -1 * ( np.e )**( ( F1 + a_CO2 * np.log( PCO2 ) ) ) * ( P )**( -1 ) \
            * ( P + ( -1 * PCO2 + -1 * C2 * PCO2 ) ) * ( ( 1 + ( C1 + C3 * PCO2 ) \
            ) )**( -1 ) + ( ( P )**( -1 ) * ( P + ( -1 * PCO2 + -1 * C2 * PCO2 ) \
            ) * ( ( 1 + ( C1 + C3 * PCO2 ) ) )**( -1 ) * xCO2tot + ( -1 * ( np.e \
            )**( ( F2 + a_H2O * np.log( (P-PCO2-C2*PCO2)/(1+C1+C3*PCO2) ) ) ) * ( -1 * ( P )**( -1 ) * PCO2 + \
            xCO2tot ) + ( ( np.e )**( ( F1 + a_CO2 * np.log( PCO2 ) ) ) * xH2Otot \
            + -1 * ( P )**( -1 ) * PCO2 * xH2Otot ) ) ) )            

@jit(nopython=True)
def jacob(y,d_H2O,a_H2O,C1,C2,C3,P,a_CO2,F1,F2,xH2Otot,xCO2tot):
    lnxH2O,lnxCO2,lnPH2O,lnPCO2,alphaG,lnPH2,lnPCH4,lnPCO = y
    return np.array( [np.array( [0,0,( np.e )**( lnPH2O ),( np.e )**( lnPCO2 \
            ),0,( np.e )**( lnPH2 ),( np.e )**( lnPCH4 ),( np.e )**( lnPCO ),] \
            ),np.array( [d_H2O * ( np.e )**( lnxH2O ),-1,0,a_CO2,0,0,0,0,] \
            ),np.array( [-1,0,a_H2O,0,0,0,0,0,] ),np.array( [( 1 + -1 * alphaG ) * \
            ( np.e )**( lnxH2O ) * P,0,alphaG * ( np.e )**( lnPH2O ),0,( 2 * ( \
            np.e )**( lnPCH4 ) + ( ( np.e )**( lnPH2 ) + ( ( np.e )**( lnPH2O ) + \
            -1 * ( np.e )**( lnxH2O ) * P ) ) ),alphaG * ( np.e )**( lnPH2 ),2 * \
            alphaG * ( np.e )**( lnPCH4 ),0,] ),np.array( [0,( 1 + -1 * alphaG ) \
            * ( np.e )**( lnxCO2 ) * P,0,alphaG * ( np.e )**( lnPCO2 ),( ( np.e \
            )**( lnPCH4 ) + ( ( np.e )**( lnPCO ) + ( ( np.e )**( lnPCO2 ) + -1 * \
            ( np.e )**( lnxCO2 ) * P ) ) ),0,alphaG * ( np.e )**( lnPCH4 ),alphaG \
            * ( np.e )**( lnPCO ),] ),np.array( [0,0,1,0,0,-1,0,0,] ),np.array( \
            [0,0,0,1,0,0,0,-1,] ),np.array( [0,0,2,1,0,0,-1,0,] ),] )


###### Solubility constants
# H2O solubility
# Constants from figure table 6 in Iacono-Marziano et al. 2012. Using Anhydrous constants
a_H2O = 0.54
b_H2O = 1.24
B_H2O = -2.95
C_H2O = 0.02

# CO2 Solubility
# Constants from table 5 in Iacono-Marziano et al. 2012. Using anhydrous
d_H2O = 2.3
d_AI = 3.8
d_FeO_MgO = -16.3
d_Na2O_K2O = 20.1
a_CO2 = 1
b_CO2 = 15.8
C_CO2 = 0.14
B_CO2 = -5.3

# Mass fractions of different species in Mt. Etna magma.
# From Table 1 in Iacono-Marziano et al. 2012.
m_SiO2 = 0.4795
m_TiO2 = 0.0167
m_Al2O3 = 0.1732
m_FeO = 0.1024
m_MgO = 0.0576
m_CaO = 0.1093
m_Na2O = 0.034
m_K2O = 0.0199
m_P2O5 = 0.0051

# molar masses in g/mol
M_SiO2 = 60
M_TiO2 = 79.866
M_Al2O3 = 101.96
M_FeO = 71.844
M_MgO = 40.3044
M_CaO = 56
M_Na2O = 61.97
M_K2O = 94.2
M_P2O5 = 141.94
M_H2O = 18.01528
M_CO2 = 44.01

def solve_gases(T,P,f_O2,mCO2tot,mH2Otot):
    '''
    This function solves for the speciation of gases produced by
    a volcano. This code assumes magma composition of the lava erupting at
    Mt. Etna Italy.

    Inputs:
    T = temperature of the magma and gas in kelvin
    P = pressure of the gas in bar
    f_O2 = oxygen fugacity of the melt
    mCO2tot = mass fraction of CO2 in the magma
    mH2Otot = mass fraction of H2O in the magma

    Outputs:
    an array which contains
    [PH2O, PH2, PCO2, PCO, PCH4, alphaG, xCO2, xH2O, mH2O,mH2,mCO2,mCO,mCH4,mO2]
     where
     PH2O = partial pressure of H2O in the gas in bar
     PH2 = partial pressure of H2 in the gas in bar
     PCO2 = partial pressure of CO2 in the gas in bar
     PCO = partial pressure of CO in the gas in bar
     PCH4 = partial pressure of CH4 in the gas in bar
     alphaG = moles of gas divide by total moles in gas and magma combined
     xCO2 = mol fraction of the CO2 in the magma
     xH2O = mol fraction of the H2O in the magma
     mH2O = fraction of H2O gas (mixing ratio), mH2O = PH2O/P
     mH2 = fraction of H2 gas (mixing ratio), mH2 = pH2/P etc.
     mCO2 = fraction of CO2 gas (mixing ratio)
     mCO = fraction of CO gas (mixing ratio)
     mCH4 = fraction of CH4 gas (mixing ratio)
     mO2 = fraction of O2 gas (mixing ratio, what you put in)
    '''

    ###### Solubility constants
    F1 = -14.234368891317805
    F2 = -5.925014547418225
    a_H2O = 0.54
    a_CO2 = 1
    d_H2O = 2.3

    # mol of magma/g of magma
    x = 0.01550152865954013

    # molar mass in g/mol
    M_H2O = 18.01528
    M_CO2 = 44.01

     ## new from Nick
    A1 = -0.4200250000201988
    A2 = -2.59560737813789
    M_CO2 = 44.01
    C_CO2 = 0.14
    C_H2O = 0.02
    F1 = np.log(1/(M_CO2*x*10**6))+C_CO2*P/T+A1
    F2 = np.log(1/(M_H2O*x*100))+C_H2O*P/T+A2

    # calculate mol fraction of CO2 and H2O in the magma
    xCO2tot=(mCO2tot/M_CO2)/x
    xH2Otot=(mH2Otot/M_H2O)/x

    # equilibrium constants
    # made with Nasa thermodynamic database (Burcat database)
    K1 = np.exp(-29755.11319228574/T+6.652127716162998)
    K2 = np.exp(-33979.12369002451/T+10.418882755464773)
    K3 = np.exp(-96444.47151911151/T+0.22260815074146403)

    #constants
    C1 = K1/f_O2**0.5
    C2 = K2/f_O2**0.5
    C3 = K3/f_O2**2

    Plim = np.exp( (np.log(xCO2tot) - F1 - d_H2O*xH2Otot)/a_CO2) + np.exp( (np.log(xH2Otot) - F2)/a_H2O)
    if  P > 1.2*Plim:
        #print ('Pressure exceeds analytic solubility limit - no outgassing')
        #print (Plim,P,xCO2tot,xH2Otot,mCO2tot,mH2Otot)
        return (0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0 )

    # Now solve simple system
    find = np.logspace(np.log10(P),-15,1000)
    find2 = np.logspace(-15,np.log10(P),1000)
    for i in range(0,len(find)):

        #print (i,find[i],P)
        #print (C1,C2,C3)
        #print (f_O2)
        if np.isnan(equation(find[i],F1,a_CO2,P,C1,C2,C3,xCO2tot,F2,a_H2O,xH2Otot))==False:
            found_high = find[i]
            #print ('yikes')
            break
    for i in range(0,len(find2)):
        if np.isnan(equation(find2[i],F1,a_CO2,P,C1,C2,C3,xCO2tot,F2,a_H2O,xH2Otot))==False:
            found_low = find2[i]
            #print ('here wat')
            break
    try:
        sol = optimize.root(equation,found_high,args=(F1,a_CO2,P,C1,C2,C3,xCO2tot,F2,a_H2O,xH2Otot),method='lm')
        if sol['success']==False:
            #print ('here tho')
            sys.exit()
    except:
        sol = optimize.root(equation,found_low,args=(F1,a_CO2,P,C1,C2,C3,xCO2tot,F2,a_H2O,xH2Otot),method='lm')
        if sol['success']==False:
            #print ('con oops')
            sys.exit('Convergence issues! First optimization.')

    P_CO2 = sol['x'][0]
    # Solve for the rest of the variables in the simple system
    P_CO = C2*P_CO2
    x_CO2 = np.exp(F1+a_CO2*np.log(P_CO2))
    alphaG = P*(x_CO2-xCO2tot)/(-P_CO2+P*x_CO2)
    P_H2O = (P-P_CO2-C2*P_CO2)/(1+C1+C3*P_CO2)
    P_H2 = C1*P_H2O
    P_CH4 = C3*P_CO2*P_H2O**2
    x_H2O = np.exp(F2+a_H2O*np.log(P_H2O))
    # use different alphaG as inital guess
    alphaG = .1

    # now use the solution of the simple system to solve the
    # harder problem. I will try to solve it two different ways to
    # make sure I avoid errors.

    # error tolerance
    tol = 1e-7

    try: 
        init_cond = [np.log(x_H2O),np.log(x_CO2),np.log(P_H2O),np.log(P_CO2),alphaG,np.log(P_H2),np.log(P_CH4),np.log(P_CO)]
        sol = optimize.root(system,init_cond,args = (d_H2O,a_H2O,C1,C2,C3,P,a_CO2,F1,F2,xH2Otot,xCO2tot),method='lm',options={'maxiter': 10000})#,jac=jacob)
        error = np.linalg.norm(system(sol['x'],d_H2O,a_H2O,C1,C2,C3,P,a_CO2,F1,F2,xH2Otot,xCO2tot))
        if error>tol or sol['success']==False:
            #print ('con here')
            sys.exit('Convergence issues!')

        ln_x_H2O,ln_x_CO2,ln_P_H2O,ln_P_CO2,alphaG,ln_P_H2,ln_P_CH4,ln_P_CO = sol['x']

        if alphaG<0:
            #print ('con there')
            sys.exit('alphaG is negative')
    except:
        alphaG = .1
        init_cond1 = [np.log(x_H2O),np.log(x_CO2),np.log(P_H2O),np.log(P_CO2),np.log(alphaG),np.log(P_H2),np.log(P_CH4),np.log(P_CO)]
        sol1 = optimize.root(system1,init_cond1,args = (d_H2O,a_H2O,C1,C2,C3,P,a_CO2,F1,F2,xH2Otot,xCO2tot),method='lm',options={'maxiter': 10000})#,jac=jacob1)
        error1 = np.linalg.norm(system1(sol1['x'],d_H2O,a_H2O,C1,C2,C3,P,a_CO2,F1,F2,xH2Otot,xCO2tot))
        ln_x_H2O,ln_x_CO2,ln_P_H2O,ln_P_CO2,ln_alphaG,ln_P_H2,ln_P_CH4,ln_P_CO = sol1['x']
        alphaG = np.exp(ln_alphaG)
        #print (alphaG)
        if (error1>tol and alphaG>1e-4 and alphaG<1.0) or sol1['success']==False:
            #print(error1)
            #print(sol1)
            sys.exit('Convergence issues!')
        if error1>tol:
            #print('warning: outgassing equations not solved to high tolerance')
            return (0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0 )

    return (np.exp(ln_P_H2O),np.exp(ln_P_H2),np.exp(ln_P_CO2),np.exp(ln_P_CO),\
            np.exp(ln_P_CH4),alphaG,np.exp(ln_x_CO2),np.exp(ln_x_H2O),np.exp(ln_P_H2O)/P,np.exp(ln_P_H2)/P,np.exp(ln_P_CO2)/P,np.exp(ln_P_CO)/P,\
            np.exp(ln_P_CH4)/P,f_O2/P )


def outgas_flux_cal(Temp,Pressure,mantle_ratio,mantle_mass,mantle_CO2_mass,mantle_H2O_mass,M_MELT,Total_Fe,F): #M_MELT is g/s melt production, Pressure in Pa, Temp in KS
    pO2 = get_fO2(mantle_ratio,Pressure,Temp,Total_Fe)
    if (mantle_CO2_mass<0.0)or(mantle_H2O_mass<0)or(Pressure<0):
        print ('Nothing in the mantle!')
        return [0.0,0.0,0.0,0.0,0.0,0.0]

    
    x = 0.01550152865954013
    M_H2O = 18.01528
    M_CO2 = 44.01

    XH2O_melt_max = x*M_H2O*0.499 # half of mol fraction allowed to be H2O
    XCO2_melt_max = x*M_CO2*0.499 # half of mol fraction allowed to be CO2
    #if (0.95*XH2O_melt_max<(1- (1-F)**(1/0.01)) * (mantle_H2O_mass/mantle_mass)/F)or(0.95*XCO2_melt_max<(1- (1-F)**(1/2e-3)) * (mantle_CO2_mass/mantle_mass)/F ):
    #    print ('would have gone alpha>1')
    XH2O_melt = np.min([0.99*XH2O_melt_max,(1- (1-F)**(1/0.01)) * (mantle_H2O_mass/mantle_mass)/F ]) # mass frac, ensures mass frac never implies all moles volatile!
    XCO2_melt =  np.min([0.99*XCO2_melt_max,(1- (1-F)**(1/2e-3)) * (mantle_CO2_mass/mantle_mass)/F ])# mass frac
    

    # do we make graphite?
    graph_on = "n"
    if graph_on == "y":
        log10_K1 = 40.07639 - 2.53932e-2 * Temp + 5.27096e-6*Temp**2 + 0.0267 * (Pressure/1e5 - 1 )/Temp
        log10_K2 = - 6.24763 - 282.56/Temp - 0.119242 * (Pressure/1e5 - 1000)/Temp
        gXCO3_melt = ((10**log10_K1)*(10**log10_K2)*pO2)/(1+(10**log10_K1)*(10**log10_K2)*pO2) #oughly matches Fig. 1b in Scientific Reports
        gXCO2_melt = (44/36.594)*gXCO3_melt / (1 - (1 - 44/36.594)*gXCO3_melt)

        XCO2_melt = np.min([XCO2_melt,gXCO2_melt])
        ## XCO3 = (wcO2/44) / [(100-wCO2)/fwm + wCO2/44] Holloway et al. 1992
        ## XCO3 * [(100-wCO2)/fwm + wCO2/44] = (wcO2/44) 
        ## XCO3 * wCO2 [(100/wCO2 - 1)/fwm + 1/44] =  (wcO2/44) 
        ##  [(100/wCO2 - 1)/fwm + 1/44]  = 1/(XCO3 * 44)
        ##  100/wCO2  = fwm * [1/(XCO3 * 44) - 1/44] + 1
        ##  wCO2 =(44/fwm) * 100*XCO3 /  ( [1 - XCO3] + 44/fwm*XCO3) = (44/fwm) * 100*XCO3 / (1-XCO3*(1 - 44/fwm))
        ## wCO2 = (44/fwm) * XCO3 / (1-XCO3*(1 - 44/fwm)) #not wt Percent, matches Scientific Reports with noack
    #print (mantle_H2O_mass/mantle_mass,mantle_CO2_mass/mantle_mass)
    #print (XH2O_melt,XCO2_melt)

    
    #print (XH2O_melt,'vs',mantle_H2O_mass/mantle_mass)
    #print (XCO2_melt,'vs',mantle_CO2_mass/mantle_mass)
    #[PH2O, PH2, PCO2, PCO, PCH4, alphaG, xCO2, xH2O, mH2O,mH2,mCO2,mCO,mCH4,mO2] = solve_gases(Temp,Pressure/1e5,pO2,mantle_CO2_mass/mantle_mass,mantle_H2O_mass/mantle_mass)
    [PH2O, PH2, PCO2, PCO, PCH4, alphaG, xCO2, xH2O, mH2O,mH2,mCO2,mCO,mCH4,mO2] = solve_gases(Temp,Pressure/1e5,pO2,XCO2_melt,XH2O_melt)
    #print('outs',[PH2O, PH2, PCO2, PCO, PCH4, alphaG, xCO2, xH2O, mH2O,mH2,mCO2,mCO,mCH4,mO2])
    if alphaG<0:
        print ('-ve alphaG, outgassing assumed to be zero!')
        return [0.0,0.0,0.0,0.0,0.0,0.0]
    xm = 1.55e-2 #mol magma / g magma
    q_H2O = mH2O * alphaG * xm / (1-alphaG) #mol gas/g magma 
    q_CO2 = mCO2 * alphaG * xm / (1-alphaG)
    q_H2 = mH2 * alphaG * xm / (1-alphaG)
    q_CO = mCO * alphaG * xm / (1-alphaG)
    q_CH4 = mCH4 * alphaG * xm / (1-alphaG)


    if alphaG>1.0: # quick fudge, hopefully will go away when make sure rock pressure used for melt calculation (fixed version):
        #import pdb
        #pdb.set_trace()
        F_H2O = 0.0#XH2O_melt*M_MELT * mH2O/(mH2O+mH2+2*mCH4)
        F_CO2 = 0.0#XCO2_melt * M_MELT * mCO2/(mCO2+mCO)
        F_CO = 0.0#XCO2_melt * M_MELT * mCO/(mCO2+mCO)
        F_H2 = 0.0#XH2O_melt*M_MELT * mH2/(mH2O+mH2+2*mCH4)
        F_CH4 = 0.0#XH2O_melt*M_MELT * 2*mCH4/(mH2O+mH2+2*mCH4)
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

