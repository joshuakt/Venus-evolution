#######################################
import numpy as np
import pylab
from scipy.interpolate import interp1d
from scipy import optimize
from numba import jit
#######################################

@jit(nopython=True)
def qr(t00,Start_time,heatscale,K_over_U):  #interior radiogenic heatproduction
    t = t00 - Start_time*365*24*60*60
    u238 = 0.9928 * (0.022e-6) * 9.17e-5 * np.exp(-(np.log(2)/4.46e9)*(t/(365*24*60*60)-4.5e9))
    u235 = 0.0072 * (0.022e-6) * 5.75e-4 * np.exp(-(np.log(2)/7.04e8)*(t/(365*24*60*60)-4.5e9))
    Th = (0.069e-6) * 2.56e-5 * np.exp(-(np.log(2)/1.4e10)*(t/(365*24*60*60)-4.5e9))
    K_abun = K_over_U * 0.022e-6 * heatscale
    K = 1.17e-4 * K_abun * 2.97e-5 * np.exp(-(np.log(2)/1.26e9)*(t/(365*24*60*60)-4.5e9))
    Al = 5e-5 * 3.54e-1 * (8650e-6) * np.exp(-(np.log(2)/7.17e5)*(t/(365*24*60*60)))   
    return (heatscale*u238 + heatscale*u235 + heatscale*Th + K + Al) #(2./3. )*

@jit(nopython=True)
def viscosity_fun(Tp,pm,visc_offset,Tsurf,Tsolidus): #calcuate mantle viscosity
    visc_rock = visc_offset*3.8e7 * np.exp(350000/(8.314*Tp))/pm 
    visc_liq = 0.00024*np.exp(4600.0 / (Tp - 1000.0))/pm   
    LowerT = Tsolidus 
    UpperT = LowerT + 600.0    
    if Tp < LowerT+0.000:
        visc = visc_rock
    elif Tp > UpperT+0.000:
        visc = visc_liq
    else:
        v1 = np.log10(visc_rock)
        v2 = np.log10(visc_liq)
        logvisc = ( v2 * 0.2*(Tp - (LowerT))**5 + v1 * 0.8*(UpperT - Tp)**5) / ( 0.2*(Tp - (LowerT))**5 + 0.8*(UpperT - Tp)**5)
        visc = 10**logvisc
    return visc

@jit(nopython=True)
def f_for_optim(x,kH2O,Mcrystal,Mliq,rp,yH2O_liq,g,MMW):
    result = (kH2O * x * Mcrystal + x * (Mliq-Mcrystal) + 4.0 *(0.018/MMW)* np.pi * (rp**2/g) * (x / 3.44e-8)**(1.0/0.74) - yH2O_liq) 
    return result

#@jit(nopython=True)
def H2O_partition_function( yH2O_liq,Mliq,Mcrystal,rp,g,kH2O,MMW):  #partition H2O between magma ocean and atmosphere
    if (Mliq >0)or(Mliq>0):
        FH2O = optimize.newton(f_for_optim,0.5,args=(kH2O,Mcrystal,Mliq,rp,yH2O_liq,g,MMW))
        Pressure_surface = (FH2O / 3.44e-8)**(1.0/0.74)
    else:
        FH2O = 3.44e-8*( yH2O_liq/(4 * (0.018/MMW) * np.pi * (rp**2/g)) ) ** (0.74)
        Pressure_surface = (FH2O / 3.44e-8)**(1.0/0.74)
    return [FH2O,Pressure_surface]

@jit(nopython=True)
def CO2_partition_function( yCO2_liq,Mliq,Mcrystal,rp,g,kCO2,MMW): #partition CO2 between magma ocean and atmosphere
    if (Mliq>0)or(Mcrystal>0):
        FCO2 = yCO2_liq / (kCO2 * Mcrystal + (Mliq-Mcrystal) + 4 * (0.044/MMW) * np.pi * (rp**2/g) * (1 /4.4e-12)) 
        Pressure_surface = (FCO2 /4.4e-12)
    else:
        FCO2 = 0.0
        Pressure_surface = (yCO2_liq*g)/(4.0 *(0.044/MMW)* np.pi * (rp**2))
    return [FCO2,Pressure_surface]

@jit(nopython=True)
def Mliq_fun(y2,rp,rs,pm): #calculate liquid mass of mantle
    if rs < rp:
        Mliq = pm * 4./3. * np.pi * (rp**3 - rs**3)
    else:
        Mliq = 0.0
    return Mliq
    
@jit(nopython=True)    
def rs_term_fun(r,a1,b1,a2,b2,g,alpha,cp,pm,rp,Tp,Poverburd): #controls solidification radius evolution
    numerator = 1 + alpha*(g/cp) * (rp - r)
    e1 = np.exp(1e-5*(-rp+r+100000))
    e2 = np.exp(1e-5*(rp-r-100000))
    sub_denom = (e1+e2)**2
    T1 = (b1+a1*g*pm*(rp-r)+a1*Poverburd)
    T2 = (b2+a2*g*pm*(rp-r)+a2*Poverburd)
    sub_num1 = ((-a1*g*pm*e1 + T1*1e-5*e1) + (-a2*g*pm*e2 - T2*1e-5*e2))*(e1+e2) 
    sub_num2 = (T1*e1+T2*e2)*(1e-5*e1-1e-5*e2)
    everything = (sub_num1 - sub_num2)/sub_denom
    if r>rp:
        return 0
    else:
        return numerator /(alpha*g*Tp/cp + everything)

@jit(nopython=True)     
def adiabat(radius,Tp,alpha,g,cp,rp):   #mantle adiabat
    Tr = Tp*(1 + alpha*(g/cp)*(rp-radius))
    return Tr   

@jit(nopython=True)    
def sol_liq(radius,g,pm,rp,Poverburd,mH2O): #For calculating solidus
    a1 = 104.42e-9
    b1 = 1420+0.000-80.0 
    TS1 = b1 + a1 * g * pm * (rp - radius) + a1 * Poverburd - 4.7e4 * mH2O**0.75
    
    if TS1 < 1170.0:
        T_sol1 = 1170.0+0*TS1
    else:
        T_sol1 = TS1
    a2 = 26.53e-9
    b2 = 1825+0.000  
    TS2 = b2 + a2 * g * pm * (rp -radius)  + a2 * Poverburd - 4.7e4 * mH2O**0.75
    if TS2 < 1170.0:
        T_sol2 = 1170.0+0*TS2
    else:
        T_sol2 = TS2
    T_sol = (T_sol1 * np.exp(1e-5*( -rp + radius + 100000)) + T_sol2 * np.exp(1e-5*(rp - radius - 100000)))/ (np.exp(1e-5*(-rp + radius + 100000)) +  np.exp(1e-5*(rp - radius - 100000)))  
    return T_sol


#@jit(nopython=True)
def find_r(r,Tp,alpha,g,cp,pm,rp,Poverburd,mH2O,delta): ## used for finding the solidus radius numerically
    Tr = adiabat(r,Tp,alpha,g,cp,rp-delta)
    rr = float(r)
    T_sol = sol_liq(rr,g,pm,rp,Poverburd,mH2O)
    return (Tr-T_sol)**2.0        

def find_r2(r,Tp,alpha,g,cp,pm,rp,Poverburd,mH2O,delta,depletionfraction): ## used for finding the solidus radius accounting for depletion fraction
    Tr = adiabat(r,Tp,alpha,g,cp,rp-delta)
    rr = float(r)
    T_sol = sol_liq(rr,g,pm,rp,Poverburd,mH2O)+600*depletionfraction
    return (Tr-T_sol)**2.0   
    
@jit(nopython=True)
def viscosity_melt(Tp,pm,Tsol,Tliq,melt_frac):
    visc_rock = (80e9 / (2*5.3e15)) * (1e-3 / 0.5e-9)**2.5 * np.exp((240e3+0*100e3)/(8.314*Tp)) * np.exp( - 26 * melt_frac)/pm 
    if melt_frac <= 0.4:
        visc = visc_rock
    else:
        visc_liq = 0.00024*np.exp(4600.0 / (Tp - 1000.0))/pm      
        visc_liq = visc_liq / (1 - (1-melt_frac)/(1-0.4))**2.5
        if float(visc_liq) < float(visc_rock):
            visc = float(visc_liq)
        else:
            visc = float(visc_rock)
    return visc

@jit(nopython=True)    
def temp_meltfrac(rc,rp,alpha,pm,Tp,cp,g,Poverburd,mH2O,rlid):  #for calculating melt fraction
    rad = np.linspace(rc,rlid,1000)
    melt_r = np.copy(rad)
    visc_r = np.copy(rad)
    vol_r = np.copy(rad)*0 + 4.0*np.pi*(rad[1]-rad[0])
    for j in range(0,len(rad)):
        Tsol = sol_liq(float(rad[j]),g,pm,rp,Poverburd,mH2O)
        Tliq = Tsol + 600.0
        T_r = adiabat(rad[j],Tp,alpha,g,cp,rlid) 
        if T_r>Tliq:
            melt_r[j] = 1.0
            visc_r[j] = 1 
            vol_r[j] = vol_r[j]*rad[j]**2
        elif T_r<Tsol:
            melt_r[j] = 0.0
            visc_r[j] = 1 
            vol_r[j] = 0.0
        else:
            melt_r[j] = (T_r - Tsol)/(Tliq - Tsol)
            visc_r[j] = 1 
            vol_r[j] = vol_r[j]*rad[j]**2
    if np.sum(vol_r) == 0.0:
        return (0.0,0.0,0.0)
    actual_phi_surf = np.sum(melt_r*vol_r)/np.sum(vol_r) 
    Va = np.sum(vol_r)
    actual_visc = np.sum(visc_r*vol_r)/np.sum(vol_r)
    return (actual_phi_surf,actual_visc,Va)


@jit(nopython=True)    
def temp_meltfrac2(rc,rp,alpha,pm,Tp,cp,g,Poverburd,mH2O,rlid,depletion_fraction): #for calculating melt fraction accounting for depletion fraction
    rad = np.linspace(rc,rlid,1000)
    melt_r = np.copy(rad)
    visc_r = np.copy(rad)
    vol_r = np.copy(rad)*0 + 4.0*np.pi*(rad[1]-rad[0])
    for j in range(0,len(rad)):
        Tsol_og = sol_liq(float(rad[j]),g,pm,rp,Poverburd,mH2O)
        Tsol = Tsol_og + 600*depletion_fraction
        Tliq = Tsol_og + 600.0
        T_r = adiabat(rad[j],Tp,alpha,g,cp,rlid) 
        if T_r>Tliq:
            melt_r[j] = 1.0
            visc_r[j] = 1
            vol_r[j] = vol_r[j]*rad[j]**2
        elif T_r<Tsol:
            melt_r[j] = 0.0
            visc_r[j] = 1
            vol_r[j] = 0.0
        else:
            melt_r[j] = (T_r - Tsol)/(Tliq - Tsol) 
            visc_r[j] = 1
            vol_r[j] = vol_r[j]*rad[j]**2
    if np.sum(vol_r) == 0.0:
        return (0.0,0.0,0.0)
    actual_phi_surf = np.sum(melt_r*vol_r)/np.sum(vol_r) 
    Va = np.sum(vol_r)
    actual_visc = np.sum(visc_r*vol_r)/np.sum(vol_r)
    return (actual_phi_surf,actual_visc,Va)

