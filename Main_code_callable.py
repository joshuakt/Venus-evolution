# This script contains the forward model

#####################
import numpy as np
import pylab
from scipy.integrate import *
from scipy.interpolate import interp1d
from scipy import optimize
import pdb
import scipy.optimize 
from radiative_functions import *
from other_functions import *
from stellar_funs import main_sun_fun
from carbon_cycle_model import *
from escape_functions import *
from all_classes import *
#from outgassing_module import *
from outgassing_module_fast import *
from Albedo_module import *
from thermodynamic_variables import *
import time
from numba import jit
#####################

def forward_model(Switch_Inputs,Planet_inputs,Init_conditions,Numerics,Stellar_inputs,MC_inputs,max_time_attempt):

   
    plot_switch = "n" # change to "y" to plot individual model runs for diagnostic purposes
    print_switch = Switch_Inputs.print_switch # This controls whether outputs print during calculations (slows things down, but useful for diagnostics)
    speedup_flag = Switch_Inputs.speedup_flag # Redundant - does not do anything in current version
    start_speed =  Switch_Inputs.start_speed # Redundant - does not do anything in current version
    fin_speed = Switch_Inputs.fin_speed # Redundant - does not do anything in current version
    heating_switch = Switch_Inputs.heating_switch # Controls locus of internal heating, keep default values
    C_cycle_switch = Switch_Inputs.C_cycle_switch  # Turns carbon cycle on or off, keep default values
    
    RE = Planet_inputs.RE #Planet radius relative Earth
    ME = Planet_inputs.ME  #Planet mass relative Earth
    pm = Planet_inputs.pm  #Average mantle density
    rc = Planet_inputs.rc #Metallic core radius (m)
    Total_Fe_mol_fraction = Planet_inputs.Total_Fe_mol_fraction # iron mol fraction in mantle
    
    Planet_sep = Planet_inputs.Planet_sep #planet-star separation (AU)
    albedoC = Planet_inputs.albedoC  #cold state albedo   
    albedoH = Planet_inputs.albedoH   #hot state albedo

    #Stellar parameters
    tsat_XUV = Stellar_inputs.tsat_XUV  #XUV saturation time
    Stellar_Mass = Stellar_inputs.Stellar_Mass #stellar mass (relative sun)
    fsat = Stellar_inputs.fsat
    beta0 = Stellar_inputs.beta0
    epsilon = Stellar_inputs.epsilon

    #generate random seed for this forward model call    
    np.random.seed(int(time.time()))
    seed_save = np.random.randint(1,1e9)

    ## Initial volatlie and redox conditions:
    Init_solid_H2O = Init_conditions.Init_solid_H2O
    Init_fluid_H2O = Init_conditions.Init_fluid_H2O
    Init_solid_O= Init_conditions.Init_solid_O
    Init_fluid_O = Init_conditions.Init_fluid_O
    Init_solid_FeO1_5 = Init_conditions.Init_solid_FeO1_5
    Init_solid_FeO = Init_conditions.Init_solid_FeO
    Init_fluid_CO2 = Init_conditions.Init_fluid_CO2
    Init_solid_CO2= Init_conditions.Init_solid_CO2

    #Oxidation parameters
    wet_oxid_eff = MC_inputs.interiord
    MFrac_hydrated = MC_inputs.interiorb
    dry_oxid_frac = MC_inputs.interiorc 
    surface_magma_fr = MC_inputs.surface_magma_frac  

    #ocean chemistry and weathering parameters
    ocean_Ca = MC_inputs.ocean_a 
    omega_ocean = MC_inputs.ocean_b 
    efold_weath = MC_inputs.ccycle_a
    alpha_exp = MC_inputs.ccycle_b
    supp_lim = MC_inputs.supp_lim

    #Escape parameters
    mult = MC_inputs.esc_c 
    mix_epsilon = MC_inputs.esc_d 
    Te_input_escape = MC_inputs.Tstrat

    #Interior parameters
    transition_time_stag = MC_inputs.interiorg
    visc_offset = MC_inputs.interiora 
    heatscale = MC_inputs.interiore

    #impact parameters
    imp_coef = MC_inputs.esc_a 
    tdc = MC_inputs.esc_b 


    MEarth = 5.972e24 #Mass of Earth (kg)
    kCO2 = 2e-3 #Crystal-melt partition coefficent for CO2
    #kCO2 = 0.99  #sensitivity test reduced mantle (CO2 retained in interior)
    G = 6.67e-11 #gravitational constant
    cp = 1.2e3 # silicate heat capacity
    rp = RE * 6.371e6 #Planet radius (m)
    Mp = ME * MEarth  #Planet mass (kg)
    delHf = 4e5 #Latent heat of silicates
    g = G*Mp/(rp**2) # gravity (m/s2)
    Tsolidus = sol_liq(rp,g,pm,rp,0.0,0.0)  #Solidus for magma ocean evolution
    Tliquid = Tsolidus + 600 #Liquidus for magma ocean evolution
    alpha = 2e-5 #Thermal expansion coefficient (per K)
    k = 4.2  #Thermal conductivity, W/m/K
    kappa = 1e-6  #Thermal diffusivity of silicates, m2/s
    Racr = 1.1e3 #Critical Rayeligh number
    kH2O = 0.01  #Crystal-melt partition coefficent for water
    a1 = 104.42e-9 #Solidus coefficient
    b1 = 1420+0.000-80.0 #Solidus coefficient
    a2 = 26.53e-9 #Solidus coefficient
    b2 = 1825+0.000 #Solidus coefficient

    min_Te = 150.0 ## Minimum Te for purposes of OLR/ASR calculations and escape calculations
    min_ASR = 5.67e-8 * (min_Te/(0.5**0.25))**4.0  ## Minimum Absorbed Shortwave Radiation (ASR)
    TMoffset = 0.0 ## Redundant
    min_Te = 207.14285714 # Threshold to prevent skin temperature from getting too low where OLR grid contains errors. Note this lower limit does not apply to stratosphere temperatures used for escape calculations.

    # Define radiogenic inventory
    Uinit = heatscale*22e-9 #Uranium abundance
    K_over_U = MC_inputs.K_over_U #K/U ratio 
    core_flow_coefficient = 12e12   #Assumed core heatflow 

    K40_over_K = 1.165e-4
    lam_Ar = 0.0581
    lam_Ca = 0.4962
    init40K = (pm * 4.*np.pi * (rp**3 - rc**3)/3.0) * Uinit * K_over_U *K40_over_K * np.exp (4.5*(lam_Ar+lam_Ca))  
    initAr40 = 0.0

    init_U238 = (pm * 4.*np.pi * (rp**3 - rc**3)/3.0) * 0.9928 * Uinit * np.exp((np.log(2)/4.46e9)*(4.5e9))
    init_U235 = (pm * 4.*np.pi * (rp**3 - rc**3)/3.0) * 0.0072 * Uinit * np.exp((np.log(2)/7.04e8)*(4.5e9))
    init_Th =  (pm * 4.*np.pi * (rp**3 - rc**3)/3.0) *(0.069e-6) * heatscale * np.exp((np.log(2)/1.4e10)*(4.5e9)) 
    init_He_mantle = 0.0
    init_He_atmo = 0.0
    He_escape_loss = 0.0

    ## D/H initialization - D/H evolution has not been fully implemented, keep switched off.
    Init_D_to_H = 1.4e-4
    DH_switch = 0.0 # 0 = off, 1 = on

    Max_mantle_H2O = 1.4e21 * MC_inputs.interiorf * (rp**3 - rc**3) / ((6.371e6)**3 - (3.4e6)**3) ## Max mantle water content (kg)
    
    Start_time = Switch_Inputs.Start_time #Model start time (relative to stellar evolution track)
    Max_time=np.max([Numerics.tfin0,Numerics.tfin1,Numerics.tfin2,Numerics.tfin3,Numerics.tfin4]) #Model end time
    test_time = np.linspace(Start_time*365*24*60*60,Max_time*365*24*60*60,10000)
    new_t = np.linspace(Start_time/1e9,Max_time/1e9,100000)

    [Relative_total_Lum,Relative_XUV_lum,Absolute_total_Lum,Absolute_XUV_Lum] = main_sun_fun(new_t,Stellar_Mass,tsat_XUV,beta0,fsat) #Calculate stellar evolution
    ASR_new = (Absolute_total_Lum/(16*3.14159*(Planet_sep*1.496e11)**2) )  #ASR flux through time (not accounting for bond albedo)
    
    for ij in range(0,len(ASR_new)):  # do not permit ASR outside of interpolation grid
        if ASR_new[ij] < min_ASR:
            ASR_new[ij] = min_ASR
    Te_ar = (ASR_new/5.67e-8)**0.25
    Tskin_ar = Te_ar*(0.5**0.25) ## Skin temperature through time
    for ij in range(0,len(Tskin_ar)): #Don't permit skin temperature to exceed range min_Te - 350 due to errors in grid (does not apply to stratospheric temperature used to calculate escape fluxes)
        if Tskin_ar[ij] > 350:
            Tskin_ar[ij] = 350.0
        if Tskin_ar[ij] < min_Te:
            Tskin_ar[ij] = min_Te
    Te_fun = interp1d(new_t*1e9*365*24*60*60,Tskin_ar) #Skin temperature function, used in OLR calculations
    ASR_new_fun = interp1d(new_t*1e9*365*24*60*60, ASR_new) #ASR function, used to calculate shortwave radiation fluxes through time
    AbsXUV = interp1d(new_t*1e9*365*24*60*60 , Absolute_XUV_Lum/(4*np.pi*(Planet_sep*1.496e11)**2)) #XUV function, used to calculate XUV-driven escape

    #@jit(nopython=True) # function for finding surface temperature that balances ASR and interior heatflow
    def funTs_general(Ts,Tp,ll,visc,beta,Te_input,ASR_input,H2O_Pressure_surface,CO2_Pressure_surface,ocean_CO3,MMW,PO2_surf): 
        if max(Ts)<Tp:
            Ra =alpha * g * (Tp -Ts) * ll**3 / (kappa * visc)
            qm = (k/ll) * (Tp - Ts) * (Ra/Racr)**beta
        else:
            qm = - 2.0 * (Ts - Tp) / (rp-rc)        
        Ts_in= max(Ts)
        [OLR_new,newpCO2,ocean_pH,ALK,Mass_oceans_crude,DIC_check] = correction(Ts_in,Te_input,H2O_Pressure_surface,CO2_Pressure_surface,rp,g,ocean_CO3,1e5,MMW,PO2_surf) 
        heat_atm = OLR_new - ASR_input 
        return (qm - heat_atm)**2      

    @jit(nopython=True) # as above but precompiled
    def funTs_general2(Ts,Tp,ll,visc,beta,Te_input,ASR_input,H2O_Pressure_surface,CO2_Pressure_surface,ocean_CO3,MMW,PO2_surf): 
        if Ts[0]<Tp:
            Ra =alpha * g * (Tp -Ts[0]) * ll**3 / (kappa * visc)
            qm = (k/ll) * (Tp - Ts[0]) * (Ra/Racr)**beta
        else:
            qm = - 2.0 * (Ts[0] - Tp) / (rp-rc)        
        Ts_in= Ts[0]
        [OLR_new,newpCO2,ocean_pH,ALK,Mass_oceans_crude,DIC_check] = correction(Ts_in,Te_input,H2O_Pressure_surface,CO2_Pressure_surface,rp,g,ocean_CO3,1e5,MMW,PO2_surf) 
        heat_atm = OLR_new - ASR_input 
        return -(qm - heat_atm)**2 

    #Radiation balance accommodating stagnant lid and plate tectonics:
    def funTs_general_stag(Ts,Tp,ll,visc,beta,Te_input,ASR_input,H2O_Pressure_surface,CO2_Pressure_surface,ocean_CO3,MMW,PO2_surf,MV,PT): 
        if ((max(Ts)<Tp)and(PT < 1)):
            Rai = alpha * g * (Tp -Ts) * ll**3 / (kappa * visc)
            theta = 300000*(Tp - Ts)/(8.314*Tp**2)
            qa = 0.5 * k/ll * (Tp - Ts) * theta**(-4.0/3.0) * Rai**(1./3.)
            deltaTm = Tp -Ts
            volc_term = MV*3000*(cp*deltaTm + delHf)
            qm = qa + volc_term/(4*np.pi*rp**2)
        elif ((max(Ts)<Tp)and(PT >= 1)):
            Ra =alpha * g * (Tp -Ts) * ll**3 / (kappa * visc)
            qm = (k/ll) * (Tp - Ts) * (Ra/Racr)**beta          
        else:
            qm = - 2.0 * (Ts - Tp) / (rp-rc)        
        Ts_in= max(Ts)
        [OLR_new,newpCO2,ocean_pH,ALK,Mass_oceans_crude,DIC_check] = correction(Ts_in,Te_input,H2O_Pressure_surface,CO2_Pressure_surface,rp,g,ocean_CO3,1e5,MMW,PO2_surf) 
        heat_atm = OLR_new - ASR_input 
        return (qm - heat_atm)**2    

    @jit(nopython=True) # as above but precompiled
    def funTs_general_stag2(Ts,Tp,ll,visc,beta,Te_input,ASR_input,H2O_Pressure_surface,CO2_Pressure_surface,ocean_CO3,MMW,PO2_surf,MV,PT): 
        if Ts[0]<Tp:#and(PT<1.0)):
            if PT < 1.0:
                Rai = alpha * g * (Tp -Ts[0]) * ll**3 / (kappa * visc)
                theta = 300000*(Tp - Ts[0])/(8.314*Tp**2)
                qa = 0.5 * k/ll * (Tp - Ts[0]) * theta**(-4.0/3.0) * Rai**(1./3.)
                deltaTm = Tp -Ts[0]
                volc_term = MV*3000*(cp*deltaTm + delHf)
                qm = qa + volc_term/(4*np.pi*rp**2)
            if PT  >= 1.0:
                Ra =alpha * g * (Tp -Ts[0]) * ll**3 / (kappa * visc)
                qm = (k/ll) * (Tp - Ts[0]) * (Ra/Racr)**beta                
        else:
            qm = - 2.0 * (Ts[0] - Tp) / (rp-rc)        
        Ts_in= Ts[0]
        [OLR_new,newpCO2,ocean_pH,ALK,Mass_oceans_crude,DIC_check] = correction(Ts_in,Te_input,H2O_Pressure_surface,CO2_Pressure_surface,rp,g,ocean_CO3,1e5,MMW,PO2_surf) 
        heat_atm = OLR_new - ASR_input 
        return -(qm - heat_atm)**2     

    
    #@jit(nopython=True) # alternative function for finding surface temperature that balances ASR and interior heatflow (does exact same thing, hardly used)
    def funTs_scalar(Ts,Tp,ll,visc,beta,Te_input,ASR_input,H2O_Pressure_surface,CO2_Pressure_surface,ocean_CO3,MMW,PO2_surf): 
        if Ts<Tp:
            Ra =alpha * g * (Tp -Ts) * ll**3 / (kappa * visc)
            qm = (k/ll) * (Tp - Ts) * (Ra/Racr)**beta           
        else:
            qm = - 2.0 * (Ts - Tp) / (rp-rc)        
        Ts_in= Ts
        [OLR_new,newpCO2,ocean_pH,ALK,Mass_oceans_crude,DIC_check] = correction(Ts_in,Te_input,H2O_Pressure_surface,CO2_Pressure_surface,rp,g,ocean_CO3,1e5,MMW,PO2_surf)  
        heat_atm =   OLR_new - ASR_input   
        return (qm - heat_atm)**2       

    def FeO_mass_frac(Total_Fe): # Convert total iron mole fraction to mass fraction
        XAl2O3 = 0.022423 
        XCaO = 0.0335 
        XNa2O = 0.0024 
        XK2O = 0.0001077 
        XMgO = 0.478144  
        XSiO2 =  0.4034    
        m_sil = XMgO * (16.+25.) + XSiO2 * (28.+32.) + XAl2O3 * (27.*2.+3.*16.) + XCaO * (40.+16.) +  (Total_Fe) * (56.0+16.0)
        return (Total_Fe) * (56.0+16.0)/m_sil        
    Total_Fe_mass_fraction = FeO_mass_frac(Total_Fe_mol_fraction)             

    global solid_switch,phi_final, ynewIC,switch_counter,speedup,liquid_switch,solid_counter,last_magma
    # Define switches used to keep track of volatile transfers between magma ocean and solid mantle convection phases (and vice versa)
    solid_switch = 0
    solid_switch = 1 #for starting as solid only
    liquid_switch = 0
    liquid_switch_worked = 0
    speedup = 0.0
    switch_counter = 0
    solid_counter = 0
    last_magma = 0.0

    model_run_time = time.time()

    def system_of_equations(t0,y):
        PltTech = 1 #Plate tectonics marker

        global last_magma
        crt = t0/(365*24*60*60) 
        if transition_time_stag > 0: # if negative, always plate tectoncis
            if (crt > transition_time_stag)and((crt - last_magma)>50e6):
                PltTech = 0 # Denotes stagnant lid
        ## For experimenting with plate tectonics regime change
        #if ((crt >3.9e9) or ((crt < 3.0e9)and(crt>2.0e9)) )and((crt - last_magma)>50e6):# switching
        #if (crt > 13e6)and((crt - last_magma)>50e6):# and (t0/(365*24*60*60) < 20e6): #staggers
        #if (crt > 3.9e9)and((crt - last_magma)>50e6):# and (crt < 20e6): # PT until 4.0 Ga
        #    PltTech = 0

        ## For experimenting with artifical heatflow injection
        Qr_cofactor = 1.0
        #if (crt >3.5e9)and(crt<4.0e9):
        #    Qr_cofactor = 3.0
        #if (crt >3.9e9)and(crt<3.95e9): #more realistic LIP, injects 50 TW for 50 Myrs USING THIS ONE
        #    Qr_cofactor = 4.0

        tic = time.time()
        if (tic - model_run_time)>60*60*max_time_attempt: #limit maximum time spent before abandon attempt
            print ("TIMED OUT")
            return np.nan*[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0] 

        Va = 0.0
        actual_visc = 0.0 
        global solid_switch,speedup,liquid_switch,liquid_switch_worked
        global phi_final,ynewIC,switch_counter,solid_counter
       
        depletion_fraction = y[52] 
        speedup=0.0

        ocean_CO3 = float(omega_ocean * Sol_prod(y[8]) / ocean_Ca)
        Mantle_mass = (4./3. * np.pi * pm * (y[2]**3 - rc**3)) #Mass solid mantle
        Mantle_mass0 = (4./3. * np.pi * pm * (rp**3 - rc**3)) #Mass solid mantle
        Vmantle0 = (4.0*np.pi/3.0) * (rp**3 - rc**3) #Volume total mantle

        if print_switch == "y":
            print (t0/(365*24*60*60))

        #################################################################################
        #### If in magma ocean phase
        if  (y[8] > Tsolidus):
            last_magma = t0/(365*24*60*60)
            beta = 1./3. #Convective heatflow exponent
            if print_switch== "y":
                print('still molten',t0/(365*24*60*60),np.transpose(y))        

            #For switching from solid to magma ocean           
            if liquid_switch == 1:
                if y[2]+1 < rp:
                    liquid_switch_worked = 1.0
                    liquid_switch = 0.0
                else:
                    T_partition = np.max([y[7],Tsolidus+0.01])
                    rad_check = optimize.minimize(find_r,x0=float(y[2]),args = (T_partition,alpha,g,cp,pm,rp,0.0,0.0,0.0))
                    y[2] = np.max([rad_check.x[0],0.0])
        
            #Calculate surface melt fraction
            if y[8] > Tliquid:
                actual_phi_surf = 1.0
            elif y[8] < Tsolidus:
                actual_phi_surf = 0.0
            else:
                actual_phi_surf =( y[8] -Tsolidus)/(Tliquid - Tsolidus)

            ll = np.max([rp - y[2],1.0])  ## length scale is depth of magma ocean pre-solidification (even if melt <0.4)
            Qr = Qr_cofactor*qr(t0,Start_time,heatscale,K_over_U)  
            Qcore = Qr_cofactor*np.exp(-(t0/(1e9*365*24*60*60)-4.5)/7.0)*core_flow_coefficient

            Mliq = Mliq_fun(y[1],rp,y[2],pm)
            Mcrystal = (1-actual_phi_surf)*Mliq
            phi_final = actual_phi_surf
            [FH2O,H2O_Pressure_surface] = H2O_partition_function( y[1],Mliq,Mcrystal,rp,g,kH2O,y[24])
            [FCO2,CO2_Pressure_surface] = CO2_partition_function( y[12],Mliq,Mcrystal,rp,g,kCO2,y[24]) #molten so can ignore aqueous CO2
            AB = AB_fun(float(y[8]),H2O_Pressure_surface,float(y[1]+y[4]+y[12]),albedoC,albedoH)

            Tsolidus_Pmod = sol_liq(rp,g,pm,rp,float(H2O_Pressure_surface+CO2_Pressure_surface+1e5),float(0*y[0]/Mantle_mass0))
            Tsolidus_visc = sol_liq(rp,g,pm,rp,float(H2O_Pressure_surface+CO2_Pressure_surface+1e5),0.0)     
            visc =  viscosity_fun(y[7],pm,visc_offset,y[8],float(Tsolidus_visc))      

            if print_switch=="y":
                print ('visc',visc)
            if np.isnan(visc):
                return [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0] 

            ASR_input = float((1-AB)*ASR_new_fun(t0))
            if ASR_input < min_ASR:
                ASR_input = min_ASR            
            Te_ar = (ASR_input/5.67e-8)**0.25
            Te_input = Te_ar*(0.5**0.25)
            if Te_input > 350:
                Te_input = 350.0
            if Te_input < min_Te:
                Te_input = min_Te        

            if (2>1):
                initialize_fast = np.min([y[8],y[7]+TMoffset,y[7]+TMoffset-1])
                if y[8] > y[7]+TMoffset:
                    initialize_fast = y[8]

                initialize_fast = np.array(initialize_fast)
                ace1 =  nelder_mead(funTs_general2, x0=initialize_fast, bounds=np.array([[100.0], [4500.0]]).T, args = (float(y[7]),ll,visc,beta,Te_input,ASR_input,H2O_Pressure_surface,CO2_Pressure_surface,ocean_CO3,float(y[24]),float(y[22])), tol_f=0.0001,tol_x=0.0001, max_iter=1000)
                SurfT = ace1.x[0]
                new_abs = abs(ace1.fun)
                if new_abs > 0.1:

                    lower = 180.0
                    upper = float(y[7]+TMoffset)+10

                    ace1b=scipy.optimize.minimize_scalar(funTs_scalar, args=(float(y[7]+TMoffset),ll,visc,beta,Te_input,ASR_input,H2O_Pressure_surface,CO2_Pressure_surface,ocean_CO3,float(y[24]),float(y[22])),bounds=[lower,upper],tol=1e-10,method='bounded',options={'maxiter':1000,'xatol':1e-10})

                    if abs(ace1b.fun) < new_abs:
                        SurfT = float(ace1b.x)
                        new_abs = abs(ace1b.fun)
                    
                
                y[8] = SurfT
              

                if new_abs > 1.0:
                    ace2= optimize.minimize(funTs_general,x0=y[8],args = (float(y[7]),ll,visc,beta,Te_input,ASR_input,H2O_Pressure_surface,CO2_Pressure_surface,ocean_CO3,float(y[24]),float(y[22])),method='L-BFGS-B',bounds = ((150,1500),))
                    if abs(ace2.fun) <  new_abs:
                        SurfT = ace2.x[0]
                        new_abs = abs(ace2.fun) 
                    if new_abs > 1.0:
                        rand_start = 150 + 1850*np.random.uniform()
                        ace3= optimize.minimize(funTs_general,x0=rand_start,args = (float(y[7]),ll,visc,beta,Te_input,ASR_input,H2O_Pressure_surface,CO2_Pressure_surface,ocean_CO3,float(y[24]),float(y[22])),method='L-BFGS-B',bounds = ((150,2000),))
                        if abs(ace3.fun) <  new_abs:
                            SurfT =ace3.x[0]     
                            new_abs = abs(ace3.fun)   

                if new_abs > 1.0:
                    differ_ace = 10.0
                    counter = 0

                    initialize_fast = np.array(initialize_fast)
                    budget_differ = 10.0
                    aceCOMP =  nelder_mead(funTs_general2, x0=initialize_fast, bounds=np.array([[150.0], [4500.0]]).T, args = (float(y[7]),ll,visc,beta,Te_input,ASR_input,H2O_Pressure_surface,CO2_Pressure_surface,ocean_CO3,float(y[24]),float(y[22])), tol_f=1e-10,tol_x=1e-10, max_iter=1000)

                    if abs(aceCOMP.fun)<new_abs:
                        SurfT = aceCOMP.x[0]
                    budget_differ = aceCOMP.fun
                    budget_counter = 0
                    while abs(budget_differ) > 0.1:
                        budget_counter = budget_counter + 1
                        ran_num = np.min([150 + 2000*np.random.uniform(),3999])
                        ran_numar=np.array(ran_num)
                        aceCOMP2 =  nelder_mead(funTs_general2, x0=ran_numar, bounds=np.array([[150.0], [4500.0]]).T, args = (float(y[7]),ll,visc,beta,Te_input,ASR_input,H2O_Pressure_surface,CO2_Pressure_surface,ocean_CO3,float(y[24]),float(y[22])), tol_f=1e-10,tol_x=1e-10, max_iter=2000)

                        if abs(aceCOMP2.fun)<abs(budget_differ):
                            SurfT = aceCOMP2.x[0]     
                            budget_differ = aceCOMP2.fun 
                        if budget_counter == 10:
                            budget_differ = -2e-3  
                                                                                                                                                                                  
                Ra =np.max([-1e15,alpha * g * (y[7]+TMoffset -SurfT) * ll**3 / (kappa * visc)  ])
                qm = (k/ll) * (y[7]+TMoffset - SurfT) * (Ra/Racr)**beta
                
                thermal_cond = 2 # W/m2/K
                qc = thermal_cond * (SurfT - y[7]- TMoffset) / (rp-rc)                      
                delta_u = k * (y[7]+TMoffset - SurfT) / qm
                
            ## check to see if anywhere near solidus
            if adiabat(rc,y[7],alpha,g,cp,rp) > sol_liq(rc,g,pm,rp,0.0,0.0): #ignore pressure overburden when magma ocean solidifying (volatiles mostly in mantle)
                rs_term = 0.0
            else: # if at solidus, start increasing it according to mantle temperature cooling
                rs_term = rs_term_fun(float(y[2]),a1,b1,a2,b2,g,alpha,cp,pm,rp,y[7],0.0) #doesnt seem to affect things
            
            if (2>1):          
                if y[7]+TMoffset>SurfT:
                    if heating_switch == 1:
                        numerator =  (4./3.)* np.pi * pm * Qr * (rp**3.0-y[2]**3.0) + Qcore -  4.0*np.pi*((rp)**2)*qm 
                    elif heating_switch == 0 :
                        numerator =  (4./3.)* np.pi * pm * Qr * (rp**3.0-rc**3.0) + Qcore -  4*np.pi*((rp)**2)*qm 
                    else:
                        print ('ERROR')
                    denominator = (4./3.) * np.pi * pm * cp *(rp**3 - y[2]**3) - 4*np.pi*(y[2]**2)* delHf*pm * rs_term
                    dy7_dt = numerator/denominator #this is Tp
                else:
                    if heating_switch == 1:
                        numerator =  (4./3.)* np.pi * pm * Qr * (rp**3.0-y[2]**3.0) + Qcore + 4.0*np.pi*((rp)**2)*qc 
                    elif heating_switch == 0:
                        numerator =  (4./3.)* np.pi * pm * Qr * (rp**3.0-rc**3.0) + Qcore +  4*np.pi*((rp)**2)*qc
                    else:
                        print ('ERROR')
                    denominator = (4./3.) * np.pi * pm * cp *(rp**3 - y[2]**3) - 4*np.pi*(y[2]**2)* delHf*pm * rs_term
                    dy7_dt = numerator/denominator
                
        
            dy16_dt = 0.0
            [OLR,newpCO2,ocean_pH,ALK,Mass_oceans_crude,DIC_check] = correction(float(SurfT),float(Te_input),H2O_Pressure_surface,CO2_Pressure_surface,rp,g,ocean_CO3,1e5,float(y[24]),float(y[22]))
            CO2_Pressure_surface = newpCO2

            ddelta_dt = 0.0
            dVcrust_dt = 0.0
            dQmantle_dt = 0.0 #Qr # per unit volume
            dQcrust_dt = 0.0 #
            

            dK40mantle_dt =  - (lam_Ar+lam_Ca) * y[35] / (1e9*365*24*60*60)
            dAr40mantle_dt =  (lam_Ar/(lam_Ar+lam_Ca)) * (y[2]**3 - rc**3)/(rp**3 - rc**3) * (lam_Ar+lam_Ca) * y[35] / (1e9*365*24*60*60)
            dK40lid_dt = 0.0
            dAr40atmo_dt = (lam_Ar/(lam_Ar+lam_Ca)) * (rp**3 - y[2]**3)/(rp**3 - rc**3) * (lam_Ar+lam_Ca) * y[35] / (1e9*365*24*60*60)


            dU238_mantle_dt = -(np.log(2)/4.46e9) * y[43] / (365*24*60*60)
            dU235_mantle_dt = -(np.log(2)/7.04e8) * y[44] / (365*24*60*60)
            dTh_mantle_dt = - (np.log(2)/1.4e10) * y[45] / (365*24*60*60)

            dU238_lid_dt = 0.0
            dU235_lid_dt = 0.0
            dTh_lid_dt = 0.0
            
            dHe_mantle_dt =  (y[2]**3 - rc**3)/(rp**3 - rc**3) *  ( (4./238) * 8.0 * (np.log(2)/4.46e9) * y[43] + (4./235) * 7.0 * (np.log(2)/7.04e8) * y[44]+ (4./232) * 6.0 * (np.log(2)/1.4e10) * y[45]) /  (365*24*60*60)
            dHe_atmo_dt  =  (rp**3 - y[2]**3)/(rp**3 - rc**3) *  ( (4./238) * 8.0 * (np.log(2)/4.46e9) * y[43] + (4./235) * 7.0 * (np.log(2)/7.04e8) * y[44]+ (4./232) * 6.0 * (np.log(2)/1.4e10) * y[45]) /  (365*24*60*60) #- He_escape_loss

            ASR = ASR_input

            y[8] = SurfT
            heat_atm = OLR - ASR  
        
            if (2>1):
                if SurfT > y[7]+TMoffset:
                    true_balance = - heat_atm - qc
                else:
                    true_balance = - heat_atm + qm
        
            if print_switch == "y":        
                print ("phi",actual_phi_surf,"visc",visc)
                print ("time",t0/(365*24*60*60*1e6),"Ra",Ra)
                print (OLR,ASR)
                print ("Heat balance",true_balance)
                print (" ")
            solid_switch = 0.0
        
            if liquid_switch ==1 : ## If transitioning from liquid to solid, adjust inventories
                T_partition = np.max([y[7],Tsolidus+0.01])
                [XFeO,XFe2O3,fO2,F_FeO1_5,F_FeO] = solve_fO2_F_redo(y[4],H2O_Pressure_surface,T_partition,Total_Fe_mol_fraction,Mliq,y[2],rp,y[24])
                mu_O = 16.0
                mu_FeO_1_5 = 56.0 + 1.5*16.0
                if liquid_switch_worked==0: #first time or hasn't worked yet

                    y[3] = y[3] - Mliq * (y[3] / (4./3. * np.pi * pm * (y[2]**3 - rc**3)))
                    y[4] = y[4] + Mliq * (y[3] / (4./3. * np.pi * pm * (y[2]**3 - rc**3)))
                    y[5] = y[5] -  Mliq * (y[5] / (4./3. * np.pi * pm * (y[2]**3 - rc**3)))     
                    y[6] = y[6] - Mliq *  (y[6] / (4./3. * np.pi * pm * (y[2]**3 - rc**3)))   
                    y[0] = y[0] - Mliq * (y[0] / (4./3. * np.pi * pm * (y[2]**3 - rc**3)))
                    y[1] = y[1] + Mliq * (y[0] / (4./3. * np.pi * pm * (y[2]**3 - rc**3)))
                    y[42] = y[42] - DH_switch*Mliq * (y[42] / (4./3. * np.pi * pm * (y[2]**3 - rc**3)))
                    y[41] = y[41] + DH_switch*Mliq * (y[42] / (4./3. * np.pi * pm * (y[2]**3 - rc**3)))
                    y[13] = y[13] - Mliq * (y[13] / (4./3. * np.pi * pm * (y[2]**3 - rc**3)))
                    y[12] = y[12] + Mliq * (y[13] / (4./3. * np.pi * pm * (y[2]**3 - rc**3)))
                    y[26] = 0.0 
                    y[27] = 0.0

                    if y[29] > 0: # put everything back in the mantle. Everythin gets mixed up again during initial plate tectonics?
                        y[28] = y[28] + y[29] 
                        y[29] = 0.0
                    deltacrust = rp - (rp**3 - 3*y[27]/(4.0*np.pi))**(1./3.)
                    if rp - y[2] > deltacrust:
                        y[38] = y[38] + y[36] * ((rp-deltacrust)**3 - y[2]**3)/((rp-deltacrust)**3 - rc **3  )
                        y[36] = y[36] - y[36] * ((rp-deltacrust)**3 - y[2]**3)/((rp-deltacrust)**3 - rc **3  )  
                        y[35] = y[35] + y[37]
                        y[37] = 0.0
		        #dy35_dt =  dK40mantle_dt 
		        #dy36_dt =  dAr40mantle_dt 
   		        #dy37_dt =  dK40lid_dt
		        #dy38_dt =  dAr40atmo_dt  
                        y[50] = y[50] + y[49] * ((rp-deltacrust)**3 - y[2]**3)/((rp-deltacrust)**3 - rc **3  )
                        y[49] = y[49] - y[49] * ((rp-deltacrust)**3 - y[2]**3)/((rp-deltacrust)**3 - rc **3  )
                        y[43] = y[43] + y[46]
                        y[46] = 0.0
                        y[44] = y[44] + y[47]
                        y[47] = 0.0
                        y[45] = y[45] + y[48]
                        y[48] = 0.0
                    else:
                        y[38] = y[38]  ## no unmelted rock gets melted, no new Ar added atmosphere
                        y[35] = y[35]+ y[37] ## crust gets mixed back into mantle 
                        y[37] = 0.0
                        y[39] = 0.0

                        y[50] = y[50] 
                        y[43] = y[43] + y[46]
                        y[44] = y[44] + y[47]
                        y[45] = y[45] + y[48]
                        y[46] = 0.0
                        y[47] = 0.0
                        y[48] = 0.0
                      
                    switch_name = "switch_garbage/switch_IC_%d" %seed_save
                    load_name2 = switch_name+".npy"
                    if switch_counter == 0 :
                        np.save(switch_name,y)
                    else:
                        y = np.load(load_name2)
                    switch_counter = switch_counter + 1
                    return [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
                else: #switch has worked, back to normal, reset for next magma ocean transition
                    liquid_switch_worked = 0.0
                    liquid_switch = 0.0  
                    switch_counter = 0.0
                    return [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0] 



        #################################################################################
        #### If in solid mantle phase           
        else:
            if print_switch =="y":
                print('Mantle fully solid',t0/(365*24*60*60),y)
            beta = 1./3.
            ll = rp - rc #this is just mantle thickness 
 
            if solid_switch == 0:            
           
                Mliq = Mliq_fun(y[1],rp,y[2],pm) 
                Mcrystal = (1-phi_final)* Mliq 
                [FH2O,H2O_Pressure_surface] = H2O_partition_function( y[1],Mliq,Mcrystal,rp,g,kH2O,y[24])
                [FCO2,CO2_Pressure_surface] = CO2_partition_function( y[12],Mliq,Mcrystal,rp,g,kCO2,y[24]) 
                T_partition = np.max([y[7],Tsolidus+0.01]) 
                [XFeO,XFe2O3,fO2,F_FeO1_5,F_FeO] = solve_fO2_F_redo(y[4],H2O_Pressure_surface,T_partition,Total_Fe_mol_fraction,Mliq,y[2],rp,y[24])
                mu_O = 16.0
                mu_FeO_1_5 = 56.0 + 1.5*16.0

                if y[2] < rp:
                                
                    y[3] =  y[3] + (4./3.) *np.pi * pm *F_FeO1_5 * (rp**3 - y[2]**3) * 0.5*mu_O / (mu_FeO_1_5)
                    y[4] = y[4] - (4./3.) *np.pi * pm *F_FeO1_5 * (rp**3 - y[2]**3) * 0.5*mu_O / (mu_FeO_1_5)
                    y[5] = y[5] + (4./3.) *np.pi * pm * F_FeO1_5  *(rp**3 - y[2]**3)  ## mass solid FeO1_5 
                    y[6] = y[6] + (4./3.) *np.pi * pm *  F_FeO * (rp**3 - y[2]**3)  #mass solid F_FeO 
                    Water_transfer = np.min([Max_mantle_H2O-y[0],FH2O * kH2O * Mcrystal + FH2O * (Mliq-Mcrystal)])
                    CO2_transfer = np.min([Max_mantle_H2O-y[13],kCO2 * FCO2 * Mcrystal  + FCO2 * (Mliq-Mcrystal) ]) 
                    DHO_transfer = DH_switch*Water_transfer*y[41]/y[1]
                    y[42] = y[42] + DHO_transfer
                    y[41] = y[41] - DHO_transfer
                    y[0] = y[0] + Water_transfer
                    y[1] = y[1] - Water_transfer
                    y[13] = y[13] + CO2_transfer
                    y[12] = y[12] - CO2_transfer
                    y[26] = 0.0 
                    y[27] = 0.0
                    y[39] = 0.0
                    y[37] = pm*y[27]*(y[35]/Mantle_mass)
                    y[35] = y[35] - pm*y[27]*(y[35]/Mantle_mass)
		    #    dy35_dt =  dK40mantle_dt 
		    #    dy36_dt =  dAr40mantle_dt 
		    #    dy37_dt =  dK40lid_dt
		    #    dy38_dt =  dAr40atmo_dt 
                    y[46] = pm*y[27]*(y[43]/Mantle_mass) 
                    y[47] = pm*y[27]*(y[44]/Mantle_mass)  
                    y[48] = pm*y[27]*(y[45]/Mantle_mass)  
                    y[43] = y[43] -  pm*y[27]*(y[43]/Mantle_mass) 
                    y[44] = y[44] -  pm*y[27]*(y[44]/Mantle_mass) 
                    y[45] = y[45] -  pm*y[27]*(y[45]/Mantle_mass) 
                             
                    y[2] = rp
                    liquid_name = "switch_garbage/liquid_IC_%d" %seed_save
                    load_name = liquid_name + ".npy"
                    if solid_counter == 0:
                        np.save(liquid_name,y)
                    else:
                        y = np.load(load_name)
                    solid_counter = solid_counter + 1
                    return [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
                else:
                    solid_switch = 1.0  
                    solid_counter = 0.0
                    return [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]        

        
            y[2] = rp
            Mliq = 0.0 
            Mcrystal = 0.0 
            
            [FH2O,H2O_Pressure_surface] = H2O_partition_function( y[1],Mliq,Mcrystal,rp,g,kH2O,y[24]) 
            [FCO2,CO2_Pressure_surface] = CO2_partition_function( y[12],Mliq,Mcrystal,rp,g,kCO2,y[24]) 
            

            Tsolidus_unaltered = sol_liq(rp,g,pm,rp,float(H2O_Pressure_surface+CO2_Pressure_surface+1e5),float(0*y[0]/Mantle_mass0))
            Tsolidus_Pmod = Tsolidus_unaltered + depletion_fraction * 600 

            if PltTech == 1:
                deltardck = 0
            else:
                deltardck = y[26]   

            Tsolidus_visc = sol_liq(rp-deltardck,g,pm,rp,float(H2O_Pressure_surface+CO2_Pressure_surface+1e5),0.0)                           
            visc =  viscosity_fun(y[7],pm,visc_offset,y[8],float(Tsolidus_visc))
            initialize_fast = np.min([y[8],y[7]+TMoffset,y[7]+TMoffset-1])   

            AB = AB_fun(float(y[8]),H2O_Pressure_surface,float(y[1]+y[4]+y[12]),albedoC,albedoH)
            ASR_input = float((1-AB)*ASR_new_fun(t0))
            if ASR_input < min_ASR:
                ASR_input = min_ASR
            Te_ar = (ASR_input/5.67e-8)**0.25
            Te_input = Te_ar*(0.5**0.25)
            if Te_input > 350:
                Te_input = 350.0
            if Te_input < min_Te:
                Te_input = min_Te 

            initialize_fast = np.array(initialize_fast)
            ace1 =  nelder_mead(funTs_general_stag2, x0=initialize_fast, bounds=np.array([[10.0], [4000.0]]).T, args = (float(y[7]),ll,visc,beta,Te_input,ASR_input,H2O_Pressure_surface,CO2_Pressure_surface,ocean_CO3,float(y[24]),float(y[22]),float(y[25]),PltTech), tol_f=0.0001,tol_x=0.0001, max_iter=1000)
            SurfT = ace1.x[0]

            if abs(ace1.fun) > 1.0:#0.1:
                ace2 = optimize.minimize(funTs_general_stag,x0=initialize_fast,args = (float(y[7]),ll,visc,beta,Te_input,ASR_input,H2O_Pressure_surface,CO2_Pressure_surface,ocean_CO3,float(y[24]),float(y[22]),float(y[25]),PltTech),method='Nelder-Mead',bounds = ((10,4000),))
              
                if (abs(ace2.fun) < abs(ace1.fun))and(abs(ace2.fun)<1.0):
                    SurfT = ace2.x[0]
                else:
                    ace3= optimize.minimize(funTs_general_stag,x0=y[8],args = (float(y[7]),ll,visc,beta,Te_input,ASR_input,H2O_Pressure_surface,CO2_Pressure_surface,ocean_CO3,float(y[24]),float(y[22]),float(y[25]),PltTech),method='L-BFGS-B',bounds = ((600,2000),))
                    if (abs(ace3.fun) <  abs(ace1.fun))and(abs(ace3.fun)<1.0):
                        SurfT = ace3.x[0]
                    else:
                        ace4= optimize.minimize(funTs_general_stag,x0=y[8],args = (float(y[7]),ll,visc,beta,Te_input,ASR_input,H2O_Pressure_surface,CO2_Pressure_surface,ocean_CO3,float(y[24]),float(y[22]),float(y[25]),PltTech),method='L-BFGS-B',bounds = ((10,600),))
                        if (abs(ace4.fun) <  abs(ace1.fun))and(abs(ace4.fun)<1.0):
                            SurfT = ace4.x[0]


            if abs(ace1.fun) > 1.0:
                differ_ace = 10.0
                counter = 0

                initialize_fast = np.array(initialize_fast)
                budget_differ = 10.0
                aceCOMP =  nelder_mead(funTs_general_stag2, x0=initialize_fast, bounds=np.array([[10.0], [4000.0]]).T, args = (float(y[7]),ll,visc,beta,Te_input,ASR_input,H2O_Pressure_surface,CO2_Pressure_surface,ocean_CO3,float(y[24]),float(y[22]),float(y[25]),PltTech), tol_f=1e-10,tol_x=1e-10, max_iter=1000)

                if abs(aceCOMP.fun)<ace1.fun:
                    SurfT = aceCOMP.x[0]
                budget_differ = aceCOMP.fun
                budget_counter = 0
                while abs(budget_differ) > 0.1:
                    budget_counter = budget_counter + 1
                    ran_num = np.min([150 + 2000*np.random.uniform(),3999])
                    ran_numar=np.array(ran_num)
                    aceCOMP2 =  nelder_mead(funTs_general_stag2, x0=ran_numar, bounds=np.array([[10.0], [4000.0]]).T, args = (float(y[7]),ll,visc,beta,Te_input,ASR_input,H2O_Pressure_surface,CO2_Pressure_surface,ocean_CO3,float(y[24]),float(y[22]),float(y[25]),PltTech), tol_f=1e-10,tol_x=1e-10, max_iter=2000)
                    if abs(aceCOMP2.fun)<abs(budget_differ):
                        SurfT = aceCOMP2.x[0]     
                        budget_differ = aceCOMP2.fun

                    if budget_counter >= 10:
                        budget_differ = -2e-3 

            else:
                SurfT = ace1.x[0] 


            T_diff = y[7]+TMoffset-SurfT 
            Qr = Qr_cofactor*qr(t0,Start_time,heatscale,K_over_U)
            Qcore = Qr_cofactor*np.exp(-(t0/(1e9*365*24*60*60)-4.5)/7.0)*core_flow_coefficient
            
            if PltTech ==1:
                Vmantle = Vmantle0
                Ra = np.max([-1e15,alpha * g * T_diff * ll**3 / (kappa * visc)])
                qm = (k/ll) * T_diff * (Ra/Racr)**beta
                volc_term = 0.0
                
                dmelt_eq = y[16] 
                dcrust = y[26]
                deltac = dmelt_eq
                Vcrust_eq = (rp**3 - (rp-dmelt_eq)**3)*4*np.pi/3.0
                dVcrust_dt = 0.0 
                y[27] =  Vcrust_eq
                ddelta_dt = 0.0
                y[26] = y[16]
                dQmantle_dt = 0.0 #Qr # per unit volume
                dQcrust_dt = 0.0 #

            else:
                Vmantle = Vmantle0 - y[27]
                delta = y[26]
                Rai =alpha * g * T_diff* ll**3 / (kappa * visc)
                theta = 300000*T_diff/(8.314*float(y[7])**2)
                qm = 0.5 * k/ll * T_diff * theta**(-4.0/3.0) * Rai**(1./3.)           
                deltaTm = y[7] - SurfT ## adiabatic loss if stagnant lid

                Vcrust = y[27]
                deltac = rp - (rp**3 - 3*Vcrust/(4.0*np.pi))**(1./3.)
                T1 = float(y[7])  - 2.5 * 8.314*float(y[7])**2/300000.0
                fm = float(y[25])

                term1 = (SurfT * (delta - deltac) + T1 * deltac)/delta

                xm =  Qr * pm * y[28]*Vmantle0/Vmantle # W/kg rock * kg rock/m3 * Vmantle0/Vmantle
                xc = Qr * pm * y[29]*Vmantle0/Vcrust 
                term2 = (xc*deltac**2 * (delta-deltac) + xm * deltac * (delta - deltac)**2 )/ (2*k*delta)
                Tc = term1 + term2
                if (abs(Tc-T1)>1):
                    kdtdz_at_Rp_minus_delta = - k*(T1 - Tc)/(delta - deltac) + xm*0.5*(delta - deltac)
                else:
                    kdtdz_at_Rp_minus_delta = - k*(T1 - SurfT)/deltac + xc*0.5 * deltac

                ddelta_dt = (-qm - kdtdz_at_Rp_minus_delta) / (pm*cp*(y[7] - T1))
                dVcrust_dt = fm - (fm - 4*np.pi*(rp - delta)**2 * np.min([0.0,ddelta_dt])) * (np.tanh((deltac -delta)*20) + 1)

                Ra = Rai
                volc_term = float(y[25])*3000*(cp*deltaTm + delHf)
           
            y[32] = qm * 4.0 * np.pi*rp**2
            y[33] = volc_term
         
            [OLR,newpCO2,ocean_pH,ALK,Mass_oceans_crude,DIC_check] = correction(float(SurfT),float(Te_input),H2O_Pressure_surface,CO2_Pressure_surface,rp,g,ocean_CO3,1e5,float(y[24]),float(y[22]))
            CO2_Pressure_surface = newpCO2 
            
            ASR = ASR_input
            heat_atm = OLR - ASR
            delta_u = k * (y[7]+TMoffset - SurfT) / qm
    
            numerator = - 4*np.pi*(rp**2)*qm  + (Qr * pm * y[28]*Vmantle0 ) + Qcore - volc_term 
            denominator = pm * cp * Vmantle
                      
            thermal_cond = 2 # W/m2/K
            qc = thermal_cond * (SurfT - y[7]-TMoffset) / (rp-rc)
            if SurfT > y[7]+TMoffset:
                numerator =  4*np.pi*(rp**2)*qc  +  (Qr * pm * y[28]*Vmantle0 ) + Qcore
                denominator = pm * cp * Vmantle 
                y[32] = -qc * 4.0 * np.pi*rp**2
                y[33] = 0
                ddelta_dt = 0.0
                dVcrust_dt = 0.0
    
            rs_term = 0.0  
            dy7_dt = numerator/denominator
 
            [dK40mantle_dt,dAr40mantle_dt,dK40lid_dt,dAr40atmo_dt] =[ 0.0,0.0,0.0,0.0]
            [dU238_mantle_dt,dU235_mantle_dt,dTh_mantle_dt,dU238_lid_dt,dU235_lid_dt,dTh_lid_dt,dHe_mantle_dt,dHe_atmo_dt] = [0,0,0,0,0,0,0,0]
               
            y[8] = SurfT
            liquid_switch = 1.0
            liquid_switch_worked = 0.0

        ####################################################################################  
        ## end magma ocean/solid mantle portion. The rest of the code applies to both phases
        ####################################################################################  

        
        toc1 = time.time()

        y[30] =  (Qr * pm * y[28]*Vmantle0) 
        y[31] = (Qr * pm * y[29]*Vmantle0 )

        y[9] = OLR
        y[10] = ASR
        if y[8]<=y[7]+TMoffset:
            y[11] = qm
        else:
            y[11] = -qc

        if (2>1): #same thing seems to work
            dy2_dt = rs_term * dy7_dt
            drs_dt = dy2_dt
            rs = np.min([y[2],rp])     

        [FH2O,H2O_Pressure_surface] = H2O_partition_function( y[1],Mliq,Mcrystal,rp,g,kH2O,y[24])
        if y[1]< 0:
            H2O_Pressure_surface = 0.0
            FH2O = 0.0

        water_frac = my_water_frac(float(y[8]),Te_input,H2O_Pressure_surface,CO2_Pressure_surface)
        
        atmo_H2O = np.max([H2O_Pressure_surface*water_frac,0.0])
        
        if solid_switch == 0:
            T_partition = np.max([y[7],Tsolidus+0.01])
            [XFeO,XFe2O3,fO2,F_FeO1_5,F_FeO] = solve_fO2_F_redo(y[4],H2O_Pressure_surface,T_partition,Total_Fe_mol_fraction,Mliq,rs,rp,y[24]) #
        else:
            [XFeO,XFe2O3,fO2,F_FeO1_5,F_FeO] = solve_fO2_F_redo(y[4],H2O_Pressure_surface,y[8],Total_Fe_mol_fraction,Mliq,rs,rp,y[24]) #

        fO2_pos = np.max([0,fO2])
        Pressure_surface = fO2_pos +atmo_H2O + CO2_Pressure_surface + 1e5 
        
    
        #####################################
        ### OPTIONAL CO2 dep Tstrat (AND [51] below)
        #CO2_upper_atmosphere = y[51]
        #Te_input_escape = 214 - 44*CO2_upper_atmosphere
        ##############################

        frac_h2o_upper = my_fH2O(float(SurfT),Te_input_escape,H2O_Pressure_surface,CO2_Pressure_surface)
        frac_h2o_upper = np.min([atmo_H2O / Pressure_surface, frac_h2o_upper]) #fudged this for 5x less escape

        if (H2O_Pressure_surface<1e-5)and(frac_h2o_upper<1e-9): # check to see if it matters
            frac_h2o_upper = 0.0
            atmo_H2O = 0.0
            H2O_Pressure_surface = 0.0
        
        #######################
        ## Atmosphsic escape calculations
        ## diffusion limited escape:
        fCO2_p = (1- frac_h2o_upper)*CO2_Pressure_surface / (CO2_Pressure_surface+fO2_pos+1e5)
        y[51] = fCO2_p
        fO2_p = (1- frac_h2o_upper)*fO2_pos / (CO2_Pressure_surface+fO2_pos+1e5)
        fN2_p = (1- frac_h2o_upper)*1e5 / (CO2_Pressure_surface+fO2_pos+1e5)
        mol_diff_H2O_flux = better_atomic_diffusion(frac_h2o_upper,Te_input_escape,g,fCO2_p,fO2_p,fN2_p) #mol H2O/m2/s I think

        #XUV-driven escape
        XH_upper = 2*frac_h2o_upper / (3*frac_h2o_upper + 2*fO2_p + fCO2_p + fN2_p) ## assumes CO2 and N2 don't dissociate, hmm
        ## now calculate atomic ratios and do XUV limited           
        XH = 2*frac_h2o_upper / (3*frac_h2o_upper + 2*fO2_p + fCO2_p + fN2_p)
        XO = (2*fO2_p+frac_h2o_upper) / (3*frac_h2o_upper + 2*fO2_p + fCO2_p + fN2_p)
        XC = fCO2_p / (3*frac_h2o_upper + 2*fO2_p + fCO2_p + fN2_p)

        true_epsilon = find_epsilon(Te_input,RE,ME,float(AbsXUV(t0)), XO, XH, XC,epsilon,mix_epsilon)
        #true_epsilon = find_epsilon(1000.0,RE,ME,float(AbsXUV(t0)), XO, XH, XC,epsilon,mix_epsilon)
        if (XC < 0)or(y[12]<1e10):
            XC = 0.0

        [mH_Odert3,mO_Odert3,mOdert_build3,mC_Odert3] = Odert_three(Te_input_escape,RE,ME,true_epsilon,float(AbsXUV(t0)), XO, XH, XC) #kg/m2 /s
        #[mH_Odert3,mO_Odert3,mOdert_build3,mC_Odert3] = Odert_three(1000.0,RE,ME,true_epsilon,float(AbsXUV(t0)), XO, XH, XC) #kg/m2 /s

        numH = ( mH_Odert3 / 0.001 ) # mol H / m2/ s
        numO = ( mO_Odert3 / 0.016 ) # mol O / m2/ s
        #numC = ( mC_Odert3 / 0.012 ) # mol C / m2/ s
        numC = ( mC_Odert3 / 0.044 ) # mol CO2 / m2/ s
        
        if 2*mol_diff_H2O_flux> numH: ## if diffusion limit exceeds XUV-driven, shift diffusion-limit downward
             mol_diff_H2O_flux = 0.5*np.copy(numH)

        ## Combined escape flux, weighted by H abundance:        
        w1 = mult*(2./3 -  XH_upper)**4
        w2 = XH_upper**4
        Mixer_H = (w1*2*mol_diff_H2O_flux + w2 * numH ) / (w1+w2) # mol H / m2 /s
        Mixer_O = (w1*0.0 + w2 * numO ) / (w1+w2)
        Mixer_C = (w1*0.0 + w2 * numC ) / (w1+w2)
        #Mixer_Build = 0.5* Mixer_H - Mixer_O + 2*Mixer_C  # mol O / m2 /s C atmo drag affects redox
        Mixer_Build = 0.5* Mixer_H - Mixer_O  ## CO2 drag doesn't affect redox

        ## nominal
        escape = 4*np.pi*rp**2 * Mixer_H*0.018/2 ## kg H2O /s
        net_escape = 4*np.pi*rp**2 * Mixer_Build*0.016 ## kg O2/s
        CO2_loss =  4*np.pi*rp**2 * Mixer_C*0.044 ## kg CO2/s

        ## ion escape test
        ### 2.0243e-14 * np.exp(8.05905*tgyr)        # bar/ gyr
        ### 2.0243e-14 * np.exp(8.05905*tgyr) * (1e5*4*np.pi*rp**2)/(g*365*24*60*60*1e9) # kg/s
        #tgyr = 4.6 - t0/(365*24*60*60*1e9) 
        #if tgyr >3.3:
        #    dOdt = 2.0243e-14 * np.exp(8.05905*tgyr) * (1e5*4*np.pi*rp**2)/(g*365*24*60*60*1e9) # kg/s
        #else:
        #    dOdt = 2.0243e-14 * np.exp(8.05905*3.3) * (1e5*4*np.pi*rp**2)/(g*365*24*60*60*1e9)        
        #escape = 4*np.pi*rp**2 * Mixer_H*0.018/2
        #net_escape = 4*np.pi*rp**2 * Mixer_Build*0.016 - dOdt
        #CO2_loss =  4*np.pi*rp**2 * Mixer_C*0.044 ## kg CO2/s
        ## end ion escape test
 
        if y[50]>0:
            He_mixing_ratio = (y[50]/0.004) / ((Pressure_surface*(4*np.pi*rp**2)/g)/y[24])
        else:
            He_mixing_ratio = 0.0
        He_escape_loss = 9.6e5/(365*24*60*60) *  (AbsXUV(t0)/AbsXUV(4.5*1e9*365*24*60*60)) * (He_mixing_ratio/10e-6)
        # done with escape calculations
        #######################
 
        ## Find ocean depth and land fraction:      
        Ocean_depth = (0.018/y[24]) * (1-water_frac) * H2O_Pressure_surface / (g*1000) ## max ocean depth continents 11.4 * gEarth/gplanet (COwan and Abbot 2014)
        Max_depth = 11400 * (9.8 / g) 
        if Ocean_depth > Max_depth:
            Linear_Ocean_fraction = 1.0
            Ocean_fraction = 1.0
        else:
            Linear_Ocean_fraction = (Ocean_depth/Max_depth) ## Approximation to Earth hypsometric curve
            Ocean_fraction = (Ocean_depth/Max_depth)**0.25

        ## Melt and crustal oxidation variables:    
        actual_phi_surf_melt = 0.0
        Va = 0.0
        F_CO2 = 0.0
        F_H2O = 0.0
        O2_consumption = 0.0
        OG_O2_consumption =0.0
        Abs_crustal_prod = 0.0
        crustal_depth = rp - (rp**3 - y[27]*3./(4.*np.pi))**(1./3.)
        Poverburd = fO2_pos +H2O_Pressure_surface + CO2_Pressure_surface + 1e5 

        if PltTech == 1:
            deltardck = 0
        else:
            deltardck = y[26]  

        Tsolidus_unaltered = sol_liq(rp-deltardck,g,pm,rp,float(Poverburd),float(0*y[0]/Mantle_mass0)) 
        Tsolidus_Pmod = Tsolidus_unaltered + depletion_fraction * 600       
        
        y[34] = 0.0
        dQmantle_dt = 0.0 
        dQcrust_dt = 0.0 #
        ddt_depletion_fraction = 0.0
        ## If solid mantle, calculate crutal production and melt volume: 
        if (y[2]>=rp)and(Ra>0)and(y[8]<=Tsolidus):
            T_for_melting = float(y[7])
            if T_for_melting <= Tsolidus_Pmod:  # no melting if potential temperature is below solidus
                [actual_phi_surf_melt, Va] = [0.0, 0.0]
                Melt_volume = 0.0
                Abs_crustal_prod = 0.0
            else: #mantle temperature above solidus, calculate melt production
                mantleH2Ofrac = float(0*y[0]/Mantle_mass)
                rdck = optimize.minimize(find_r2,x0=float(y[2]),args = (T_for_melting,alpha,g,cp,pm,rp,float(Poverburd),mantleH2Ofrac,deltardck,depletion_fraction))
                rad_check = float(rdck.x[0])  # find radius where partial melting begins
                y[34] = rp - rad_check
                if rad_check>rp:
                    rad_check = rp
                if PltTech == 1:
                    rlid = rp
                    #calculate melt production
                    [actual_phi_surf_melt,actual_visc,Va] = temp_meltfrac2(0.99998*rad_check,rp,alpha,pm,T_for_melting,cp,g,Poverburd,mantleH2Ofrac,rlid,depletion_fraction)
                    crustal_depth = rp - (rp**3 - actual_phi_surf_melt * Va*3./(4.*np.pi))**(1./3.) #calulate crustal depth (depth of melt)
                    Ra_melt =alpha * g * (y[7]+TMoffset - SurfT) * ll**3 / (kappa * visc) 
                    Q =  (k/ll) * (y[7]+TMoffset - SurfT) * (Ra_melt/Racr)**beta
                    Aoc = 4*np.pi*rp**2
                    Melt_volume = (Q*4*np.pi*rp**2)**2/(2*k*(y[7]+TMoffset - SurfT))**2 * (np.pi*kappa)/(Aoc)  * crustal_depth 
                    Abs_crustal_prod = Melt_volume
                    ddt_depletion_fraction = 0
                elif (PltTech <1)and(0.99998*rad_check < rp-y[26]):
                    rlid = rp - y[26]
                    [actual_phi_surf_melt,actual_visc,Va] = temp_meltfrac2(0.99998*rad_check,rp,alpha,pm,T_for_melting,cp,g,Poverburd,mantleH2Ofrac,rlid,depletion_fraction ) 
                    crustal_depth = rp - (rp**3 - y[27]*3./(4.*np.pi))**(1./3.)
                    dm = rp - 0.99998*rad_check #where melting begins
                    velocity = 0.05 * kappa /(rp - rc) * (Rai/theta)**(2.0/3.0)
                    Melt_volume = 17.8 * np.pi * rp * velocity * (dm - y[26]) * actual_phi_surf_melt
                    ddt_depletion_fraction =  0.0 #Melt_volume*(1 - actual_phi_surf_melt) * ( 1 - depletion_fraction) / Vmantle # Uncomment for mantle depletion test
                    Abs_crustal_prod = 0.1*Melt_volume 
                elif (PltTech <1) and (0.99998*rad_check >= rp-y[26]):
                    [actual_phi_surf_melt,actual_visc,Va]  = [0,0,0]
                    Melt_volume = 0.0
                    Abs_crustal_prod = 0.0
                    crustal_depth = rp - (rp**3 - y[27]*3./(4.*np.pi))**(1./3.)
                    ddt_depletion_fraction =  0.0

            Dradio = 0.002
            D_K_potassium = 1.0
            if PltTech == 1:
                dQmantle_dt = 0.0 #Qr # per unit volume
                dQcrust_dt = 0.0 #

                dK40mantle_dt =  - (lam_Ar+lam_Ca) * y[35] / (1e9*365*24*60*60) #- y[25]*pm * (y[35]/Mantle_mass)*(1 - (1-actual_phi_surf_melt)**(1/Dradio))/actual_phi_surf_melt
                dAr40mantle_dt =  (lam_Ar/(lam_Ar+lam_Ca)) * (lam_Ar+lam_Ca) * y[35] / (1e9*365*24*60*60) - pm * y[25]*y[36]/Mantle_mass
                dK40lid_dt = 0.0
                y[37] = pm*y[27]*(y[35]/Mantle_mass)
                dAr40atmo_dt = pm * y[25]*y[36]/Mantle_mass  # +(lam_Ar+lam_Ca) * y[37] / (1e9*365*24*60*60) 

                #if (y[8]<=Tsolidus)and(y[27]>0):
                #   dQcrust_dt =      dVcrust_dt*(y[29]/y[27])
                #   dMantle_dt = - dVcrust_dt*(y[29]/y[27])
                #   dK40lid_dt = dK40lid_dt + dVcrust_dt*(y[37]/y[27])
                #   dK40mantle_dt = dK40mantle_dt - dVcrust_dt*(y[37]/y[27])
                #   print (dQcrust_dt,dMantle_dt)

                dU238_mantle_dt = -(np.log(2)/4.46e9) * y[43] / (365*24*60*60)
                dU235_mantle_dt = -(np.log(2)/7.04e8) * y[44] / (365*24*60*60)
                dTh_mantle_dt = - (np.log(2)/1.4e10) * y[45] / (365*24*60*60)

                dU238_lid_dt = 0.0
                dU235_lid_dt = 0.0
                dTh_lid_dt = 0.0
                y[46] = pm*y[27]*(y[43]/Mantle_mass)
                y[47] = pm*y[27]*(y[44]/Mantle_mass)
                y[48] = pm*y[27]*(y[45]/Mantle_mass)
                 
                dHe_mantle_dt =   ( (4./238) * 8.0 * (np.log(2)/4.46e9) * y[43] + (4./235) * 7.0 * (np.log(2)/7.04e8) * y[44]+ (4./232) * 6.0 * (np.log(2)/1.4e10) * y[45]) /  (365*24*60*60) - pm * y[25]*y[49]/Mantle_mass
                dHe_atmo_dt  =  pm * y[25]*y[49]/Mantle_mass #- He_escape_loss


            else:
                
                if actual_phi_surf_melt > 0:
                    Vmantle_plus_Vlid = (4.0*np.pi/3.0) * (rp**3 - rc**3) - y[27]
                    dQcrust_dt = (y[28]/Vmantle_plus_Vlid)*y[25]/actual_phi_surf_melt *(1 - (1-actual_phi_surf_melt)**(1/Dradio)) - (y[29]/y[27]) * (y[25] - 4*np.pi*(rp - y[26])**2 *np.min([0,ddelta_dt]))* (np.tanh((deltac - y[26])*20) + 1) 
                    dQmantle_dt = - (y[28]/Vmantle_plus_Vlid)*y[25]/actual_phi_surf_melt *(1 - (1-actual_phi_surf_melt)**(1/Dradio)) + (y[29]/y[27]) * (y[25] - 4*np.pi*(rp - y[26])**2 *np.min([0,ddelta_dt]))* (np.tanh((deltac - y[26])*20) + 1) 


                    dK40mantle_dt =  - (lam_Ar+lam_Ca) * y[35] / (1e9*365*24*60*60) - pm*(y[35]/Mantle_mass)*y[25]/actual_phi_surf_melt *(1 - (1-actual_phi_surf_melt)**(1/D_K_potassium)) + pm*(y[37]/(pm*y[27])) * (y[25] - 4*np.pi*(rp - y[26])**2 *np.min([0,ddelta_dt]))* (np.tanh((deltac - y[26])*20) + 1) 
                    dAr40mantle_dt =  (lam_Ar/(lam_Ar+lam_Ca)) * (lam_Ar+lam_Ca) * y[35] / (1e9*365*24*60*60) -  pm * y[25]*y[36]/Mantle_mass
                    dK40lid_dt =  + pm*(y[35]/Mantle_mass)*y[25]/actual_phi_surf_melt *(1 - (1-actual_phi_surf_melt)**(1/D_K_potassium)) - pm*(y[37]/(pm*y[27])) * (y[25] - 4*np.pi*(rp - y[26])**2 *np.min([0,ddelta_dt]))* (np.tanh((deltac - y[26])*20) + 1)  - (lam_Ar+lam_Ca) * y[37] / (1e9*365*24*60*60) 
                    dAr40atmo_dt =  pm * y[25]*y[36]/Mantle_mass  +  (lam_Ar/(lam_Ar+lam_Ca)) * (lam_Ar+lam_Ca) * y[37] / (1e9*365*24*60*60) 


                    dU238_mantle_dt = - (np.log(2)/4.46e9) * y[43] / (365*24*60*60) - pm*(y[43]/Mantle_mass)*y[25]/actual_phi_surf_melt *(1 - (1-actual_phi_surf_melt)**(1/Dradio)) + pm*(y[46]/(pm*y[27])) * (y[25] - 4*np.pi*(rp - y[26])**2 *np.min([0,ddelta_dt]))* (np.tanh((deltac - y[26])*20) + 1) 
                    dU235_mantle_dt = -(np.log(2)/7.04e8) * y[44] / (365*24*60*60)  - pm*(y[44]/Mantle_mass)*y[25]/actual_phi_surf_melt *(1 - (1-actual_phi_surf_melt)**(1/Dradio)) + pm*(y[47]/(pm*y[27])) * (y[25] - 4*np.pi*(rp - y[26])**2 *np.min([0,ddelta_dt]))* (np.tanh((deltac - y[26])*20) + 1) 
                    dTh_mantle_dt = - (np.log(2)/1.4e10) * y[45] / (365*24*60*60)  - pm*(y[45]/Mantle_mass)*y[25]/actual_phi_surf_melt *(1 - (1-actual_phi_surf_melt)**(1/Dradio)) + pm*(y[48]/(pm*y[27])) * (y[25] - 4*np.pi*(rp - y[26])**2 *np.min([0,ddelta_dt]))* (np.tanh((deltac - y[26])*20) + 1) 

                    dU238_lid_dt = pm*(y[43]/Mantle_mass)*y[25]/actual_phi_surf_melt *(1 - (1-actual_phi_surf_melt)**(1/Dradio)) - pm*(y[46]/(pm*y[27])) * (y[25] - 4*np.pi*(rp - y[26])**2 *np.min([0,ddelta_dt]))* (np.tanh((deltac - y[26])*20) + 1)  - (np.log(2)/4.46e9) * y[46] / (365*24*60*60) 
                    dU235_lid_dt = pm*(y[44]/Mantle_mass)*y[25]/actual_phi_surf_melt *(1 - (1-actual_phi_surf_melt)**(1/Dradio)) - pm*(y[47]/(pm*y[27])) * (y[25] - 4*np.pi*(rp - y[26])**2 *np.min([0,ddelta_dt]))* (np.tanh((deltac - y[26])*20) + 1) -  (np.log(2)/7.04e8) * y[47] / (365*24*60*60)  
                    dTh_lid_dt =  pm*(y[45]/Mantle_mass)*y[25]/actual_phi_surf_melt *(1 - (1-actual_phi_surf_melt)**(1/Dradio)) - pm*(y[48]/(pm*y[27])) * (y[25] - 4*np.pi*(rp - y[26])**2 *np.min([0,ddelta_dt]))* (np.tanh((deltac - y[26])*20) + 1) - (np.log(2)/1.4e10) * y[48] / (365*24*60*60)  

                 
                    dHe_mantle_dt =   ( (4./238) * 8.0 * (np.log(2)/4.46e9) * y[43] + (4./235) * 7.0 * (np.log(2)/7.04e8) * y[44]+ (4./232) * 6.0 * (np.log(2)/1.4e10) * y[45]) /  (365*24*60*60) - pm * y[25]*y[49]/Mantle_mass
                    dHe_atmo_dt  =  pm * y[25]*y[49]/Mantle_mass + ( (4./238) * 8.0 * (np.log(2)/4.46e9) * y[46] + (4./235) * 7.0 * (np.log(2)/7.04e8) * y[47]+ (4./232) * 6.0 * (np.log(2)/1.4e10) * y[48]) /  (365*24*60*60) # - He_escape_loss



                else:
                    dQcrust_dt =  - (y[29]/y[27]) * (y[25] - 4*np.pi*(rp - y[26])**2 *np.min([0,ddelta_dt]))* (np.tanh((deltac - y[26])*20) + 1)  
                    dQmantle_dt =   (y[29]/y[27]) * (y[25] - 4*np.pi*(rp - y[26])**2 *np.min([0,ddelta_dt]))* (np.tanh((deltac - y[26])*20) + 1) 

                    dK40mantle_dt =  - (lam_Ar+lam_Ca) * y[35] / (1e9*365*24*60*60) + pm*(y[37]/(pm*y[27])) * (y[25] - 4*np.pi*(rp - y[26])**2 *np.min([0,ddelta_dt]))* (np.tanh((deltac - y[26])*20) + 1) 
                    dAr40mantle_dt =  (lam_Ar/(lam_Ar+lam_Ca)) * (lam_Ar+lam_Ca) * y[35] / (1e9*365*24*60*60) 
                    dK40lid_dt =  - pm*(y[37]/(pm*y[27])) * (y[25] - 4*np.pi*(rp - y[26])**2 *np.min([0,ddelta_dt]))* (np.tanh((deltac - y[26])*20) + 1)  -  (lam_Ar+lam_Ca) * y[37] / (1e9*365*24*60*60) 
                    dAr40atmo_dt =   (lam_Ar/(lam_Ar+lam_Ca)) * (lam_Ar+lam_Ca) * y[37] / (1e9*365*24*60*60) 
   


                    dU238_mantle_dt = - (np.log(2)/4.46e9) * y[43] / (365*24*60*60) + pm*(y[46]/(pm*y[27])) * (y[25] - 4*np.pi*(rp - y[26])**2 *np.min([0,ddelta_dt]))* (np.tanh((deltac - y[26])*20) + 1) 
                    dU235_mantle_dt = -(np.log(2)/7.04e8) * y[44] / (365*24*60*60)  + pm*(y[47]/(pm*y[27])) * (y[25] - 4*np.pi*(rp - y[26])**2 *np.min([0,ddelta_dt]))* (np.tanh((deltac - y[26])*20) + 1) 
                    dTh_mantle_dt = - (np.log(2)/1.4e10) * y[45] / (365*24*60*60) + pm*(y[48]/(pm*y[27])) * (y[25] - 4*np.pi*(rp - y[26])**2 *np.min([0,ddelta_dt]))* (np.tanh((deltac - y[26])*20) + 1) 

                    dU238_lid_dt =  - pm*(y[46]/(pm*y[27])) * (y[25] - 4*np.pi*(rp - y[26])**2 *np.min([0,ddelta_dt]))* (np.tanh((deltac - y[26])*20) + 1)  - (np.log(2)/4.46e9) * y[46] / (365*24*60*60) 
                    dU235_lid_dt =  - pm*(y[47]/(pm*y[27])) * (y[25] - 4*np.pi*(rp - y[26])**2 *np.min([0,ddelta_dt]))* (np.tanh((deltac - y[26])*20) + 1) -  (np.log(2)/7.04e8) * y[47] / (365*24*60*60)  
                    dTh_lid_dt =  - pm*(y[48]/(pm*y[27])) * (y[25] - 4*np.pi*(rp - y[26])**2 *np.min([0,ddelta_dt]))* (np.tanh((deltac - y[26])*20) + 1) - (np.log(2)/1.4e10) * y[48] / (365*24*60*60)  

                    dHe_mantle_dt =   ( (4./238) * 8.0 * (np.log(2)/4.46e9) * y[43] + (4./235) * 7.0 * (np.log(2)/7.04e8) * y[44]+ (4./232) * 6.0 * (np.log(2)/1.4e10) * y[45]) /  (365*24*60*60) 
                    dHe_atmo_dt  =    ( (4./238) * 8.0 * (np.log(2)/4.46e9) * y[46] + (4./235) * 7.0 * (np.log(2)/7.04e8) * y[47]+ (4./232) * 6.0 * (np.log(2)/1.4e10) * y[48]) /  (365*24*60*60) # -He_escape_loss
            
            ## Fresh crust production
            dmelt_dt = pm * Melt_volume #kg/s over whole planet
            iron3 = y[5]*56/(56.0+1.5*16.0) #kg fe203 * 56 g fe/mol / x g fe2o3 /mol fe2o3
            iron2 = y[6]*56/(56.0+16.0)
            iron_ratio_mantle = 0.5*iron3/iron2

            ## Given melt production and land fraction, calculate outgassing from seafloor and continents
            if (Ocean_fraction<1.0) and (T_for_melting > Tsolidus_Pmod):  # Land outgassing
                [F_H2O_L,F_CO2_L,F_H2_L,F_CO_L,F_CH4_L,OG_O2_consumption_L] = outgas_flux_cal_fast(Tsolidus,Pressure_surface,iron_ratio_mantle,Mantle_mass,y[13],y[0],dmelt_dt*1000.0,Total_Fe_mol_fraction,actual_phi_surf_melt)
            else: 
                [F_H2O_L,F_CO2_L,F_H2_L,F_CO_L,F_CH4_L,OG_O2_consumption_L] = [0.0,0.0,0.0,0.0,0.0,0.0]   
           
            if (T_for_melting > Tsolidus_Pmod): #Seafloor outgassing
                [F_H2O_O,F_CO2_O,F_H2_O,F_CO_O,F_CH4_O,OG_O2_consumption_O] = outgas_flux_cal_fast(Tsolidus,Poverburd,iron_ratio_mantle,Mantle_mass,y[13],y[0],dmelt_dt*1000.0,Total_Fe_mol_fraction,actual_phi_surf_melt)
            else: 
                [F_H2O_O,F_CO2_O,F_H2_O,F_CO_O,F_CH4_O,OG_O2_consumption_O] = [0.0,0.0,0.0,0.0,0.0,0.0]

            # Total outgassing weighted by land fraction
            [F_H2O,F_CO2,F_H2,F_CO,F_CH4,OG_O2_consumption] = np.array([F_H2O_O,F_CO2_O,F_H2_O,F_CO_O,F_CH4_O,OG_O2_consumption_O])*Ocean_fraction+np.array([F_H2O_L,F_CO2_L,F_H2_L,F_CO_L,F_CH4_L,OG_O2_consumption_L])*(1-Ocean_fraction)
            #F_CO2 = F_CO2 + F_CO + F_CH4 #photochemical oxidation of all C to CO2
            O2_consumption = np.copy(OG_O2_consumption)
        else:
            dmelt_dt = 0.0
            Melt_volume = 0.0

                  
        if rs<rp:
            fudge = 1.0-(rs/rp)**1000.0 ## helps with numerical issues
        else:
            fudge = 0.0

        y[17] = Mcrystal

        mu_O = 16.0
        mu_FeO_1_5 = 56.0 + 1.5*16.0 
        mu_FeO = 56.0 + 16.0   
        # Calculate max surface emplacement available for oxidation
        # 1e7 ~ 80 km3/yr emplacement, 1e8 ~ 800 km3/yr emplacement = 1mm/yr oxidized over whole surface
        surface_emplacement =np.min([dmelt_dt,surface_magma_fr*1e9/dry_oxid_frac]) 
        

        if y[8]< 973: #If surface temperature below serpentine stability, calculate crustal hydration
            if PltTech == 1:
                hydration_depth = np.max([0.0,delta_u * (973 - y[8])/(y[7] - y[8])])
                frac_hydrat = np.min([hydration_depth/crustal_depth,1.0])
                water_c = Linear_Ocean_fraction*MFrac_hydrated *frac_hydrat * dmelt_dt 
                water_crust = water_c * np.max([0.0,1 - y[0]/Max_mantle_H2O])
                wet_oxidation = wet_oxid_eff *(water_c/MFrac_hydrated) * Total_Fe_mass_fraction * (y[6]/(y[6]+y[5])) # kg FeO/s of the hydrated crust, what fraction iron oxidized
                # so this means, of the crust that is hydrated, how much iron oxidized
                ##additionally, restrict water to be less than surface inventory
                total_water_loss_surf = water_crust + wet_oxidation*18.0/(3.0* mu_FeO) #kg H2O/s
                total_water_gain_interior = water_crust
                new_wet_oxidation = wet_oxidation * 16.0 / (3.0* mu_FeO)  #kg O/s added solid
                total_wet_FeO_lost = wet_oxidation * (2.0/3.0) # 
                total_wet_FeO1_5 = wet_oxidation * (2.0/3.0) * mu_FeO_1_5/mu_FeO #kg FeO1_5 /s  
                total_wet_H2_gained = wet_oxidation * 2.0/(3.0* mu_FeO) #kg H2/s
                dH2O_crust_dt = 0.0
                y[39] = pm * y[27] * MFrac_hydrated * frac_hydrat
            else: 
               
                hydration_depth = np.max([0.0, crustal_depth * (973 - y[8])/(y[7] - y[8])])
                frac_hydrat = np.min([hydration_depth/crustal_depth,1.0])
                Max_H2O_crust = pm * Vcrust * 0.03 * frac_hydrat
                dH2O_crust_dt = dmelt_dt*MFrac_hydrated*frac_hydrat*Linear_Ocean_fraction* np.max([0.0,1 - y[39]/Max_H2O_crust])
                wet_oxidation = wet_oxid_eff * (dH2O_crust_dt/MFrac_hydrated) * Total_Fe_mass_fraction * (y[6]/(y[6]+y[5]))
                total_water_loss_surf = dH2O_crust_dt + wet_oxidation*18.0/(3.0* mu_FeO) #kg H2O/s
                total_water_gain_interior = dH2O_crust_dt
                new_wet_oxidation = wet_oxidation * 16.0 / (3.0* mu_FeO)  #kg O/s added solid
                total_wet_FeO_lost = wet_oxidation * (2.0/3.0) 
                total_wet_FeO1_5 = wet_oxidation * (2.0/3.0) * mu_FeO_1_5/mu_FeO #kg FeO1_5 /s  
                total_wet_H2_gained = wet_oxidation * 2.0/(3.0* mu_FeO) #kg H2/s

        else:
            dH2O_crust_dt = 0.0
            wet_oxidation = 0.0
            water_crust = 0.0
            [total_water_loss_surf,total_water_gain_interior,new_wet_oxidation,total_wet_FeO_lost,total_wet_FeO1_5,total_wet_H2_gained] = [0,0,0,0,0,0]

        y[19] = net_escape  
        
        #impacts (not implemented in this version of the code)
        imp_flu = imp_coef/(365*24*60*60)*np.exp(-(t0/(365*24*60*60*1e9))/tdc) #kg/s
        if imp_flu < 1e7/(365*24*60*60):
            imp_flu = 0.0
        if fO2>0:
            O_imp_sink = imp_flu * 0.3 * 8.0/(56.0+16.0) #kg O2/s for 30% FeO
            O_imp_sink = imp_flu * 0.3 * 24.0/56.0 #kg O2/s for 30% metallic iron
        else:
            O_imp_sink = 0.0
        O_imp_sink = 0.0

        if (fO2>0) and (y[2] >=rp) and(Ra>0.0)and(y[8]<=Tsolidus):  ## Calculate oxygen sinks for solid surface
            ## Assumes dry magma oxidation only happens on non-submerged surface: instant cooling of magma in water precludes significant oxidation
            O2_dry_magma_oxidation  = (1- Ocean_fraction)*dry_oxid_frac*Total_Fe_mass_fraction * (y[6]/(y[6]+y[5])) * surface_emplacement *  0.5*mu_O / mu_FeO 
            Fe_dry_magma_oxidation = (1- Ocean_fraction)*dry_oxid_frac*Total_Fe_mass_fraction * (y[6]/(y[6]+y[5])) * surface_emplacement  
            O2_magma_oxidation = new_wet_oxidation + O2_dry_magma_oxidation
            Fe_magma_oxidation = Fe_dry_magma_oxidation + total_wet_FeO_lost
            y[18] = -O2_dry_magma_oxidation
        else:
            y[18] = 0.0
            O2_dry_magma_oxidation  = 0.0
            Fe_dry_magma_oxidation = 0.0
            O2_magma_oxidation = 0.0
            Fe_magma_oxidation = 0.0
            O2_consumption = 0.0
            new_wet_oxidation = 0.0 
        
        
        y[18] = y[18] - O_imp_sink 
        y[20] = -new_wet_oxidation  #Melt_volume#
        y[21] = -O2_consumption*0.032 # crustal_depth#
        y[22] = fO2
        y[23] = CO2_Pressure_surface
        y[25] = Melt_volume 
        ##y[52] = depletion_fraction

        O2_magma_oxidation_solid = np.copy(O2_magma_oxidation)+ O2_consumption*0.032
        O2_magma_oxidation_volatile = np.copy(O2_magma_oxidation) + O2_consumption*0.032 + O_imp_sink
        Fe_magma_oxidation_solid = np.copy(Fe_magma_oxidation) + 4*O2_consumption*(0.056+0.016) # kg FeO/s
        Fe_magma_oxidation_solid_1_5 = (mu_FeO_1_5/mu_FeO) * np.copy(Fe_magma_oxidation) +  2*O2_consumption*(0.016*3 + 0.056*2) # kg FeO1.5/s
        
        # If anoxic atmosphere, adjust oxygen sinks accordingly 
        if (fO2_pos<1e-1)and(fO2>=0)and(net_escape<O2_magma_oxidation_volatile)and(y[2]>=rp)and(Ra>0.0)and(y[8]<=Tsolidus): # only do this solid state
            O2_magma_oxidation_volatile = net_escape
            if net_escape<O2_consumption*0.032+new_wet_oxidation+O_imp_sink: #fast outgassing sinks win, no crustal oxidation except via H2 escape
                Fe_dry_magma_oxidation = 0.0
                O2_dry_magma_oxidation = 0.0
                O2_magma_oxidation_solid = (O2_consumption*0.032+new_wet_oxidation - net_escape) ## gain Kg O/s as excess reductants escape
                Fe_magma_oxidation_solid = (O2_consumption*0.032+new_wet_oxidation - net_escape) * 2*mu_FeO/ mu_O # loss Kg FeO/s as excess reductant escape
                Fe_magma_oxidation_solid_1_5 = (O2_consumption*0.032+new_wet_oxidation - net_escape) * 2*mu_FeO_1_5/ mu_O # need factor of 2 bceause 4 mol FeO (or FeO1.5) for every 
            elif (net_escape > O2_consumption*0.032+new_wet_oxidation+O_imp_sink): # oxygen produced exceeds fast sinks, mopped up by magma, rate oxidation crust is just H loss
                O2_dry_magma_oxidation =   net_escape - (new_wet_oxidation + O2_consumption*0.032+O_imp_sink)
                O2_magma_oxidation_solid = (O2_consumption*0.032+new_wet_oxidation+O2_dry_magma_oxidation+O_imp_sink) ## gain Kg O/s as excess reductants escape
                Fe_magma_oxidation_solid =   (O2_consumption*0.032+new_wet_oxidation+O2_dry_magma_oxidation)* 2*mu_FeO/ mu_O # loss Kg FeO/s as excess reductant escape
                Fe_magma_oxidation_solid_1_5 =  (O2_consumption*0.032+new_wet_oxidation+O2_dry_magma_oxidation) * 2*mu_FeO_1_5/ mu_O # need factor of 2 bceause 4 mol FeO (or FeO1.5) for every 


        ###########################################################################################
        ### Time-evolution of reservoirs

        HDO_H2O_atmo = y[41]/y[1]
        if y[1] >0:
           HHDO_H2O_atmo = y[41]/y[1]
        else:
           HDO_H2O_atmo = 0.0
        if y[0] >0:
           HDO_H2O_solid = y[42]/y[0]
        else:
           HDO_H2O_solid = 0.0

        #F_TL = np.max([0.0,np.min([0.3,-0.3*(1e6*365*24*60*60)*dy7_dt/600.0])]) #newFTL, only use newFTL for volatile trapping in freezing front test
        dy0_dt = fudge * 4*np.pi * pm * kH2O * FH2O * rs**2 * drs_dt  + total_water_gain_interior - F_H2O*0.018 #oldFTL
        #dy0_dt = fudge * 4*np.pi * pm *  FH2O * rs**2 * drs_dt *( (1 - F_TL)*kH2O + F_TL)  + total_water_gain_interior - F_H2O*0.018 #newFTL

        dyHDO_dt_solid = HDO_H2O_atmo*fudge * 4*np.pi * pm * kH2O * FH2O * rs**2 * drs_dt  + HDO_H2O_atmo*total_water_gain_interior - HDO_H2O_solid*F_H2O*0.018
        if drs_dt < 0.0:
            dy0_dt = fudge * 4*np.pi * pm *  rs**2 * drs_dt * y[0] / (4./3. * np.pi * pm * (y[2]**3 - rc**3)) 
            dyHDO_dt_solid = HDO_H2O_atmo*fudge * 4*np.pi * pm *  rs**2 * drs_dt * y[0] / (4./3. * np.pi * pm * (y[2]**3 - rc**3)) 
            dy1_dt = - dy0_dt - escape
            dyHDO_dt_atmo = - HDO_H2O_atmo*fudge * 4*np.pi * pm *  rs**2 * drs_dt * y[0] / (4./3. * np.pi * pm * (y[2]**3 - rc**3))  - escape*0.1*HDO_H2O_atmo
        else:
            dy1_dt = - fudge * 4*np.pi * pm * kH2O * FH2O * rs**2 * drs_dt  - escape  - total_water_loss_surf + F_H2O*0.018    #oldFTL 
            #dy1_dt = - fudge * 4*np.pi * pm *  FH2O * rs**2 * drs_dt * ( (1 - F_TL)*kH2O + F_TL) - escape  - total_water_loss_surf + F_H2O*0.018    #newFTL 
            dyHDO_dt_atmo = - HDO_H2O_atmo*fudge * 4*np.pi * pm * kH2O * FH2O * rs**2 * drs_dt  - HDO_H2O_atmo*0.1*escape  - HDO_H2O_atmo*total_water_loss_surf + HDO_H2O_solid*F_H2O*0.018

        dy3_dt = fudge * 4 *np.pi * pm *F_FeO1_5 * rs**2 * drs_dt * 0.5*mu_O / (mu_FeO_1_5) + O2_magma_oxidation_solid ## free O in solid , factor of half because only half is free oxygen
        if drs_dt < 0.0:
            dy3_dt = fudge * 4*np.pi * pm *  rs**2 * drs_dt * (y[3] / (4./3. * np.pi * pm * (y[2]**3 - rc**3)))# * 0.5*mu_O / (mu_FeO_1_5) 
            dy4_dt = net_escape - dy3_dt
        else:
            dy4_dt = net_escape - fudge * 4 *np.pi * pm *F_FeO1_5 * rs**2 * drs_dt * 0.5*mu_O / (mu_FeO_1_5) - O2_magma_oxidation_volatile  #*np.min([1.0,abs(fO2)/10.0]) ## magma ocean and atmo, free O   

        dy5_dt = fudge * 4 *np.pi * pm * F_FeO1_5 * rs**2 * drs_dt +  Fe_magma_oxidation_solid_1_5 ## mass FeO1_5 flux, FeO + O2 = 2Fe2O3
        dy6_dt = fudge * 4 * np.pi * pm * F_FeO * rs**2 * drs_dt - Fe_magma_oxidation_solid #F_FeO flux
        # O2_consumption is mol of free O2
        # 4FeO + O2 -> 2Fe2O3, so 1 mol O2_consumption -> 2 mol Fe2O3 = 2*O2_consumption * M(Fe2O3) for kg/s Fe2O3 = kgFeO1.5
        # similarly 1 mol O2_consumption > 4 mol FeO = 4 * O2_consumption * M(FeO) for kg/s FeO 

        if drs_dt < 0.0:
            dy5_dt = fudge * 4*np.pi * pm *  rs**2 * drs_dt * (y[5] / (4./3. * np.pi * pm * (y[2]**3 - rc**3)))  
            dy6_dt = fudge * 4*np.pi * pm *  rs**2 * drs_dt * (y[6] / (4./3. * np.pi * pm * (y[2]**3 - rc**3))) 

    #### Only makes sense for increasing radius, removing CO2 from melt, FCO2 is mass fraction in melt
    ### still need to figure out crystals though
        dy13_dt = fudge * 4*np.pi * pm * kCO2 * FCO2 * rs**2 * drs_dt #- CO2_outgas ##mass solid CO2 #oldFTL
        #dy13_dt = fudge * 4*np.pi * pm *  FCO2 * rs**2 * drs_dt *( (1 - F_TL)*kCO2 + F_TL) #newFTL 
        if drs_dt < 0.0:
            dy13_dt = fudge * 4*np.pi * pm *  rs**2 * drs_dt * y[13] / (4./3. * np.pi * pm * (y[2]**3 - rc**3)) ##mass solid CO2

            dK40mantle_dt =  - (lam_Ar+lam_Ca) * y[35] / (1e9*365*24*60*60)
            dAr40mantle_dt =  (lam_Ar/(lam_Ar+lam_Ca)) * (y[2]**3 - rc**3)/(rp**3 - rc**3) * (lam_Ar+lam_Ca) * y[35] / (1e9*365*24*60*60) + pm * drs_dt * 4 * np.pi * y[2]**2 * (y[36]/Mantle_mass) 
            dK40lid_dt = 0.0
            dAr40atmo_dt = - pm * drs_dt * 4 * np.pi * y[2]**2 * (y[36]/Mantle_mass)  +(lam_Ar/(lam_Ar+lam_Ca)) * (rp**3 - y[2]**3)/(rp**3 - rc**3) * (lam_Ar+lam_Ca) * y[35] / (1e9*365*24*60*60)

            dU238_mantle_dt = -(np.log(2)/4.46e9) * y[43] / (365*24*60*60)
            dU235_mantle_dt = -(np.log(2)/7.04e8) * y[44] / (365*24*60*60)
            dTh_mantle_dt = - (np.log(2)/1.4e10) * y[45] / (365*24*60*60)

            dU238_lid_dt = 0.0
            dU235_lid_dt = 0.0
            dTh_lid_dt = 0.0
                 
            dHe_mantle_dt =   (y[2]**3 - rc**3)/(rp**3 - rc**3) * ( (4./238) * 8.0 * (np.log(2)/4.46e9) * y[43] + (4./235) * 7.0 * (np.log(2)/7.04e8) * y[44]+ (4./232) * 6.0 * (np.log(2)/1.4e10) * y[45]) /  (365*24*60*60) + pm * drs_dt * 4 * np.pi * y[2]**2 * (y[49]/Mantle_mass) 
            dHe_atmo_dt  =  -  pm * drs_dt * 4 * np.pi * y[2]**2 * (y[49]/Mantle_mass) + (rp**3 - y[2]**3)/(rp**3 - rc**3) * ( (4./238) * 8.0 * (np.log(2)/4.46e9) * y[43] + (4./235) * 7.0 * (np.log(2)/7.04e8) * y[44]+ (4./232) * 6.0 * (np.log(2)/1.4e10) * y[45]) /  (365*24*60*60) + pm * drs_dt * 4 * np.pi * y[2]**2 * (y[49]/Mantle_mass)  

    
        dy12_dt = -dy13_dt - CO2_loss ##  escape CO2 assumed
        dy18_dt = 0.0
        

        Weather = 0.0
        Outgas = 0.0
        dCO2_crust_dt = 0.0
        
        if (C_cycle_switch == "y")and(SurfT<Tsolidus):
            Outgas = F_CO2*0.044
            Weather = weathering_flux(t0,CO2_Pressure_surface,float(SurfT),y[7],water_frac,efold_weath,H2O_Pressure_surface,g,y[12],alpha_exp,supp_lim,ocean_pH,omega_ocean,Abs_crustal_prod,float(y[24]))
            if (CO2_Pressure_surface<50)and(CO2_Pressure_surface>0)and(water_frac<0.9999999)and(SurfT<647)and(Weather>Outgas): 
                Weather = Outgas
            dy12_dt = dy12_dt - Weather + Outgas
            dy13_dt = dy13_dt + Weather - Outgas
            if Mass_oceans_crude >0:
                dy18_dt = 0.0

        y[16] = rp - (rp**3 - actual_phi_surf_melt * Va*3./(4.*np.pi))**(1./3.) #crustal depth

        dy26_dt = ddelta_dt
        dy27_dt = dVcrust_dt
    
        dy28_dt = dQmantle_dt
        dy29_dt = dQcrust_dt

        dy35_dt =  dK40mantle_dt 
        dy36_dt =  dAr40mantle_dt 
        dy37_dt =  dK40lid_dt
        dy38_dt =  dAr40atmo_dt  

        dy39_dt = dH2O_crust_dt
        dy40_dt = dCO2_crust_dt

         
        dy41_dt = DH_switch*dyHDO_dt_atmo
        dy42_dt = DH_switch*dyHDO_dt_solid

        dy43_dt = dU238_mantle_dt
        dy44_dt = dU235_mantle_dt
        dy45_dt = dTh_mantle_dt
        dy46_dt = dU238_lid_dt
        dy47_dt = dU235_lid_dt
        dy48_dt = dTh_lid_dt
        dy49_dt = dHe_mantle_dt
        dHe_atmo_dt = dHe_atmo_dt - He_escape_loss
        dy50_dt = dHe_atmo_dt

        toc = time.time()
        y[15] = 1e-12 * Outgas*365*24*60*60/0.044 #convert outgassing flux (Kg/s) to Tmol CO2/yr
        y[14] = 1e-12 * Weather*365*24*60*60/0.044 #convert weathering flux (Kg/s) to Tmol CO2/yr

        y[53] = escape + wet_oxidation*18.0/(3.0* mu_FeO)
        y[54] = total_water_gain_interior
        y[55] = F_H2O*0.018

        y[24] = (fO2_pos*0.032 + atmo_H2O*0.018 + CO2_Pressure_surface*0.044 + 1e5*0.028)/Pressure_surface  #Mean molecular weight

        return [dy0_dt,dy1_dt,dy2_dt,dy3_dt,dy4_dt,dy5_dt,dy6_dt,dy7_dt,0.0,0.0,0.0,0.0,dy12_dt,dy13_dt,0.0,0.0,0.0,0.0,dy18_dt,0.0,0.0,0.0,0.0,0.0,0.0,0.0,dy26_dt,dy27_dt,dy28_dt,dy29_dt,0.0,0.0,0.0,0.0,0.0,dy35_dt,dy36_dt,dy37_dt,dy38_dt,dy39_dt,dy40_dt,dy41_dt,dy42_dt,dy43_dt,dy44_dt,dy45_dt,dy46_dt,dy47_dt,dy48_dt,dy49_dt,dy50_dt,0.0,ddt_depletion_fraction,0.0,0.0,0.0]

    ########################################################################################### end forward model function
    ######################################################################################################################
    ######################################################################################################################

    ##############################################
    # Iron speciation and oxygen fugacity functions
    @jit(nopython=True) 
    def fff(logXFe2O3,XMgO,XSiO2,XAl2O3,XCaO,XNa2O,XK2O,y4,P,T,Total_Fe,Mliq,rs,rp,MMW):
        XFe2O3 = np.exp(logXFe2O3)
        m_sil = XMgO * (16.+25.) + XSiO2 * (28.+32.) + XAl2O3 * (27.*2.+3.*16.) + XCaO * (40.+16.) + XFe2O3 * (56.*2 + 16.*3) + (Total_Fe-2*XFe2O3) * (56.0+16.0)
        if (y4 - Mliq *  (0.5*16/(56+1.5*16))*XFe2O3*(56*2+3*16)/m_sil) <0:
            return 1e8
        ## g per mol of BSE.... so on next line mol Xfe / mol BSE * gXfe/molXfe / g/mol BSE = g Xfe / mol BSe / g/mol BSE = gXfe/g BSE
        #terms1 = 0.196*np.log( (y4 - Mliq *  XFe2O3*(56*2+3*16)/m_sil) / (4*np.pi*(rp**2)/g)) + 11492.0/T - 6.675 - 2.243*XAl2O3
        terms1 = 0.196*np.log( (MMW/0.032) * (y4 - Mliq *  (0.5*16/(56+1.5*16))*XFe2O3*(56*2.0+3.0*16.0)/m_sil) / (4*np.pi*(rp**2)/g)) + 11492.0/T - 6.675 - 2.243*XAl2O3 
        terms2 = 3.201*XCaO + 5.854 * XNa2O
        terms3 = 6.215*XK2O - 3.36 * (1 - 1673.0/T - np.log(T/1673.0))
        terms4 = -7.01e-7 * P/T - 1.54e-10 * P * (T - 1673)/T + 3.85e-17 * P**2 / T
        terms = terms1+terms2+terms3+terms4
        return (np.log((XFe2O3 /(Total_Fe -2*XFe2O3))) + 1.828 * Total_Fe - terms)**2.0 
        

    @jit(nopython=True) 
    def fff2(logXFe2O3,XMgO,XSiO2,XAl2O3,XCaO,XNa2O,XK2O,y4,P,T,Total_Fe,Mliq,rs,rp,MMW):
        XFe2O3 = np.exp(logXFe2O3[0])
        m_sil = XMgO * (16.+25.) + XSiO2 * (28.+32.) + XAl2O3 * (27.*2.+3.*16.) + XCaO * (40.+16.) + XFe2O3 * (56.*2 + 16.*3) + (Total_Fe-2*XFe2O3) * (56.0+16.0)
        if (y4 - Mliq *  (0.5*16/(56+1.5*16))*XFe2O3*(56*2+3*16)/m_sil) <0:
            return -1e8
        ## g per mol of BSE, so on next line mol Xfe / mol BSE * gXfe/molXfe / g/mol BSE = g Xfe / mol BSe / g/mol BSE = gXfe/g BSE
        terms1 = 0.196*np.log( 1e-5*(MMW/0.032) * (y4 - Mliq *  (0.5*16/(56+1.5*16))*XFe2O3*(56*2.0+3.0*16.0)/m_sil) / (4*np.pi*(rp**2)/g)) + 11492.0/T - 6.675 - 2.243*XAl2O3  ## fO2 in bar not Pa
        terms2 = 3.201*XCaO + 5.854 * XNa2O
        terms3 = 6.215*XK2O - 3.36 * (1 - 1673.0/T - np.log(T/1673.0))
        terms4 = -7.01e-7 * P/T - 1.54e-10 * P * (T - 1673)/T + 3.85e-17 * P**2 / T
        terms = terms1+terms2+terms3+terms4  
        return -(np.log((XFe2O3 /(Total_Fe -2*XFe2O3))) + 1.828 * Total_Fe - terms)**2.0 


    @jit(nopython=True)         
    def solve_fO2_F_redo(y4,P,T,Total_Fe,Mliq,rs,rp,MMW): 
        if T > Tsolidus:
            XAl2O3 = 0.022423 
            XCaO = 0.0335
            XNa2O = 0.0024 
            XK2O = 0.0001077 
            XMgO = 0.478144  
            XSiO2 =  0.4034   

            m_sil = XMgO * (16.+25.) + XSiO2 * (28.+32.) + XAl2O3 * (27.*2.+3.*16.) + XCaO * (40.+16.) +  (Total_Fe) * (56.0+16.0)

            initialize_fast = np.array(float(-50.0))
            logXFe2O3 =  nelder_mead(fff2, x0=initialize_fast, bounds=np.array([[-100.0], [0.0]]).T, args = (XMgO,XSiO2,XAl2O3,XCaO,XNa2O,XK2O,y4,P,T,Total_Fe,Mliq,rs,rp,MMW), tol_f=1e-10,tol_x=1e-10, max_iter=1000)
            XFe2O3 = np.exp(logXFe2O3.x[0])

            XFeO =(Total_Fe - 2*XFe2O3)
            m_sil = XMgO * (16.+25.) + XSiO2 * (28.+32.) + XAl2O3 * (27.*2.+3.*16.) + XCaO * (40.+16.) + XFe2O3 * (56.*2 + 16.*3) + XFeO * (56.0+16.0)
            F_FeO1_5 = XFe2O3*(56.0*2.0+3.0*16.0)/m_sil 
            F_FeO = XFeO * (56.0 + 16.0) / m_sil 
            fO2_out =  (MMW/0.032) *(y4 - (0.5*16/(56+1.5*16)) * Mliq * XFe2O3*(56.0*2.0+3.0*16.0)/m_sil ) / (4*np.pi*(rp**2)/g)

        else:
            fO2_out =  (MMW/0.032) *(y4 / (4*np.pi*(rp**2)/g))
            XFeO = 0.0
            XFe2O3 = 0.0
            F_FeO1_5 = 0.0
            F_FeO = 0.0
        return [XFeO,XFe2O3,fO2_out,F_FeO1_5,F_FeO]
    ##############################################
    ##############################################

    ### Initialize forward model

    init_fluid_HDO = 2*Init_D_to_H * Init_fluid_H2O*0.019/0.018
    ICs = [Init_solid_H2O,Init_fluid_H2O, rc, Init_solid_O,Init_fluid_O,Init_solid_FeO1_5,Init_solid_FeO,4000,3999,0.0,0.0,0.0,Init_fluid_CO2,Init_solid_CO2,0.0,0.0,3999.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.044,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,init40K,initAr40,0.0,0.0,0.0,0.0,init_fluid_HDO,0.0,init_U238,init_U235,init_Th,0.0,0.0,0.0,init_He_mantle,init_He_atmo,0.5,0.0,0.0,0.0,0.0] 

    ### Various numerical inputs 
    if Numerics.total_steps==3:
        sol = solve_ivp(system_of_equations, [Start_time*365*24*60*60, Numerics.tfin0*365*24*60*60], ICs,dense_output=True, method = 'RK45',max_step=Numerics.step0*365*24*60*60) 
        ## this next sol2 changed from RK23 to RK45 to reduce model failure for waterworlds, but RK23 seems to work better for nominal!!
        sol2 = solve_ivp(system_of_equations, [sol.t[-1], Numerics.tfin1*365*24*60*60], sol.y[:,-1], method = 'RK23', vectorized=False, max_step=Numerics.step1*365*24*60*60)
        sol3 = solve_ivp(system_of_equations, [sol2.t[-1], Numerics.tfin2*365*24*60*60], sol2.y[:,-1], method = 'RK23', vectorized=False, max_step=Numerics.step2*365*24*60*60)
        sol4 = solve_ivp(system_of_equations, [sol3.t[-1], Max_time*365*24*60*60], sol3.y[:,-1], method = 'RK23', vectorized=False, max_step=Numerics.step3*365*24*60*60)
        total_time = np.concatenate((sol.t,sol2.t,sol3.t,sol4.t))
        total_y = np.concatenate((sol.y,sol2.y,sol3.y,sol4.y),axis=1)
    elif Numerics.total_steps ==2: ### works better for Venus maybe?
        sol = solve_ivp(system_of_equations, [Start_time*365*24*60*60, Numerics.tfin0*365*24*60*60], ICs,dense_output=True, method = 'RK45',max_step=Numerics.step0*365*24*60*60)#,max_step=365*24*60*60*1e3) # FIX REASONABLE ICs?
        sol2 = solve_ivp(system_of_equations, [sol.t[-1], Numerics.tfin1*365*24*60*60], sol.y[:,-1], method = 'RK45', vectorized=False, max_step=Numerics.step1*365*24*60*60)
        sol3 = solve_ivp(system_of_equations, [sol2.t[-1],Max_time*365*24*60*60], sol2.y[:,-1], method = 'RK23', vectorized=False, max_step=Numerics.step2*365*24*60*60)
        total_time = np.concatenate((sol.t,sol2.t,sol3.t))
        total_y = np.concatenate((sol.y,sol2.y,sol3.y),axis=1)
    elif Numerics.total_steps == 7 : ### Venus second attempt (actually still 2 steps)
        sol = solve_ivp(system_of_equations, [Start_time*365*24*60*60, Numerics.tfin0*365*24*60*60], ICs,dense_output=True, method = 'RK45',max_step=Numerics.step0*365*24*60*60)#,max_step=365*24*60*60*1e3) # FIX REASONABLE ICs?
        sol2 = solve_ivp(system_of_equations, [sol.t[-1], Numerics.tfin1*365*24*60*60], sol.y[:,-1], method = 'RK23', vectorized=False, max_step=Numerics.step1*365*24*60*60)
        sol3 = solve_ivp(system_of_equations, [sol2.t[-1],Max_time*365*24*60*60], sol2.y[:,-1], method = 'RK23', vectorized=False, max_step=Numerics.step2*365*24*60*60)
        total_time = np.concatenate((sol.t,sol2.t,sol3.t))
        total_y = np.concatenate((sol.y,sol2.y,sol3.y),axis=1)
    elif Numerics.total_steps == 8 : ### Venus second attempt (actually still 2 steps)
        sol = solve_ivp(system_of_equations, [Start_time*365*24*60*60, Numerics.tfin0*365*24*60*60], ICs,dense_output=True, method = 'RK45',max_step=Numerics.step0*365*24*60*60/10.0)#,max_step=365*24*60*60*1e3) # FIX REASONABLE ICs?
        sol2 = solve_ivp(system_of_equations, [sol.t[-1], Numerics.tfin1*365*24*60*60], sol.y[:,-1], method = 'RK23', vectorized=False, max_step=Numerics.step1*365*24*60*60/10.0)
        sol3 = solve_ivp(system_of_equations, [sol2.t[-1],Max_time*365*24*60*60], sol2.y[:,-1], method = 'RK23', vectorized=False, max_step=Numerics.step2*365*24*60*60)
        total_time = np.concatenate((sol.t,sol2.t,sol3.t))
        total_y = np.concatenate((sol.y,sol2.y,sol3.y),axis=1)
    elif Numerics.total_steps == 9 : ### Venus second attempt (actually still 2 steps)
        sol = solve_ivp(system_of_equations, [Start_time*365*24*60*60, Numerics.tfin0*365*24*60*60], ICs,dense_output=True, method = 'RK23',max_step=Numerics.step0*365*24*60*60)#,max_step=365*24*60*60*1e3) # FIX REASONABLE ICs?
        sol2 = solve_ivp(system_of_equations, [sol.t[-1], Numerics.tfin1*365*24*60*60], sol.y[:,-1], method = 'RK23', vectorized=False, max_step=Numerics.step1*365*24*60*60)
        sol3 = solve_ivp(system_of_equations, [sol2.t[-1],Max_time*365*24*60*60], sol2.y[:,-1], method = 'RK23', vectorized=False, max_step=Numerics.step2*365*24*60*60)
        total_time = np.concatenate((sol.t,sol2.t,sol3.t))
        total_y = np.concatenate((sol.y,sol2.y,sol3.y),axis=1)


    elif Numerics.total_steps == 0: #manually enter ICs and new start time
        ICs = [4.42928490e+20,   2.17603056e+20,   6.37100000e+06,4.90476289e+21,   2.38085608e+21,   4.90476289e+22, 2.58963126e+23,   1.53445831e+03,   3.47126528e+02,2.60727569e+02,   2.59807581e+02,   9.19988525e-01, 5.51838330e+17,   3.99448162e+20,   5.42073065e+03,8.74370829e+15,   6.35569496e+06,   0.00000000e+00]
        Start_time_new = 277333013.66152197
        sol = solve_ivp(system_of_equations, [Start_time_new*365*24*60*60, Numerics.tfin0*365*24*60*60], ICs,dense_output=True, method = 'LSODA',max_step=Numerics.step0*365*24*60*60)#,max_step=365*24*60*60*1e3) 
        total_time = sol.t
        total_y = sol.y     

    ###################################
    ## Filling in output arrays (post-processing)
    t_array = total_time 
    FH2O_array= 0.0 * np.copy(t_array)
    FCO2_array= 0.0 * np.copy(t_array)
    MH2O_liq = np.copy(FH2O_array)
    MH2O_crystal = np.copy(FH2O_array)
    MCO2_liq = np.copy(FH2O_array)
    MCO2_crystal = np.copy(FH2O_array)
    Pressre_H2O = np.copy(FH2O_array)
    CO2_Pressure_array = np.copy(FH2O_array)
    CO2_Pressure_array_atmo = np.copy(FH2O_array)
    fO2_array = np.copy(FH2O_array)
    Mass_O_atm =  np.copy(FH2O_array)
    Mass_O_dissolved = np.copy(FH2O_array)
    water_frac = np.copy(FH2O_array)
    Ocean_depth = np.copy(FH2O_array)
    Max_depth = 11400 * (9.8 / g)  + 0.0*Ocean_depth
    Ocean_fraction = np.copy(FH2O_array)
    f_O2_FMQ = np.copy(FH2O_array)
    f_O2_IW = np.copy(FH2O_array)
    f_O2_MH = np.copy(FH2O_array)
    f_O2_mantle = np.copy(FH2O_array)
    
    for i in range(0,len(t_array)): #Post model-run processing of outputs
        rs = total_y[2][i]
        Mliq = Mliq_fun(total_y[2][i],rp,rs,pm)
        Mcrystal = 0.0
        Mcrystal = total_y[17][i]
        [FH2O_array[i],Pressre_H2O[i]] = H2O_partition_function( total_y[1][i],Mliq,Mcrystal,rp,g,kH2O,total_y[24][i])
        [FCO2_array[i],CO2_Pressure_array[i]] = CO2_partition_function( total_y[12][i],Mliq,Mcrystal,rp,g,kCO2,total_y[24][i])
        ocean_CO3 = float(omega_ocean * Sol_prod(total_y[8][i]) / ocean_Ca)
        
        [heat_atm,newpCO2,ocean_pH,ALK,Mass_oceans_crude,DIC_check] = correction(float(total_y[8][i]),float(Te_fun(t_array[i])),float(Pressre_H2O[i]),float(CO2_Pressure_array[i]),rp,g,ocean_CO3,1e5,float(total_y[24][i]),float(total_y[22][i])) 
        CO2_Pressure_array_atmo[i] = newpCO2

        MH2O_liq[i] = (Mliq - Mcrystal) * FH2O_array[i]
        MH2O_crystal[i] = kH2O * Mcrystal * FH2O_array[i]
    
        MCO2_liq[i] = (Mliq- Mcrystal) * FCO2_array[i]
        MCO2_crystal[i] = kCO2 * Mcrystal* FCO2_array[i]
    
        water_frac[i] = my_water_frac(float(total_y[8][i]),float(Te_fun(t_array[i])),float(Pressre_H2O[i]),float(CO2_Pressure_array_atmo[i]))
        Ocean_depth[i] = (0.018/total_y[24][i])*(1-water_frac[i]) * Pressre_H2O[i] / (g*1000)
        
        Ocean_fraction[i] = np.min([1.0,(Ocean_depth[i]/Max_depth[0])**0.25 ])
        
        [XFeO,XFe2O3,fO2,F_FeO1_5,F_FeO] = solve_fO2_F_redo(total_y[4][i],Pressre_H2O[i],total_y[8][i],Total_Fe_mol_fraction,Mliq,rs,rp,total_y[24][i])
        #fO2_array[i] = fO2 
        fO2_array[i] = total_y[22][i]
        fO2  =  total_y[22][i]
        Pressure_surface = fO2 + Pressre_H2O[i]*water_frac[i] + CO2_Pressure_array_atmo[i] + 1e5      

        Mass_O_dissolved[i] = Mliq * F_FeO1_5*0.5*16/(56+1.5*16)
        Mass_O_atm[i] = fO2*4 *(0.032/total_y[24][i])*np.pi * rp**2 / g
        
        f_O2_FMQ[i] = buffer_fO2(total_y[7][i],Pressure_surface/1e5,'FMQ')
        f_O2_IW[i] = buffer_fO2(total_y[7][i],Pressure_surface/1e5,'IW')
        f_O2_MH[i] =  buffer_fO2(total_y[7][i],Pressure_surface/1e5,'MH')
        iron3 = total_y[5][i]*56/(56.0+1.5*16.0)
        iron2 = total_y[6][i]*56/(56.0+16.0)
        iron_ratio = iron3/iron2
        f_O2_mantle[i] = get_fO2(0.5*iron3/iron2,Pressure_surface,total_y[7][i],Total_Fe_mol_fraction)
    
    total_time = total_time/(365*24*60*60)-Start_time

#########################################################################################################################################################
#########################################################################################################################################################
#### Everything is done - the rest is optional plotting of individual model runs for diagnostic purposes ################################################
#########################################################################################################################################################
#########################################################################################################################################################
    
    if plot_switch == "y": # for diagnostic plotting of individual model runs - labels may not be correct
 
        pylab.figure()
        pylab.subplot(2,1,1)
        pylab.semilogx(new_t*1e9-1e7,Absolute_total_Lum)
        pylab.ylabel('Stellar flux (W)')
        pylab.subplot(2,1,2)
        pylab.semilogx(new_t*1e9-1e7,Absolute_XUV_Lum/Absolute_total_Lum)
        pylab.ylabel('XUV/Total Lum.')
        pylab.xlabel('Time (yrs)')

        pylab.figure()
        pylab.ylabel('MMW')
        pylab.semilogx(total_time,total_y[24])
             
        pylab.figure()
        pylab.subplot(6,1,1)
        pylab.ylabel('Mass H2O solid, kg')
        pylab.semilogx(total_time,total_y[0])
        pylab.subplot(6,1,2)
        pylab.ylabel('H2O reservoir (kg)')
        pylab.semilogx(total_time,total_y[1],'r',label='Mass H2O, magma ocean + atmo')
        pylab.semilogx(total_time,MH2O_liq,'b',label='Mass H2O, magma ocean')
        pylab.semilogx(total_time,Pressre_H2O *4 * (0.018/total_y[24]) * np.pi * (rp**2/g) ,label='Mass H2O atmosphere')
        pylab.semilogx(total_time,MH2O_crystal,label= 'crystal H2O')
        pylab.semilogx(total_time,total_y[0] +MH2O_liq + Pressre_H2O *4 * (0.018/total_y[24]) * np.pi * (rp**2/g)+MH2O_crystal ,'g--' ,label='Total H2O (kg)')
        pylab.legend()
        pylab.subplot(6,1,3)
        pylab.ylabel('Radius of solidification (m)')
        pylab.semilogx(total_time,total_y[2])
        pylab.subplot(6,1,4)
        pylab.ylabel("Pressure (bar)")
        pylab.loglog(total_time,Mass_O_atm*g/(4*(0.032/total_y[24])*np.pi*rp**2*1e5),label='fO2')
        pylab.loglog(total_time,Pressre_H2O/1e5,label='fH2O')
        pylab.loglog(total_time,total_y[23]/1e5,label='fCO2')
        pylab.loglog(total_time,CO2_Pressure_array/1e5,label='fCO2 total')
        pylab.legend()
        pylab.subplot(6,1,5)
        pylab.ylabel('O reservoir (kg)')
        pylab.semilogx(total_time,total_y[3],'k' ,label = 'Oxygen in solid' )
        pylab.semilogx(total_time,total_y[4],'r' ,label = 'Oxygen in magma ocean + atmo')
        pylab.legend()
        pylab.subplot(6,1,6)
        pylab.ylabel('Solid Fe reservoir (kg)')
        pylab.semilogx(total_time,total_y[5],'k' ,label = 'FeO1.5')
        pylab.semilogx(total_time,total_y[6],'r'  ,label = 'FeO') # iron 2+
        pylab.legend()
        pylab.xlabel('Time (yrs)')
        
        pylab.figure()
        pylab.subplot(3,1,1)
        pylab.plot(total_time,Mass_O_dissolved,'k',label='Magma oc')
        pylab.plot(total_time,Mass_O_atm,'r--', label= "Atmo")
        pylab.plot(total_time,Mass_O_dissolved+Mass_O_atm,'y*',label='Magma oc + atm')
        pylab.semilogx(total_time,total_y[3],'c' ,label = 'Oxygen in solid' )
        pylab.semilogx(total_time,Mass_O_dissolved+Mass_O_atm + total_y[3],'m' ,label = 'Total free O' )
        pylab.xlabel('Time (yrs)')
        pylab.ylabel('Free oxgen reservoir (kg)')
        pylab.legend()
        
        iron3 = total_y[5]*56/(56.0+1.5*16.0)
        iron2 = total_y[6]*56/(56.0+16.0)
        iron_ratio = iron3/(iron3+iron2)
        pylab.subplot(3,1,2)
        pylab.title("Fe3+/TotalFe in solid, mol ratio")
        pylab.semilogx(total_time,iron_ratio,'k')
        pylab.xlabel('Time (yrs)')
        
        pylab.subplot(3,1,3)
        pylab.ylabel('CO2 reservoir (kg)')
        pylab.semilogx(total_time,total_y[12],'r',label='Mass CO2, magma ocean + atmo')
        pylab.semilogx(total_time,MCO2_liq,'b',label='Mass CO2, magma ocean')
        pylab.semilogx(total_time,total_y[23] *4 *(0.044/total_y[24]) * np.pi * (rp**2/g) ,label='Mass CO2 atmosphere')
        pylab.semilogx(total_time,CO2_Pressure_array* 4 *(0.044/total_y[24])* np.pi* (rp**2/g),label= 'Mass CO2 volatiles')
        pylab.semilogx(total_time,MCO2_crystal,label= 'crystal CO2')
        pylab.semilogx(total_time,total_y[13],label= 'Mantle CO2')
        pylab.semilogx(total_time,total_y[13] +MCO2_liq + CO2_Pressure_array *4 *(0.044/total_y[24])* np.pi * (rp**2/g) +MCO2_crystal,'g--' ,label='Total CO2 (kg)')
        pylab.legend()
        
        pylab.figure()
        pylab.subplot(7,1,1)
        pylab.semilogx(total_time,total_y[7])
        pylab.ylabel("Mantle potential temperature (K)")
        pylab.subplot(7,1,2)
        pylab.semilogx(total_time,total_y[2])
        pylab.ylabel("Radius of solidification (m)")
        pylab.subplot(7,1,3)
        pylab.semilogx(total_time,total_y[8])
        pylab.ylabel("Surface temperature (K)")
        pylab.xlabel("Time (yrs)")
        pylab.subplot(7,1,4)
        pylab.semilogx(total_time,total_y[7],'b')
        pylab.semilogx(total_time,total_y[8],'r')
        pylab.subplot(7,1,5)
        pylab.semilogx(total_time,water_frac)
        pylab.subplot(7,1,6)
        pylab.semilogx(total_time,water_frac*Pressre_H2O *4 *(0.018/total_y[24])* np.pi * (rp**2/g) , 'g' ,label = 'Atmospheric H2O' )
        pylab.semilogx(total_time,(1 - water_frac)*Pressre_H2O *4 *(0.018/total_y[24])* np.pi * (rp**2/g),'m' ,label = 'Ocean H2O' )
        pylab.ylabel("Surface water (kg)")
        pylab.xlabel("Time (yrs)")
        pylab.legend()
        pylab.subplot(7,1,7)
        pylab.loglog(total_time,total_y[9] , 'b' ,label = 'OLR' )
        pylab.loglog(total_time,total_y[10] , 'r' ,label = 'ASR' )
        pylab.loglog(total_time,total_y[11] , 'g' ,label = 'qm' )
        pylab.ylabel("OLR and ASR (W/m2)")
        pylab.xlabel("Time (yrs)")
        pylab.legend()

        pylab.figure()
        pylab.subplot(3,1,1)
        pylab.title('qm')
        pylab.loglog(total_time,total_y[11])
        pylab.xlim([1.0, np.max(total_time)])
        pylab.subplot(3,1,2)
        pylab.loglog(total_time,total_y[14]/1000.0)
        pylab.title('delta_u')
        pylab.xlim([1.0, np.max(total_time)])
        pylab.subplot(3,1,3)
        pylab.title('Ra')
        pylab.loglog(total_time,total_y[15])
        pylab.xlim([1.0, np.max(total_time)])
        
        pylab.figure()
        pylab.semilogx(total_time,total_y[9] - total_y[10] - total_y[11] , 'k' ,label = 'net loss' )
        pylab.ylabel("Radiation balance (W/m2)")
        pylab.xlabel("Time (yrs)")
        pylab.legend()
        
        
        pylab.figure()
        pylab.subplot(2,1,1)
        pylab.semilogx(total_time,Max_depth/1000.0,'r-',label='Max depth')
        pylab.semilogx(total_time,Ocean_depth/1000.0,'b',label='Ocean depth')
        pylab.legend()
        pylab.subplot(2,1,2)
        pylab.semilogx(total_time,Ocean_fraction,'k-',label='Ocean fraction')
        pylab.semilogx(total_time,total_time*0 + (2.5/11.4)**0.25,'b-',label='Modern Earth')
        pylab.legend()

        pylab.figure()
        pylab.subplot(5,1,1)
        pylab.semilogx(total_time,total_y[7],'b',label = "Mantle Temp.")
        pylab.semilogx(total_time,total_y[8],'r',label = "Surface Temp.")
        pylab.ylabel("Temperature (K)")
        pylab.xlabel("Time (yrs)")
        pylab.xlim([1.0, np.max(total_time)])
        pylab.legend()

        pylab.subplot(5,1,2)
        pylab.semilogx(total_time,total_y[2])
        pylab.ylabel("Radius of solidification (m)")
        pylab.xlabel("Time (yrs)")
        pylab.xlim([1.0, np.max(total_time)])
        
        pylab.subplot(5,1,3)
        pylab.ylabel("Pressure (bar)")
        pylab.loglog(total_time,Mass_O_atm*g/(4*(0.032/total_y[24])*np.pi*rp**2*1e5),label='fO2')
        pylab.loglog(total_time,water_frac*Pressre_H2O/1e5,label='fH2O')
        pylab.loglog(total_time,total_y[23]/1e5,label='fCO2')
        pylab.xlabel("Time (yrs)")
        pylab.xlim([1.0, np.max(total_time)])
        pylab.legend()
        
        pylab.subplot(5,1,4)
        pylab.ylabel('Solid Fe reservoir (kg)')
        pylab.semilogx(total_time,total_y[5],'k' ,label = 'FeO1.5')
        pylab.semilogx(total_time,total_y[6],'r'  ,label = 'FeO') # iron 2+
        pylab.xlabel('Time (yrs)')
        pylab.xlim([1.0, np.max(total_time)])
        pylab.legend()
        
        pylab.subplot(5,1,5)
        pylab.ylabel('C fluxes')
        pylab.semilogx(total_time,total_y[14],'k' ,label = 'Weathering/Escape')
        pylab.semilogx(total_time,total_y[15],'r' ,label = 'Outgassing')
        pylab.xlabel('Time (yrs)')
        pylab.xlim([1.0, np.max(total_time)])
        pylab.legend()
               
        pylab.figure()
        pylab.subplot(4,1,1)
        pylab.title('CO2 and H2O atmosphere')
        pylab.semilogx(total_time,total_y[7],'b',label = "Tp, my model")
        pylab.semilogx(total_time,total_y[8],'r',label = "Ts, my model")
        pylab.ylabel("Temperature (K)")
        pylab.xlabel("Time (yrs)")
        pylab.xlim([1.0, np.max(total_time)])
        
        pylab.legend()
        
        pylab.subplot(4,1,2)
        pylab.semilogx(total_time,total_y[2]/1000,'b')
        pylab.semilogx(total_time,total_y[16]/1000,'r') #radius check
        pylab.ylabel("Solidus radius (km)")
        pylab.xlabel("Time (yrs)")
        pylab.xlim([1.0, np.max(total_time)])
        
        pylab.subplot(4,1,3)
        pylab.loglog(total_time,f_O2_FMQ,'b',label = 'FMQ')
        pylab.loglog(total_time,f_O2_IW,'r',label = 'IW') 
        pylab.loglog(total_time,f_O2_MH,'k',label = 'MH') 
        pylab.loglog(total_time,f_O2_mantle,'g--', label = 'fO2') 
        pylab.ylabel("Oxygen fugacity mantle")
        pylab.xlabel("Time (yrs)")
        pylab.xlim([1.0, np.max(total_time)])
        pylab.legend()

        pylab.subplot(4,1,4)
        pylab.title('Redox Budget')
        pylab.semilogx(total_time,total_y[18]*365*24*60*60/(0.032*1e12),'g' ,label = 'Dry crustal')
        pylab.semilogx(total_time,total_y[19]*365*24*60*60/(0.032*1e12),'k' ,label = 'Net Escape')
        pylab.semilogx(total_time,total_y[20]*365*24*60*60/(0.032*1e12),'b' ,label = 'Wet crustal oxidation')
        pylab.semilogx(total_time,total_y[21]*365*24*60*60/(0.032*1e12),'r' ,label = 'Outgassing')
        pylab.semilogx(total_time,(total_y[18]+total_y[19]+total_y[20]+total_y[21])*365*24*60*60/(0.032*1e12),'c--' ,label = 'Net')
        pylab.ylabel("O2 flux (Tmol/yr)")
        pylab.xlabel("Time (yrs)")
        pylab.xlim([1.0, np.max(total_time)])
        pylab.legend()
        
        pylab.figure()
        pylab.loglog(total_time,total_y[16],'r') #radius check
        pylab.figure()
        pylab.title('Fe conservation test')
        pylab.loglog(total_time,total_y[5]/(0.056+1.5*0.016),'b')
        pylab.loglog(total_time,total_y[6]/(0.056+0.016),'r')
        pylab.loglog(total_time,total_y[5]/(0.056+1.5*0.016)+total_y[6]/(0.056+0.016),'k')

        pylab.figure()
        pylab.subplot(5,1,1)
        pylab.loglog(total_time,total_y[22],'r') #radius check
        pylab.loglog(total_time,fO2_array,'b--')

        pylab.subplot(5,1,2)
        pylab.semilogx(total_time,total_y[26],'r',label='delta') #radius check
        deltac_plot = rp - (rp**3 - 3*total_y[27]/(4.0*np.pi))**(1./3.)
        pylab.semilogx(total_time,deltac_plot,'g--',label='deltac') #radius check
        pylab.semilogx(total_time,total_y[34],'b',label='delta_meltr') #radius check

        pylab.legend()

        pylab.subplot(5,1,3)
        pylab.loglog(total_time,total_y[25]*365*24*60*60/1e9,'r') #radius check
        pylab.ylabel('Melt production (km3/yr)')

        pylab.subplot(5,1,4)
        pylab.semilogx(total_time,total_y[27],'r') #radius check
        

        pylab.subplot(5,1,5)
        pylab.semilogx(total_time,total_y[30]/1e12,'r',label='mantle') #radius check
        pylab.semilogx(total_time,total_y[31]/1e12,'b',label='crust') #radius check
        pylab.semilogx(total_time,total_y[30]/1e12+total_y[31]/1e12,'g',label='total')
        pylab.legend()

        pylab.figure()
        pylab.loglog(total_time,total_y[32]/1e12,'g',label='Convective loss') #radius check
        pylab.loglog(total_time,total_y[33]/1e12,'k',label='Volanic loss') #radius check
        pylab.loglog(total_time,total_y[30]/1e12,'r',label='Mantle heat prod')
        pylab.loglog(total_time,total_y[32]/1e12+total_y[33]/1e12,'b--',label='Total loss (TW)')
        pylab.legend()

        pylab.figure()
        pylab.loglog(total_time,1000*total_y[32]/(4*np.pi*rp**2),'g',label='Convective loss') #radius check
        pylab.loglog(total_time,1000*total_y[33]/(4*np.pi*rp**2),'k',label='Volanic loss') #radius check
        pylab.loglog(total_time,1000*total_y[30]/(4*np.pi*rp**2),'r',label='Mantle heat prod')
        pylab.loglog(total_time,1000*total_y[32]/(4*np.pi*rp**2)+1000*total_y[33]/(4*np.pi*rp**2),'b--',label='Total loss (mW/m2)')
        pylab.legend()


        pylab.figure()
        pylab.subplot(2,1,1)
        pylab.loglog(total_time,total_y[35],'r',label='40K Mantle') 
        pylab.loglog(total_time,total_y[36],'b',label='40Ar Mantle') 
        pylab.loglog(total_time,total_y[37],'g',label='40K lid') #radius check
        pylab.loglog(total_time,total_y[37]+total_y[35],'k--',label='total 40K') #radius check
        
        
        pylab.legend()       
        pylab.subplot(2,1,2)
        pylab.ylabel('40Ar atmo')
        pylab.loglog(total_time,total_y[38],'g')#
        pylab.loglog(total_time,total_y[38]+total_y[36],'c--',label='total 40Ar') #radius check
        pylab.legend()

    
        pylab.figure()
        pylab.subplot(3,1,1)
        pylab.semilogx(total_time,0.5*total_y[41]/total_y[1],'g',label='D/H surface')
        pylab.semilogx(total_time,0.5*total_y[42]/total_y[0],'r',label='D/H interior')
        pylab.semilogx(total_time,0.5*(total_y[41]+total_y[42])/(total_y[1]+total_y[0]),'k',label='D/H planet')
        pylab.legend()
   

        pylab.subplot(3,1,2)
        pylab.semilogx(total_time,total_y[41],'g',label='D surface')
        pylab.semilogx(total_time,total_y[42],'r',label='D interior')
        pylab.semilogx(total_time,total_y[41]+total_y[42],'k',label='total')
        pylab.legend()

        pylab.subplot(3,1,3)
        pylab.semilogx(total_time,total_y[1],'g',label='H surface')
        pylab.semilogx(total_time,total_y[0],'r',label='H interior')
        pylab.semilogx(total_time,total_y[0]+total_y[1],'k',label='total')
        pylab.legend()        

        pylab.show()
        
    output_class = Model_outputs(total_time,total_y,FH2O_array,FCO2_array,MH2O_liq,MH2O_crystal,MCO2_liq,Pressre_H2O,CO2_Pressure_array,fO2_array,Mass_O_atm,Mass_O_dissolved,water_frac,Ocean_depth,Max_depth,Ocean_fraction)
    return output_class
