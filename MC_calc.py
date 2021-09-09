##################### 
# load modules
import time
#model_run_time = time.time()
#time.sleep(1400)
import numpy as np
import pylab
from joblib import Parallel, delayed
from all_classes import * 
from Main_code_callable import forward_model
import sys
import os
import shutil
import contextlib
####################

num_runs = 720 # Number of forward model runs
num_cores = 60 # For parallelization, check number of cores with multiprocessing.cpu_count()
os.mkdir('switch_garbage3')

#Choose planet
#which_planet = "E" # Earth (Earth model is untested for this version of the code)
which_planet = "V" # Venus

if which_planet=="E":
    Earth_inputs = Switch_Inputs(print_switch = "n", speedup_flag = "n", start_speed=15e6 , fin_speed=100e6,heating_switch = 0,C_cycle_switch="y",Start_time=10e6)   
    Earth_Numerics = Numerics(total_steps = 3 ,step0 = 50.0, step1=10000.0 , step2=1e6, step3=1e5, step4=-999, tfin0=Earth_inputs.Start_time+10000, tfin1=Earth_inputs.Start_time+10e6, tfin2=4.4e9, tfin3=4.5e9, tfin4 = -999) #standard model runs

    ## PARAMETER RANGES ##
    #initial volatile inventories
    init_water = 10**np.random.uniform(20,22,num_runs) 
    init_CO2 = 10**np.random.uniform(20,22,num_runs) 
    init_O = np.random.uniform(2e21,6e21,num_runs)

    #Weathering and ocean chemistry parameters
    Tefold = np.random.uniform(5,30,num_runs)
    alphaexp = np.random.uniform(0.1,0.5,num_runs)
    suplim_ar = 10**np.random.uniform(5,7,num_runs)
    ocean_Ca_ar = 10**np.random.uniform(-4,np.log10(0.3),num_runs)
    ocean_Omega_ar = np.random.uniform(1.0,10.0,num_runs) 

    #impact parameters (not used)
    imp_coef = 10**np.random.uniform(11,14.5,num_runs)
    tdc = np.random.uniform(0.06,0.14,num_runs)

    #escape parameters
    mult_ar = 10**np.random.uniform(-2,2,num_runs)
    mix_epsilon_ar = np.random.uniform(0.0,1.0,num_runs)
    Epsilon_ar = np.random.uniform(0.01,0.3,num_runs)
    Tstrat_array = np.random.uniform(180.5,219.5,num_runs)

    #Albedo parameters
    Albedo_C_range = np.random.uniform(0.25,0.35,num_runs)
    Albedo_H_range = np.random.uniform(0.20,0.30,num_runs)
    for k in range(0,len(Albedo_C_range)):
        if Albedo_C_range[k] < Albedo_H_range[k]:
            Albedo_H_range[k] = Albedo_C_range[k]-1e-5 

    #Stellar evolution parameters
    Omega_sun_ar = 10**np.random.uniform(np.log10(1.8),np.log10(45),num_runs)
    tsat_sun_ar = (2.9*Omega_sun_ar**1.14)/1000
    fsat_sun = 10**(-3.13)
    beta_sun_ar = 1.0/(0.35*np.log10(Omega_sun_ar) - 0.98)
    beta_sun_ar = 0.86*beta_sun_ar 

    # Interior parameters
    offset_range = 10**np.random.uniform(1.0,3.0,num_runs)
    heatscale_ar = np.random.uniform(0.5,1.5,num_runs)
    K_over_U_ar = np.random.uniform(7000.0,15200,num_runs)
    Stag_trans_ar = -5+0*np.random.uniform(50e6,4e9,num_runs)

    #Oxidation parameters
    MFrac_hydrated_ar = 10**np.random.uniform(np.log10(0.001),np.log10(0.03),num_runs) 
    dry_ox_frac_ac = 10**np.random.uniform(-4,-1,num_runs)
    wet_oxid_eff_ar = 10**np.random.uniform(-3,-1,num_runs)
    Mantle_H2O_max_ar = 10**np.random.uniform(np.log10(0.5),np.log10(15.0),num_runs) 
    surface_magma_frac_array = 10**np.random.uniform(-4,0,num_runs)

elif which_planet=="V":
    Venus_inputs = Switch_Inputs(print_switch = "n", speedup_flag = "n", start_speed=15e6 , fin_speed=100e6,heating_switch = 0,C_cycle_switch="y",Start_time=30e6)   
    Venus_Numerics = Numerics(total_steps = 2 ,step0 = 50.0, step1=10000.0 , step2=1e6, step3=-999, step4=-999, tfin0=Venus_inputs.Start_time+10000, tfin1=Venus_inputs.Start_time+30e6, tfin2=4.5e9, tfin3=-999, tfin4 = -999)

    ## PARAMETER RANGES ##
    #initial volatile inventories
    init_water = 10**np.random.uniform(20,22,num_runs)
    init_water = 10**np.random.uniform(19.5,21.5,num_runs)  
    init_CO2 = 10**np.random.uniform(20.4,21.5,num_runs)
    init_O = np.random.uniform(2e21,6e21,num_runs)
    ##init_O = np.random.uniform(0.5e21,1.5e21,num_runs) #reducing mantle sensitivity test
    #init_O = np.random.uniform(0.3e21,1.5e21,num_runs) #extra reducing mantle sensitivity test

    #Weathering and ocean chemistry parameters
    Tefold = np.random.uniform(5,30,num_runs)
    alphaexp = np.random.uniform(0.1,0.5,num_runs)
    suplim_ar = 10**np.random.uniform(5,7,num_runs)
    ocean_Ca_ar = 10**np.random.uniform(-4,np.log10(0.3),num_runs)
    ocean_Omega_ar = np.random.uniform(1.0,10.0,num_runs) 

    #impact parameters (not used)
    imp_coef = 10**np.random.uniform(11,14.5,num_runs)
    tdc = np.random.uniform(0.06,0.14,num_runs)

    #escape parameters
    mult_ar = 10**np.random.uniform(-2,2,num_runs)
    mix_epsilon_ar = np.random.uniform(0.0,1.0,num_runs)
    Epsilon_ar = np.random.uniform(0.01,0.3,num_runs)
    Tstrat_array = np.random.uniform(180.5,219.5,num_runs)

    #Albedo parameters
    Albedo_C_range = np.random.uniform(0.2,0.7,num_runs)
    Albedo_H_range = np.random.uniform(0.0001,0.3,num_runs)
    for k in range(0,len(Albedo_C_range)):
        if Albedo_C_range[k] < Albedo_H_range[k]:
            Albedo_H_range[k] = Albedo_C_range[k]-1e-5   

    #Stellar evolution parameters
    Omega_sun_ar = 10**np.random.uniform(np.log10(1.8),np.log10(45),num_runs)
    tsat_sun_ar = (2.9*Omega_sun_ar**1.14)/1000
    fsat_sun = 10**(-3.13)
    beta_sun_ar = 1.0/(0.35*np.log10(Omega_sun_ar) - 0.98)
    beta_sun_ar = 0.86*beta_sun_ar 

    # Interior parameters
    offset_range = 10**np.random.uniform(1.0,3.0,num_runs)
    heatscale_ar = np.random.uniform(0.5,2.0,num_runs)
    K_over_U_ar = np.random.uniform(6000.0,8440,num_runs) 
    Stag_trans_ar = np.random.uniform(50e6,4e9,num_runs)

    #Oxidation parameters
    MFrac_hydrated_ar = 10**np.random.uniform(np.log10(0.001),np.log10(0.03),num_runs) 
    dry_ox_frac_ac = 10**np.random.uniform(-4,-1,num_runs)
    wet_oxid_eff_ar = 10**np.random.uniform(-3,-1,num_runs)
    Mantle_H2O_max_ar = 10**np.random.uniform(np.log10(0.5),np.log10(15.0),num_runs) 
    surface_magma_frac_array = 10**np.random.uniform(-4,0,num_runs)  

##Output arrays and parameter inputs to be filled:
output = []
inputs = range(0,len(init_water))

for zzz in inputs:
    ii = zzz
    
    if which_planet=="E":
        Earth_Planet_inputs = Planet_inputs(RE = 1.0, ME = 1.0, rc=3.4e6, pm=4000.0, Total_Fe_mol_fraction = 0.06, Planet_sep=1.0, albedoC=Albedo_C_range[ii], albedoH=Albedo_H_range[ii])   
        Earth_Init_conditions = Init_conditions(Init_solid_H2O=0.0, Init_fluid_H2O=init_water[ii] , Init_solid_O=0.0, Init_fluid_O=init_O[ii],Init_solid_FeO1_5 = 0.0, Init_solid_FeO=0.0, Init_solid_CO2=0.0, Init_fluid_CO2 = init_CO2[ii])   
        Sun_Stellar_inputs = Stellar_inputs(tsat_XUV=tsat_sun_ar[ii], Stellar_Mass=1.0, fsat=fsat_sun, beta0=beta_sun_ar[ii], epsilon=Epsilon_ar[ii] )
        MC_inputs_ar =  MC_inputs(esc_a=imp_coef[ii], esc_b=tdc[ii], esc_c = mult_ar[ii],esc_d = mix_epsilon_ar[ii],ccycle_a=Tefold[ii] , ccycle_b=alphaexp[ii], supp_lim =suplim_ar[ii], interiora =offset_range[ii], interiorb=MFrac_hydrated_ar[ii],interiorc=dry_ox_frac_ac[ii],interiord = wet_oxid_eff_ar[ii],interiore = heatscale_ar[ii], interiorf = Mantle_H2O_max_ar[ii],interiorg = Stag_trans_ar[ii],ocean_a=ocean_Ca_ar[ii],ocean_b=ocean_Omega_ar[ii],K_over_U = K_over_U_ar[ii])
        inputs_for_later = [Earth_inputs,Earth_Planet_inputs,Earth_Init_conditions,Earth_Numerics,Sun_Stellar_inputs,MC_inputs_ar]

    elif which_planet=="V":    
        Venus_Planet_inputs = Planet_inputs(RE = 0.9499, ME = 0.8149, rc=3.11e6, pm=3500.0, Total_Fe_mol_fraction = 0.06, Planet_sep=0.7,albedoC=Albedo_C_range[ii], albedoH=Albedo_H_range[ii])   
        Venus_Init_conditions = Init_conditions(Init_solid_H2O=0.0, Init_fluid_H2O=init_water[ii] , Init_solid_O=0.0, Init_fluid_O=init_O[ii], Init_solid_FeO1_5 = 0.0, Init_solid_FeO=0.0, Init_solid_CO2=0.0, Init_fluid_CO2 = init_CO2[ii])   
        Sun_Stellar_inputs = Stellar_inputs(tsat_XUV=tsat_sun_ar[ii], Stellar_Mass=1.0, fsat=fsat_sun, beta0=beta_sun_ar[ii], epsilon=Epsilon_ar[ii] )
        MC_inputs_ar = MC_inputs(esc_a=imp_coef[ii], esc_b=tdc[ii],  esc_c = mult_ar[ii], esc_d = mix_epsilon_ar[ii],ccycle_a=Tefold[ii] , ccycle_b=alphaexp[ii],  supp_lim = suplim_ar[ii], interiora =offset_range[ii], interiorb=MFrac_hydrated_ar[ii],interiorc=dry_ox_frac_ac[ii],interiord = wet_oxid_eff_ar[ii],interiore = heatscale_ar[ii], interiorf = Mantle_H2O_max_ar[ii], interiorg = Stag_trans_ar[ii], ocean_a=ocean_Ca_ar[ii],ocean_b=ocean_Omega_ar[ii],K_over_U = K_over_U_ar[ii],Tstrat=Tstrat_array[ii],surface_magma_frac=surface_magma_frac_array[ii])
        inputs_for_later = [Venus_inputs,Venus_Planet_inputs,Venus_Init_conditions,Venus_Numerics,Sun_Stellar_inputs,MC_inputs_ar]
    
    sve_name = 'switch_garbage3/inputs4L%d' %ii
    np.save(sve_name,inputs_for_later)


def processInput(i):
    load_name = 'switch_garbage3/inputs4L%d.npy' %i
    try:
        if which_planet=="E": 
            print ('starting ',i)
            max_time_attempt = 1.5
            [Earth_inputs,Earth_Planet_inputs,Earth_Init_conditions,Earth_Numerics,Sun_Stellar_inputs,MC_inputs_ar] = np.load(load_name,allow_pickle=True)
            outs = forward_model(Earth_inputs,Earth_Planet_inputs,Earth_Init_conditions,Earth_Numerics,Sun_Stellar_inputs,MC_inputs_ar,max_time_attempt)
            
        elif which_planet =="V":  
            print ('starting ',i)
            max_time_attempt = 1.5
            [Venus_inputs,Venus_Planet_inputs,Venus_Init_conditions,Venus_Numerics,Sun_Stellar_inputs,MC_inputs_ar] = np.load(load_name,allow_pickle=True)
            outs = forward_model(Venus_inputs,Venus_Planet_inputs,Venus_Init_conditions,Venus_Numerics,Sun_Stellar_inputs,MC_inputs_ar,max_time_attempt) 
            
    
    except:
        print ('try again here')  # try again with slightly different numerical options
        try:
            [Venus_inputs,Venus_Planet_inputs,Venus_Init_conditions,Venus_Numerics,Sun_Stellar_inputs,MC_inputs_ar] = np.load(load_name,allow_pickle=True)
            Venus_Numerics.total_steps = 7
            max_time_attempt = 0.7
            outs = forward_model(Venus_inputs,Venus_Planet_inputs,Venus_Init_conditions,Venus_Numerics,Sun_Stellar_inputs,MC_inputs_ar,max_time_attempt)   
            if outs.total_time[-1] < 4e9:
                print ("Not enough time!")
                raise Exception      
        except:
            print ('Third attempt')  # try again with slightly different numerical options
            try:
                [Venus_inputs,Venus_Planet_inputs,Venus_Init_conditions,Venus_Numerics,Sun_Stellar_inputs,MC_inputs_ar] = np.load(load_name,allow_pickle=True)
                Venus_Numerics.total_steps = 8
                max_time_attempt = 0.2
                Venus_Init_conditions.Init_fluid_H2O = np.random.uniform(0.98,1.02)*Venus_Init_conditions.Init_fluid_H2O
                Venus_Init_conditions.Init_fluid_CO2 = np.random.uniform(0.98,1.02)*Venus_Init_conditions.Init_fluid_CO2
                outs = forward_model(Venus_inputs,Venus_Planet_inputs,Venus_Init_conditions,Venus_Numerics,Sun_Stellar_inputs,MC_inputs_ar,max_time_attempt)  
                if outs.total_time[-1] < 4e9:
                    print ("Not enough time!")
                    raise Exception           
            except:
                print ('Fourth attempt')  # try again with slightly different numerical options
                try:
                    [Venus_inputs,Venus_Planet_inputs,Venus_Init_conditions,Venus_Numerics,Sun_Stellar_inputs,MC_inputs_ar] = np.load(load_name,allow_pickle=True)
                    max_time_attempt = 0.1
                    Venus_Numerics.total_steps = 9
                    Venus_Init_conditions.Init_fluid_H2O = np.random.uniform(0.98,1.02)*Venus_Init_conditions.Init_fluid_H2O
                    Venus_Init_conditions.Init_fluid_CO2 = np.random.uniform(0.98,1.02)*Venus_Init_conditions.Init_fluid_CO2
                    outs = forward_model(Venus_inputs,Venus_Planet_inputs,Venus_Init_conditions,Venus_Numerics,Sun_Stellar_inputs,MC_inputs_ar,max_time_attempt)     
                    if outs.total_time[-1] < 4e9:
                        raise Exception      
                except:
                    print()
                    print()
                    print ('didint work ',i)
                    outs = []
                    fail_name = 'failed_outputs3/%d' %i
                    np.save(fail_name,[Venus_inputs,Venus_Planet_inputs,Venus_Init_conditions,Venus_Numerics,Sun_Stellar_inputs,MC_inputs_ar])
       

    print ('done with ',i)
    return outs

Everything = Parallel(n_jobs=num_cores)(delayed(processInput)(i) for i in inputs) #Run parallelized code
input_mega=[] # Collect input parameters for saving
for kj in range(0,len(inputs)):
    print ('saving garbage',kj)
    load_name = 'switch_garbage3/inputs4L%d.npy' %kj
    input_mega.append(np.load(load_name,allow_pickle=True))

np.save('Venus_ouputs_revisions',Everything) 
np.save('Venus_inputs_revisions',input_mega) 

shutil.rmtree('switch_garbage3')

