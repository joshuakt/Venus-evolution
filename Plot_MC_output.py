import numpy as np
import pylab
from scipy.integrate import *
from scipy.interpolate import interp1d
from scipy import optimize
import scipy.optimize 
from all_classes import *
import time
from other_functions import *
import pdb

new_inputs = [] #actually ouputs
inputs_for_MC=[]
Total_Fe_array = []

global g,rp,mp

def use_one_output(inputs,MCinputs):
    k= 0
    global g,rp,mp
    while k<len(inputs):
        if (np.size(inputs[k])==1)and (inputs[k].total_time[-1] > 4.4e9):        
            top = (inputs[k].total_y[13][0] + inputs[k].total_y[12][0]) 
            bottom =  (inputs[k].total_y[1][0] + inputs[k].total_y[0][0] )
            CO2_H2O_ratio = top/bottom

            ## Choose which outputs to plot:

            if (2>1): #All model outputs

            #if (inputs[k].total_y[4][-1] <1e15)and(np.max(inputs[k].Ocean_depth[-1])==0)and(np.max(inputs[k].Ocean_depth[:])>0)and (inputs[k].total_y[1][-1] <2e16)and(inputs[k].total_y[12][-1]>2e20)and(inputs[k].total_y[12][-1]<1e21): # Recover modern Venus, transient habitable time  (reproduces Fig. 3 in main text)

            #if (inputs[k].total_y[4][-1] <1e15)and(np.max(inputs[k].Ocean_depth[:])==0)and (inputs[k].total_y[1][-1] <2e16)and(inputs[k].total_y[12][-1]>2e20)and(inputs[k].total_y[12][-1]<1e21): # Recover modern Venus, never habitable (reproduces Fig. 2 in main text)

            #if (inputs[k].total_y[4][-1] <1e15)and(np.max(inputs[k].Ocean_depth[-1])==0)and(inputs[k].total_y[1][-1] <2e16)and(inputs[k].total_y[12][-1]>2e20): #Recover modern Venus


                rp = 6371000*MCinputs[k][1].RE
                mp = 5.972e24*MCinputs[k][1].ME
                g = 6.67e-11*mp/(rp**2)
                Pbase = (inputs[k].Pressre_H2O[-1] + inputs[k].CO2_Pressure_array[-1] + inputs[k].total_y[22][-1])
                Mvolatiles = (Pbase*4*np.pi*rp**2)/g 

                new_inputs.append(inputs[k])
                inputs_for_MC.append(MCinputs[k])
                Total_Fe_array.append(MCinputs[k][1].Total_Fe_mol_fraction)
                rp = 6371000*MCinputs[k][1].RE
                mp = 5.972e24*MCinputs[k][1].ME
                g = 6.67e-11*mp/(rp**2)

        k= k+1 

N2_Pressure = 1e5 # Do not change

### Load outputs and inputs. Note it is possible to load multiple output files and process them all at once
inputs = np.load('Venus_ouputs_revisions.npy',allow_pickle = True) 
MCinputs = np.load('Venus_inputs_revisions.npy',allow_pickle = True)
use_one_output(inputs,MCinputs)
#inputs = np.load('Venus_ouputs_revisions3.npy',allow_pickle = True) 
#MCinputs = np.load('Venus_inputs_revisions3.npy',allow_pickle = True)
#use_one_output(inputs,MCinputs)

#Pause here to check number of successful model runs etc. Type 'c' to continue.
pdb.set_trace()
inputs = np.array(new_inputs)

def interpolate_class(saved_outputs):
    outs=[]
    for i in range(0,len(saved_outputs)):

        time_starts = np.min([np.where(saved_outputs[i].total_time>1)])-1
        time = saved_outputs[i].total_time[time_starts:]
        num_t_elements = 1000 
        new_time = np.logspace(np.log10(np.min(time[:])),np.max([np.log10(time[:-1])]),num_t_elements)
        num_y = 56
        new_total_y = np.zeros(shape=(num_y,len(new_time)))
        for k in range(0,num_y):
            f1 = interp1d(time,saved_outputs[i].total_y[k][time_starts:])
            new_total_y[k] = f1(new_time)
 
        f1 = interp1d(time,saved_outputs[i].FH2O_array[time_starts:])
        new_FH2O_array = f1(new_time)

        f1 = interp1d(time,saved_outputs[i].FCO2_array[time_starts:])
        new_FCO2_array = f1(new_time)
        
        f1 = interp1d(time,saved_outputs[i].MH2O_liq[time_starts:])
        new_MH2O_liq = f1(new_time)    

        f1 = interp1d(time,saved_outputs[i].MH2O_crystal[time_starts:])
        new_MH2O_crystal = f1(new_time)    
  
        f1 = interp1d(time,saved_outputs[i].MCO2_liq[time_starts:])
        new_MCO2_liq = f1(new_time)    

        f1 = interp1d(time,saved_outputs[i].Pressre_H2O[time_starts:])
        new_Pressre_H2O = f1(new_time)    

        f1 = interp1d(time,saved_outputs[i].CO2_Pressure_array[time_starts:])
        new_CO2_Pressure_array = f1(new_time)  

        f1 = interp1d(time,saved_outputs[i].fO2_array[time_starts:])
        new_fO2_array = f1(new_time)  

        f1 = interp1d(time,saved_outputs[i].Mass_O_atm[time_starts:])
        new_Mass_O_atm = f1(new_time)  

        f1 = interp1d(time,saved_outputs[i].Mass_O_atm[time_starts:])
        new_Mass_O_atm = f1(new_time)  

        f1 = interp1d(time,saved_outputs[i].Mass_O_dissolved[time_starts:])
        new_Mass_O_dissolved = f1(new_time) 

        f1 = interp1d(time,saved_outputs[i].water_frac[time_starts:])
        new_water_frac = f1(new_time) 

        f1 = interp1d(time,saved_outputs[i].Ocean_depth[time_starts:])
        new_Ocean_depth = f1(new_time) 

        f1 = interp1d(time,saved_outputs[i].Max_depth[time_starts:])
        new_Max_depth = f1(new_time) 

        f1 = interp1d(time,saved_outputs[i].Ocean_fraction[time_starts:])
        new_Ocean_fraction = f1(new_time) 

        output_class = Model_outputs(new_time,new_total_y,new_FH2O_array,new_FCO2_array,new_MH2O_liq,new_MH2O_crystal,new_MCO2_liq,new_Pressre_H2O,new_CO2_Pressure_array,new_fO2_array,new_Mass_O_atm,new_Mass_O_dissolved,new_water_frac,new_Ocean_depth,new_Max_depth,new_Ocean_fraction)
        outs.append(output_class)

    return outs



interp_outputs = interpolate_class(inputs)
inputs = interp_outputs 
 
################################################################################
## Post-processing of outputs:
# Oxygen fugacity and mantle redox functions (for post-processing)
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

def get_fO2(XFe2O3_over_XFeO,P,T,Total_Fe): ## Total_Fe is a mole fraction of iron minerals XFeO + XFeO1.5 = Total_Fe, and XFe2O3 = 0.5*XFeO1.5, xo XFeO + 2XFe2O3 = Total_Fe
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

total_time = []
total_y = []
FH2O_array=  []
FCO2_array= []
MH2O_liq =  []
MCO2_liq =  []
Pressre_H2O = []
CO2_Pressure_array =  []
fO2_array =  []
Mass_O_atm =  []
Mass_O_dissolved =  []
water_frac =  []
Ocean_depth =  []
Max_depth = []
Ocean_fraction =  []

for k in range(0,len(inputs)):
    total_time.append( inputs[k].total_time )
    total_y.append(inputs[k].total_y)
    FH2O_array.append(inputs[k].FH2O_array )
    FCO2_array.append(inputs[k].FCO2_array )
    MH2O_liq.append(inputs[k].MH2O_liq )
    MCO2_liq.append(inputs[k].MCO2_liq )
    Pressre_H2O.append(inputs[k].Pressre_H2O) 
    CO2_Pressure_array.append(inputs[k].CO2_Pressure_array )
    fO2_array.append(inputs[k].fO2_array )
    Mass_O_atm.append(inputs[k].Mass_O_atm )
    Mass_O_dissolved.append(inputs[k].Mass_O_dissolved )
    water_frac.append(inputs[k].water_frac )
    Ocean_depth.append(inputs[k].Ocean_depth )
    Max_depth.append(inputs[k].Max_depth )
    Ocean_fraction.append(inputs[k].Ocean_fraction )

[int1,int2,int3] = [2.5,50,97.5]

import scipy.stats
confidence_y=scipy.stats.scoreatpercentile(total_y ,[int1,int2,int3], interpolation_method='fraction',axis=0)
confidence_FH2O = scipy.stats.scoreatpercentile(FH2O_array ,[int1,int2,int3], interpolation_method='fraction',axis=0)
confidence_FCO2 = scipy.stats.scoreatpercentile(FCO2_array ,[int1,int2,int3], interpolation_method='fraction',axis=0)
confidence_MH2O_liq = scipy.stats.scoreatpercentile(MH2O_liq ,[int1,int2,int3], interpolation_method='fraction',axis=0)
confidence_MCO2_liq = scipy.stats.scoreatpercentile(MCO2_liq ,[int1,int2,int3], interpolation_method='fraction',axis=0)
confidence_Pressre_H2O = scipy.stats.scoreatpercentile(Pressre_H2O ,[int1,int2,int3], interpolation_method='fraction',axis=0)
confidence_CO2_Pressure_array = scipy.stats.scoreatpercentile(CO2_Pressure_array ,[int1,int2,int3], interpolation_method='fraction',axis=0)
confidence_fO2_array = scipy.stats.scoreatpercentile(fO2_array ,[int1,int2,int3], interpolation_method='fraction',axis=0)
confidence_Mass_O_atm = scipy.stats.scoreatpercentile(Mass_O_atm ,[int1,int2,int3], interpolation_method='fraction',axis=0)
confidence_Mass_O_dissolved = scipy.stats.scoreatpercentile(Mass_O_dissolved ,[int1,int2,int3], interpolation_method='fraction',axis=0)
confidence_water_frac = scipy.stats.scoreatpercentile(water_frac ,[int1,int2,int3], interpolation_method='fraction',axis=0)
confidence_Ocean_depth = scipy.stats.scoreatpercentile(Ocean_depth ,[5,50,95], interpolation_method='fraction',axis=0)
confidence_Max_depth = scipy.stats.scoreatpercentile(Max_depth ,[int1,int2,int3], interpolation_method='fraction',axis=0)
confidence_Ocean_fraction = scipy.stats.scoreatpercentile(Ocean_fraction ,[int1,int2,int3], interpolation_method='fraction',axis=0)

f_O2_FMQ = []
f_O2_IW = []
f_O2_MH = []
f_O2_mantle = []
iron_ratio = []
total_iron = []
iron2_array = []
iron3_array = []
actual_phi_surf_melt_ar = []
XH2O_melt = []
XCO2_melt = []
Max_runaway_temperatures = []

rc = MCinputs[0][1].rc
mantle_mass = 0.0

x_low = 1.0 #Start time for plotting (years)
x_high =np.max(total_time[0])  #Finish time for plotting (years)

rp = 6371000*MCinputs[0][1].RE
ll = rp - rc
alpha = 2e-5
k = 4.2 
kappa = 1e-6
Racr = 1.1e3

Melt_volume = np.copy(total_time)
Plate_velocity = np.copy(total_time)
total_Ar40  = np.copy(total_time) 
total_K40  = np.copy(total_time)
HTmin = []
HTmax = []
HT_duration = []

DH_atmo =  np.copy(total_time)
DH_solid =  np.copy(total_time)

F_H2O_new = []
F_CO2_new = []
F_CO_new = []
F_H2_new = []
F_CH4_new = []
F_SO2_new = []
F_H2S_new = []
F_S2_new = []
O2_consumption_new = []
Late_melt_production = []

mantle_CO2_fraction = []
mantle_H2O_fraction=[]
from outgassing_module import *

for k in range(0,len(inputs)):
    f_O2_FMQ.append(inputs[k].Ocean_fraction*0 )
    f_O2_IW.append(inputs[k].Ocean_fraction*0 )
    f_O2_MH.append(inputs[k].Ocean_fraction*0 )
    f_O2_mantle.append(inputs[k].Ocean_fraction*0 )
    iron_ratio.append(inputs[k].Ocean_fraction*0 )
    total_iron.append(inputs[k].Ocean_fraction*0 )
    iron2_array.append(inputs[k].Ocean_fraction*0 )
    iron3_array.append(inputs[k].Ocean_fraction*0 )
    actual_phi_surf_melt_ar.append(inputs[k].Ocean_fraction*0 )
    XH2O_melt.append(inputs[k].Ocean_fraction*0 )
    XCO2_melt.append(inputs[k].Ocean_fraction*0 )
    F_H2O_new.append(inputs[k].Ocean_fraction*0 )
    F_CO2_new.append(inputs[k].Ocean_fraction*0 )
    F_CO_new.append(inputs[k].Ocean_fraction*0 )
    F_H2_new.append(inputs[k].Ocean_fraction*0 )
    F_CH4_new.append(inputs[k].Ocean_fraction*0 )
    F_SO2_new.append(inputs[k].Ocean_fraction*0 )
    F_H2S_new.append(inputs[k].Ocean_fraction*0 )
    F_S2_new.append(inputs[k].Ocean_fraction*0 )
    O2_consumption_new.append(inputs[k].Ocean_fraction*0 )

    mantle_CO2_fraction.append(inputs[k].Ocean_fraction*0 ) 
    mantle_H2O_fraction.append(inputs[k].Ocean_fraction*0 )

    try:
        ocean_start_index = np.min(np.where(inputs[k].Ocean_depth>0))
        max_T_runaway = np.max(total_y[k][8][ocean_start_index:])
        Max_runaway_temperatures.append(max_T_runaway)
    except:
        abc = 1 + 2

    for i in range(0,len(total_time[k])):
        mantle_CO2_fraction[k][i] = total_y[k][13][i]/(total_y[k][13][i]+total_y[k][12][i])
        mantle_H2O_fraction[k][i] = total_y[k][0][i]/(total_y[k][0][i]+total_y[k][1][i])
        Pressure_surface =fO2_array[k][i] + Pressre_H2O[k][i]*water_frac[k][i] + CO2_Pressure_array[k][i] + N2_Pressure  
        
        f_O2_FMQ[k][i] = buffer_fO2(total_y[k][7][i],Pressure_surface/1e5,'FMQ')
        f_O2_IW[k][i] = buffer_fO2(total_y[k][7][i],Pressure_surface/1e5,'IW')
        f_O2_MH[k][i] =  buffer_fO2(total_y[k][7][i],Pressure_surface/1e5,'MH')
        iron3 = total_y[k][5][i]*56/(56.0+1.5*16.0)
        iron2 = total_y[k][6][i]*56/(56.0+16.0)
        iron2_array[k][i] = iron2
        iron3_array[k][i] = iron3
        total_iron[k][i] = iron3+iron2
        iron_ratio[k][i] = iron3/iron2
        f_O2_mantle[k][i] = get_fO2(0.5*iron3/iron2,Pressure_surface,total_y[k][7][i],Total_Fe_array[k])
        T_for_melting = float(total_y[k][7][i])
        Poverburd = fO2_array[k][i] + Pressre_H2O[k][i] + CO2_Pressure_array[k][i] + N2_Pressure  
        alpha = 2e-5
        cp = 1.2e3 
        pm = 4000.0
        rdck = optimize.minimize(find_r,x0=float(total_y[k][2][i]),args = (T_for_melting,alpha,g,cp,pm,rp,float(Poverburd),0,0.0))
        rad_check = float(rdck.x[0])
        if rad_check>rp:
            rad_check = rp
        rlid = rp - total_y[k][26][i]
        [actual_phi_surf_melt,actual_visc,Va] = temp_meltfrac(0.99998*rad_check,rp,alpha,pm,T_for_melting,cp,g,Poverburd,0,rlid)
        actual_phi_surf_melt_ar[k][i]= actual_phi_surf_melt
        F = actual_phi_surf_melt_ar[k][i]
        x = 0.01550152865954013
        M_H2O = 18.01528
        M_CO2 = 44.01
        mantle_mass = (4./3. * np.pi * pm * (rp**3 - rc**3))
        XH2O_melt_max = x*M_H2O*0.499 # half of mol fraction allowed to be H2O
        XCO2_melt_max = x*M_CO2*0.499 # half of mol fraction allowed to be CO2
        if F >0:
            XH2O_melt[k][i] = np.min([0.99*XH2O_melt_max,(1- (1-F)**(1/0.01)) * (total_y[k][0][i]/mantle_mass)/F ]) # mass frac, ensures mass frac never implies all moles volatile!
            XCO2_melt[k][i] =  np.min([0.99*XCO2_melt_max,(1- (1-F)**(1/2e-3)) * (total_y[k][13][i]/mantle_mass)/F ])# mass frac
        else:
            XH2O_melt[k][i] = 0.0 
            XCO2_melt[k][i] =  0.0
  
    Late_melt_production.append(np.mean(total_y[k][25][994:]))
    

    Q =  total_y[k][11]
    Aoc = 4*np.pi*rp**2
    for i in range(0,len(Q)):
        if total_y[k][16][i]/1000 < 1e-11:
            total_y[k][16][i] = 0.0
                
    Melt_volume[k] = total_y[k][25]  
    Plate_velocity[k] = 365*24*60*60*Melt_volume[k]/(total_y[k][16]  * 3 * np.pi*rp)

    min_HabTime = 0
    hab_counter = 0
    max_HabTime = 0
    for i in range(0,len(Q)):
        if total_y[k][16][i]/1000 < 1e-11:
            Plate_velocity[k][i] = 0  
        total_Ar40[k][i] = total_y[k][38][i]/(total_y[k][38][i] + total_y[k][36][i])
        total_K40[k][i] = total_y[k][35][i] + total_y[k][37][i]

        DH_atmo[k][i] = 0.5*total_y[k][41][i]/total_y[k][1][i]
        DH_solid[k][i] = 0.5*total_y[k][42][i]/total_y[k][0][i]
        if (inputs[k].Ocean_depth[i]>0.0):
            if hab_counter == 0:
                min_HabTime = total_time[k][i]
                hab_counter = 1.0
            max_HabTime = total_time[k][i]

        #### outgassing aside 
        Pressure_surface =fO2_array[k][i] + Pressre_H2O[k][i]*water_frac[k][i] + CO2_Pressure_array[k][i] + N2_Pressure  
        melt_mass = total_y[k][25][i]*pm*1000 
        Tsolidus = sol_liq(rp,g,pm,rp,0.0,0.0)
        if (0.5*iron_ratio[k][i]>0)and(melt_mass>0)and(actual_phi_surf_melt_ar[k][i])>0:
            try:
                [F_H2O,F_CO2,F_H2,F_CO,F_CH4,F_SO2,F_H2S,F_S2,O2_consumption] = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
            except:
                [F_H2O,F_CO2,F_H2,F_CO,F_CH4,F_SO2,F_H2S,F_S2,O2_consumption] = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        else:
            [F_H2O,F_CO2,F_H2,F_CO,F_CH4,F_SO2,F_H2S,F_S2,O2_consumption] = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    
        F_H2O_new[k][i] = F_H2O*365*24*60*60/(1e12)
        F_CO2_new[k][i] = F_CO2*365*24*60*60/(1e12)
        F_CO_new[k][i] = F_CO*365*24*60*60/(1e12)
        F_H2_new[k][i] = F_H2*365*24*60*60/(1e12)
        F_CH4_new[k][i] = F_CH4*365*24*60*60/(1e12)
        F_SO2_new[k][i] = F_SO2*365*24*60*60/(1e12)
        F_H2S_new[k][i] = F_H2S*365*24*60*60/(1e12)
        F_S2_new[k][i] = F_S2*365*24*60*60/(1e12)
        O2_consumption_new[k][i] = O2_consumption*365*24*60*60/(1e12)
    
    HTmin.append(min_HabTime)
    HTmax.append(max_HabTime)
    HT_duration.append(  max_HabTime - min_HabTime)
  

HTmin= np.array(HTmin)
HTmax = np.array(HTmax)
HT_duration = np.array(HT_duration)
Late_melt_production = np.array(Late_melt_production)
## End post-processing of outputs:
################################################################################

################################################################################
## Plotting of results:
conf_DH_atmo = scipy.stats.scoreatpercentile(DH_atmo ,[int1,int2,int3], interpolation_method='fraction',axis=0)
conf_DH_solid = scipy.stats.scoreatpercentile(DH_solid ,[int1,int2,int3], interpolation_method='fraction',axis=0)

conf_HT_duration = scipy.stats.scoreatpercentile(HT_duration ,[int1,int2,int3], interpolation_method='fraction',axis=0)
conf_HTmax = scipy.stats.scoreatpercentile(HTmax ,[int1,int2,int3], interpolation_method='fraction',axis=0)
conf_HTmin = scipy.stats.scoreatpercentile(HTmin ,[int1,int2,int3], interpolation_method='fraction',axis=0)

conf_Late_melt_production = scipy.stats.scoreatpercentile(365*24*60*60*Late_melt_production/1e9 ,[int1,int2,int3], interpolation_method='fraction',axis=0)

four_percentilesa = scipy.stats.percentileofscore(Max_runaway_temperatures ,700)
four_percentilesb = scipy.stats.percentileofscore(Max_runaway_temperatures ,800)
four_percentilesc = scipy.stats.percentileofscore(Max_runaway_temperatures ,900)
four_percentilesd = scipy.stats.percentileofscore(Max_runaway_temperatures,1000)

print ("Hab duration", conf_HT_duration[0],conf_HT_duration[1],conf_HT_duration[2])   
print ("Hab max", conf_HTmax[0],conf_HTmax[1],conf_HTmax[2])   
print ("Hab min", conf_HTmin[0],conf_HTmin[1],conf_HTmin[2])   
print ("Late_melt_production", conf_Late_melt_production[0],conf_Late_melt_production[1],conf_Late_melt_production[2])   
print ("four_percentiles",four_percentilesa,four_percentilesb,four_percentilesc,four_percentilesd)

try:
    confidence_Max_runaway_temperatures = scipy.stats.scoreatpercentile(Max_runaway_temperatures ,[int1,int2,int3], interpolation_method='fraction',axis=0)
    print('confidence_Max_runaway_temperatures')
    print(confidence_Max_runaway_temperatures)
except:
    print ('no confidence max runaway')

## Mantle and magma ocean redox relative to FMQ:
confidence_mantle_CO2_fraction = scipy.stats.scoreatpercentile(mantle_CO2_fraction ,[int1,int2,int3], interpolation_method='fraction',axis=0)
confidence_mantle_H2O_fraction = scipy.stats.scoreatpercentile(mantle_H2O_fraction ,[int1,int2,int3], interpolation_method='fraction',axis=0)
confidence_f_O2_FMQ = scipy.stats.scoreatpercentile(f_O2_FMQ ,[int1,int2,int3], interpolation_method='fraction',axis=0)
confidence_f_O2_IW = scipy.stats.scoreatpercentile(f_O2_IW ,[int1,int2,int3], interpolation_method='fraction',axis=0)
confidence_f_O2_MH = scipy.stats.scoreatpercentile(f_O2_MH ,[int1,int2,int3], interpolation_method='fraction',axis=0)
confidence_f_O2_mantle = scipy.stats.scoreatpercentile(f_O2_mantle ,[int1,int2,int3], interpolation_method='fraction',axis=0)
confidence_iron_ratio = scipy.stats.scoreatpercentile(iron_ratio ,[int1,int2,int3], interpolation_method='fraction',axis=0)
confidence_f_O2_relative_FMQ = scipy.stats.scoreatpercentile(np.log10(f_O2_mantle) - np.log10(f_O2_FMQ) ,[int1,int2,int3], interpolation_method='fraction',axis=0)
                     
  
Melt_volume = 365*24*60*60*Melt_volume/1e9
Melt_volumeCOPY = np.copy(Melt_volume)
confidence_melt=scipy.stats.scoreatpercentile(Melt_volume ,[int1,int2,int3], interpolation_method='fraction',axis=0)
confidence_velocity =  scipy.stats.scoreatpercentile(Plate_velocity ,[int1,int2,int3], interpolation_method='fraction',axis=0)

#### Plotting individual model runs:
pylab.figure()
for k in range(0,len(inputs)):
    pylab.semilogx(total_time[k],total_y[k][0]+total_y[k][1],'b')
    pylab.semilogx(total_time[k],total_y[k][1],'r--')
pylab.ylabel("Total water (kg)")
pylab.xlabel("Time (yrs)")
pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
pylab.xlim([x_low, x_high])
pylab.legend(frameon=False)

pylab.figure()
pylab.subplot(2,1,1)
for k in range(0,len(inputs)):
    pylab.semilogx(total_time[k],total_y[k][52],'b')
pylab.ylabel("depletion_fraction")
pylab.xlabel("Time (yrs)")
pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
pylab.xlim([x_low, x_high])
pylab.legend(frameon=False)
pylab.subplot(2,1,2)
Vmantle0 = ((4.0*np.pi/3.0) * (rp**3 - rc**3))/1e9
for k in range(0,len(inputs)):
    for i in range(0,len(Melt_volume[k])): 
        if (total_time[k][i]<inputs_for_MC[k][5].interiorg):
            Melt_volume[k][i] = 0.0
    pylab.semilogx(total_time[k],scipy.integrate.cumtrapz(Melt_volume[k],x=total_time[k],initial=0)/Vmantle0,'b')
pylab.ylabel("cumulative melt")
pylab.xlabel("Time (yrs)")
pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
pylab.xlim([x_low, x_high])

pylab.figure()
for k in range(0,len(inputs)):
    pylab.semilogx(total_time[k],total_y[k][24],'b')
pylab.ylabel("MMW")
pylab.xlabel("Time (yrs)")
pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
pylab.xlim([x_low, x_high])
pylab.legend(frameon=False)

pylab.figure(figsize=(20,10))
pylab.subplot(4,3,1)
for k in range(0,len(inputs)):
    pylab.semilogx(total_time[k],total_y[k][7],'b', label="Mantle" if k == 0 else "")
    pylab.semilogx(total_time[k],total_y[k][8],'r', label="Surface" if k == 0 else "")
pylab.ylabel("Temperature (K)")
pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
pylab.xlim([x_low, x_high])
pylab.legend(frameon=False)

pylab.subplot(4,3,2)
for k in range(0,len(inputs)):
    pylab.semilogx(total_time[k],total_y[k][2]/1000.0)
pylab.ylabel("Radius of solidification (km)")
pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
pylab.xlim([x_low, x_high])

pylab.subplot(4,3,3)
pylab.ylabel("Pressure (bar)")
for k in range(0,len(inputs)):
    pylab.loglog(total_time[k],Mass_O_atm[k]*g/(4*np.pi*(0.032/total_y[k][24])*rp**2*1e5),'g',label='O2'if k == 0 else "")
    pylab.loglog(total_time[k],water_frac[k]*Pressre_H2O[k]/1e5,'b',label='H2O'if k == 0 else "")
    pylab.loglog(total_time[k],total_y[k][23]/1e5,'r',label='CO2'if k == 0 else "")
pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
pylab.xlim([x_low, x_high])
pylab.legend(frameon=False)

pylab.subplot(4,3,4)
pylab.ylabel("Liquid water depth (km)")
for k in range(0,len(inputs)):
    pylab.semilogx(total_time[k],Max_depth[k]/1000.0,'k--',label='Max elevation land' if k == 0 else "")
    pylab.semilogx(total_time[k],Ocean_depth[k]/1000.0,'b',label='Ocean depth' if k == 0 else "")
pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
pylab.xlim([x_low, x_high])
pylab.legend(frameon=False)

pylab.subplot(4,3,5)
for k in range(0,len(inputs)):
    pylab.loglog(total_time[k],total_y[k][9] , 'b' ,label = 'OLR' if k == 0 else "")
    pylab.loglog(total_time[k],total_y[k][10] , 'r' ,label = 'ASR'  if k == 0 else "")
    pylab.loglog(total_time[k],total_y[k][11] , 'g' ,label = 'q_m'  if k == 0 else "")
    pylab.loglog(total_time[k],280+0*total_y[k][9] , 'k--' ,label = 'Runaway limit' if k == 0 else "")
pylab.ylabel("Heat flux (W/m2)")
pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
pylab.xlim([x_low, x_high])
pylab.legend(frameon=False)


pylab.subplot(4,3,6)
pylab.ylabel('Carbon fluxes (Tmol/yr)')
for k in range(0,len(inputs)):
    pylab.semilogx(total_time[k],total_y[k][14],'k' ,label = 'Weathering'  if k == 0 else "")
    pylab.semilogx(total_time[k],total_y[k][15],'r' ,label = 'Outgassing' if k == 0 else "") 
pylab.xlabel('Time (yrs)')
pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
pylab.xlim([x_low, x_high])
pylab.legend(frameon=False)

pylab.subplot(4,3,7)
pylab.ylabel('Melt production, MP (km$^3$/yr)')
for k in range(0,len(inputs)):
    pylab.loglog(total_time[k],Melt_volumeCOPY[k],'r')
pylab.xlabel('Time (yrs)')
pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
pylab.xlim([x_low, x_high])

pylab.subplot(4,3,8)
for k in range(0,len(inputs)):
    pylab.semilogx(total_time[k],np.log10(f_O2_mantle[k]) - np.log10(f_O2_FMQ[k]),'r', label = 'Mantle fO2'  if k == 0 else "")
pylab.ylabel("Mantle oxygen fugacity ($\Delta$QFM)")
pylab.xlabel("Time (yrs)")
pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
pylab.xlim([x_low, x_high])
pylab.legend(frameon=False)


pylab.subplot(4,3,9)
for k in range(0,len(inputs)):
    pylab.semilogx(total_time[k],total_y[k][18]*365*24*60*60/(0.032*1e12),'g' ,label = 'Dry crustal' if k == 0 else "")
    pylab.semilogx(total_time[k],total_y[k][19]*365*24*60*60/(0.032*1e12),'k' ,label = 'Escape' if k == 0 else "")
    pylab.semilogx(total_time[k],total_y[k][20]*365*24*60*60/(0.032*1e12),'b' ,label = 'Wet crustal' if k == 0 else "")
    pylab.semilogx(total_time[k],total_y[k][21]*365*24*60*60/(0.032*1e12),'r' ,label = 'Outgassing' if k == 0 else "")
    pylab.semilogx(total_time[k],(total_y[k][18]+total_y[k][19]+total_y[k][20]+total_y[k][21])*365*24*60*60/(0.032*1e12),'c--' ,label = 'Net' if k == 0 else "")
pylab.ylabel("O2 flux (Tmol/yr)")
pylab.xlabel("Time (yrs)")
pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
pylab.xlim([1.0, np.max(total_time)])
pylab.yscale('symlog',linthreshy = 0.01)
pylab.legend(frameon=False,loc = 3,ncol=2)

pylab.figure()
pylab.subplot(4,1,1)
pylab.ylabel("Solid mantle Fe3+/Fe2+")
for k in range(0,len(inputs)):
    pylab.semilogx(total_time[k],iron_ratio[k])#,'k')
pylab.xlabel('Time (yrs)')

pylab.subplot(4,1,2)
pylab.ylabel("total solid iron (kg)")
for k in range(0,len(inputs)):
    pylab.semilogx(total_time[k],total_iron[k])#,'k')
pylab.xlabel('Time (yrs)')

pylab.subplot(4,1,3)
pylab.ylabel("Fe3+ (kg)")
for k in range(0,len(inputs)):
    pylab.semilogx(total_time[k],iron3_array[k])#,'k')
pylab.xlabel('Time (yrs)')

pylab.subplot(4,1,4)
pylab.ylabel("Fe2+ (kg)")
for k in range(0,len(inputs)):
    pylab.semilogx(total_time[k],iron2_array[k])#,'k')
pylab.xlabel('Time (yrs)')
#### End plot individual model runs

pylab.figure()
pylab.semilogx(total_time[0],confidence_y[1][1]+confidence_y[1][0],'b', label='Tstrat')
pylab.fill_between(total_time[0],confidence_y[0][0]+confidence_y[0][1], confidence_y[2][0]+confidence_y[2][1], color='blue', alpha='0.4')  
pylab.ylabel("Total water (kg)")
pylab.xlabel("Time (yrs)")
pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
pylab.xlim([x_low, x_high])
pylab.legend(frameon=False)

pylab.figure()
pylab.ylabel("Tstrat")
pylab.xlabel("Time (yrs)")
pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
pylab.xlim([x_low, x_high])
pylab.legend(frameon=False)


#######################################################################
#######################################################################
##### 95% confidence interval plots 
pylab.figure(figsize=(20,10))
pylab.subplot(4,3,1)
Mantlelabel="Mantle, T$_p$"
pylab.semilogx(total_time[0],confidence_y[1][7],'b', label=Mantlelabel)
pylab.fill_between(total_time[0],confidence_y[0][7], confidence_y[2][7], color='blue', alpha='0.4')  
surflabel="Surface, T$_{surf}$"
pylab.semilogx(total_time[0],confidence_y[1][8],'r', label=surflabel)
pylab.fill_between(total_time[0],confidence_y[0][8], confidence_y[2][8], color='red', alpha='0.4')  
sol_val = sol_liq(rp,g,4000,rp,0.0,0.0)
sol_val2 = sol_liq(rp,g,4000,rp,3e9,0.0)
modlabel="Modern T$_{surf}$"
pylab.semilogx(total_time[0],0*confidence_y[1][8]+737,'r--', label=modlabel)
pylab.ylabel("Temperature (K)")
pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
pylab.xlim([x_low, x_high])
pylab.ylim([250, 4170])
pylab.legend(frameon=False)

pylab.subplot(4,3,2)
pylab.semilogx(total_time[0],confidence_y[1][2]/1000.0,'k')
pylab.fill_between(total_time[0],confidence_y[0][2]/1000.0, confidence_y[2][2]/1000.0, color='grey', alpha='0.4')  
pylab.ylabel("Radius of solidification, r$_s$ (km)")
pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
pylab.xlim([x_low, x_high])

pylab.subplot(4,3,3)
pylab.ylabel("Pressure (bar)")
O2_label = 'O$_2$'
pylab.loglog(total_time[0],confidence_Mass_O_atm[1]*g/(4*np.pi*(0.032/total_y[k][24])*rp**2*1e5),'g',label=O2_label)
pylab.fill_between(total_time[0],confidence_Mass_O_atm[0]*g/(4*np.pi*(0.032/total_y[k][24])*rp**2*1e5),confidence_Mass_O_atm[2]*g/(4*np.pi*(0.032/total_y[k][24])*rp**2*1e5), color='green', alpha='0.4')  
H2O_label = 'H$_2$O'
pylab.loglog(total_time[0],confidence_water_frac[1]*confidence_Pressre_H2O[1]/1e5,'b',label=H2O_label)
pylab.fill_between(total_time[0],confidence_water_frac[0]*confidence_Pressre_H2O[0]/1e5, confidence_water_frac[2]*confidence_Pressre_H2O[2]/1e5, color='blue', alpha='0.4')  
CO2_label = 'CO$_2$'
pylab.loglog(total_time[0],confidence_y[1][23]/1e5,'r',label=CO2_label)
pylab.fill_between(total_time[0],confidence_y[0][23]/1e5, confidence_y[2][23]/1e5, color='red', alpha='0.4')  
pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
pylab.xlim([x_low, x_high])
pylab.legend(frameon=False)

try:
    print ('Ocean depth maximums')
    print(np.max(confidence_Ocean_depth[0])/1000.0,np.max(confidence_Ocean_depth[1])/1000.0,np.max(confidence_Ocean_depth[2])/1000.0)
except:
    print ('no ocean')
pylab.subplot(4,3,4)
pylab.ylabel("Liquid water depth (km)")
pylab.semilogx(total_time[0],confidence_Ocean_depth[1]/1000.0,'b',label='Ocean depth')
pylab.fill_between(total_time[0],confidence_Ocean_depth[0]/1000.0, confidence_Ocean_depth[2]/1000.0, color='blue', alpha='0.4')  
pylab.yscale('symlog',linthreshy = 0.001)
pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
pylab.xlim([x_low, x_high])
pylab.ylim([-1e-4, 3.5])
pylab.legend(frameon=False)

pylab.subplot(4,3,5)
pylab.loglog(total_time[0],280+0*confidence_y[1][9] , 'k:' ,label = 'Runaway limit')
pylab.loglog(total_time[0],confidence_y[1][9] , 'b' ,label = 'OLR' )
pylab.fill_between(total_time[0],confidence_y[0][9],confidence_y[2][9], color='blue', alpha='0.4')  
pylab.loglog(total_time[0],confidence_y[1][10] , 'r' ,label = 'ASR')
pylab.fill_between(total_time[0],confidence_y[0][10],confidence_y[2][10], color='red', alpha='0.4')  
q_interior_label = 'q$_m$'
pylab.loglog(total_time[0],confidence_y[1][11] , 'g' ,label = q_interior_label)
pylab.fill_between(total_time[0],confidence_y[0][11],confidence_y[2][11], color='green', alpha='0.4')  
modlabel = 'Modern q$_m$'
pylab.semilogx(total_time[0],0*confidence_y[1][11]+0.02,'g--', label=modlabel)
pylab.ylabel("Heat flux (W/m2)")
pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
pylab.xlim([x_low, x_high])
pylab.legend(frameon=False,ncol=2)

pylab.subplot(4,3,6)
pylab.ylabel('Volatile fluxes (Tmol/yr)')

nextlabel= 'H$_2$O Escape'
pylab.loglog(total_time[0],-confidence_y[1][53]*365*24*60*60/(0.018*1e12),'b' ,label = nextlabel )
pylab.fill_between(total_time[0],-confidence_y[2][53]*365*24*60*60/(0.018*1e12),-confidence_y[0][53]*365*24*60*60/(0.018*1e12), color='blue', alpha='0.4')  
nextlabel= 'H$_2$O Ingassing'
pylab.loglog(total_time[0],-confidence_y[1][54]*365*24*60*60/(0.018*1e12),'m' ,label = nextlabel) 
pylab.fill_between(total_time[0],-confidence_y[2][54]*365*24*60*60/(0.018*1e12),-confidence_y[0][54]*365*24*60*60/(0.018*1e12), color='magenta', alpha='0.4')  
nextlabel= 'H$_2$O Outgassing'
pylab.loglog(total_time[0],confidence_y[1][55]*365*24*60*60/(0.018*1e12),'k' ,label = nextlabel) 
pylab.fill_between(total_time[0],confidence_y[0][55]*365*24*60*60/(0.018*1e12),confidence_y[2][55]*365*24*60*60/(0.018*1e12), color='grey', alpha='0.4') 

nextlabel= 'CO$_2$ Weathering'
pylab.loglog(total_time[0],-confidence_y[1][14],'g:' ,label = nextlabel )
pylab.fill_between(total_time[0],-confidence_y[2][14],-confidence_y[0][14], color='green', alpha='0.4')  
nextlabel= 'CO$_2$ Outgassing'
pylab.loglog(total_time[0],confidence_y[1][15],'r:' ,label = nextlabel ) 
pylab.fill_between(total_time[0],confidence_y[0][15],confidence_y[2][15], color='red', alpha='0.4')  

pylab.yscale('symlog',linthreshy = 0.001)
pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
pylab.xlim([x_low, x_high])
pylab.legend(frameon=False,ncol=2)

pylab.subplot(4,3,7)
pylab.ylabel('Melt production, MP (km$^3$/yr)')
pylab.loglog(total_time[0],confidence_melt[1],'r')
pylab.fill_between(total_time[0],confidence_melt[0],confidence_melt[2], color='red', alpha='0.4')  
pylab.xlabel('Time (yrs)')
pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
pylab.xlim([x_low, x_high])


pylab.subplot(4,3,8)
pylab.semilogx(total_time[0],confidence_f_O2_relative_FMQ[1],'r')#,label = 'Mantle fO2' )
pylab.fill_between(total_time[0],confidence_f_O2_relative_FMQ[0],confidence_f_O2_relative_FMQ[2], color='red', alpha='0.4') 

ypts = np.array([ -1.7352245862884175,1.827423167848699,2.425531914893617,3.5177304964539005,0.9172576832151291])
xpts = 4.5e9 - np.array([4.027378964941569, 4.162604340567613, 4.176627712854758, 4.345909849749583,4.363939899833055])*1e9
yrpts = 0*xpts + 2.3

pylab.ylabel("Mantle oxygen fugacity ($\Delta$QFM)")
pylab.xlabel("Time (yrs)")
pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
pylab.xlim([x_low, x_high])
pylab.ylim([-2.1, 2.1])
pylab.legend(frameon=False,ncol=2)

pylab.subplot(4,3,9)
pylab.semilogx(total_time[0],confidence_y[1][18]*365*24*60*60/(0.032*1e12),'g' ,label = 'Dry crustal')
pylab.fill_between(total_time[0],confidence_y[0][18]*365*24*60*60/(0.032*1e12),confidence_y[2][18]*365*24*60*60/(0.032*1e12), color='green', alpha='0.4')  
pylab.semilogx(total_time[0],confidence_y[1][19]*365*24*60*60/(0.032*1e12),'k' ,label = 'Escape')
pylab.fill_between(total_time[0],confidence_y[0][19]*365*24*60*60/(0.032*1e12),confidence_y[2][19]*365*24*60*60/(0.032*1e12), color='grey', alpha='0.4')  
pylab.semilogx(total_time[0],confidence_y[1][20]*365*24*60*60/(0.032*1e12),'b' ,label = 'Wet crustal')
pylab.fill_between(total_time[0],confidence_y[0][20]*365*24*60*60/(0.032*1e12),confidence_y[2][20]*365*24*60*60/(0.032*1e12), color='blue', alpha='0.4')  
pylab.semilogx(total_time[0],confidence_y[1][21]*365*24*60*60/(0.032*1e12),'r' ,label = 'Outgassing')
pylab.fill_between(total_time[0],confidence_y[0][21]*365*24*60*60/(0.032*1e12),confidence_y[2][21]*365*24*60*60/(0.032*1e12), color='red', alpha='0.4')  
pylab.semilogx(total_time[0],(confidence_y[1][18]+confidence_y[1][19]+confidence_y[1][20]+confidence_y[1][21])*365*24*60*60/(0.032*1e12),'c:' ,label = 'Net')
O2_label = 'O$_2$ flux (Tmol/yr)'
pylab.ylabel(O2_label)
pylab.xlabel("Time (yrs)")
pylab.yscale('symlog',linthreshy = 0.01)
pylab.legend(frameon=False,loc = 3,ncol=2)
pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
pylab.xlim([x_low, x_high])
pylab.minorticks_on()

pylab.subplot(4,3,10)
pylab.ylabel('Crustal thickness (km)')
pylab.semilogx(total_time[0],confidence_y[1][27]/(4*np.pi*rp**2*1000),'r')
pylab.fill_between(total_time[0],confidence_y[0][27]/(4*np.pi*rp**2*1000),confidence_y[2][27]/(4*np.pi*rp**2*1000), color='red', alpha='0.4')  
modlabel = 'Modern range'
pylab.semilogx(total_time[0],0*confidence_y[1][27]+20.0,'r--')
pylab.semilogx(total_time[0],0*confidence_y[1][27]+60.0,'r--', label=modlabel) 
pylab.xlabel('Time (yrs)')
pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
pylab.xlim([x_low, x_high])
pylab.ylim([-1, 125])
pylab.minorticks_on()
pylab.legend(frameon=False)

pylab.subplot(4,3,11)
pylab.ylabel('Atmospheric $^{40}$Ar (kg)')
pylab.loglog(total_time[0],confidence_y[1][38],'g')
pylab.fill_between(total_time[0],confidence_y[0][38],confidence_y[2][38], color='green', alpha='0.4')  
modlabel = 'Modern $^{40}$Ar'
pylab.semilogx(total_time[0],0*confidence_y[1][11]+1.61e16,'g--', label=modlabel)
pylab.xlabel('Time (yrs)')
pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
pylab.xlim([x_low, x_high])
pylab.minorticks_on()
pylab.legend(frameon=False)


pylab.ylabel('Atmospheric $^{4}$He (kg)')
pylab.ylabel('Trace gas mass (kg)')
pylab.loglog(total_time[0],confidence_y[1][50],'b')
pylab.fill_between(total_time[0],confidence_y[0][50],confidence_y[2][50], color='blue', alpha='0.4') 
modlabel = 'Modern $^{4}$He range'
pylab.semilogx(total_time[0],0*confidence_y[1][50]+1.3e14,'b--')
pylab.semilogx(total_time[0],0*confidence_y[1][50]+6.5e14,'b--', label=modlabel) 
pylab.xlabel('Time (yrs)')
pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
pylab.xlim([x_low, x_high])
pylab.minorticks_on()
pylab.legend(frameon=False)

pylab.subplot(4,3,12)
CO2label = 'CO$_2$'
pylab.semilogx(total_time[0],confidence_mantle_CO2_fraction[1],'g',label=CO2label)
pylab.fill_between(total_time[0],confidence_mantle_CO2_fraction[0],confidence_mantle_CO2_fraction[2], color='green', alpha='0.4')  
pylab.ylabel('Fraction solid')
H2Olabel = 'H$_2$O'
pylab.semilogx(total_time[0],confidence_mantle_H2O_fraction[1],'b',label=H2Olabel)
pylab.fill_between(total_time[0],confidence_mantle_H2O_fraction[0],confidence_mantle_H2O_fraction[2], color='blue', alpha='0.4') 
pylab.xlabel('Time (yrs)')
pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])
pylab.xlim([x_low, x_high])
pylab.minorticks_on()
pylab.legend(frameon=False)

### waterworlds comparison
confidence_y90=scipy.stats.scoreatpercentile(total_y ,[5.0,50,95.0], interpolation_method='fraction',axis=0)
pylab.figure()
pylab.subplot(5,1,1)
pylab.semilogx(total_time[0],confidence_y[1][13],'g')
for k in range(0,len(inputs)):
    pylab.semilogx(total_time[0],total_y[k][13],'g')
pylab.xlabel('Time (yrs)')
pylab.ylabel('Mantle CO2')
pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])

pylab.subplot(5,1,2)
pylab.semilogx(total_time[0],confidence_y[1][15],'r' ,label = 'Outgassing' ) 
for k in range(0,len(inputs)):
    pylab.semilogx(total_time[0],total_y[k][15],'g')
pylab.xlabel('Time (yrs)')
pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])

pylab.subplot(5,1,3)
pylab.ylabel('Crustal thickness (km)')
pylab.semilogx(total_time[0],confidence_y[1][16]/1000,'r')
for k in range(0,len(inputs)):
    pylab.semilogx(total_time[0],total_y[k][16]/1000,'g')
    pylab.semilogx(total_time[0],total_y[k][27]/(4*np.pi*rp**2*1000),'k')
pylab.xlabel('Time (yrs)')
pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])

pylab.subplot(5,1,4)
pylab.ylabel('Outgassing  melt frac')
for k in range(0,len(inputs)):
    pylab.semilogx(total_time[0],actual_phi_surf_melt_ar[k],'g')
pylab.xlabel('Time (yrs)')
pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9]) 

pylab.subplot(5,1,5)
pylab.ylabel('Outgassing  melt frac')
for k in range(0,len(inputs)):
    pylab.semilogx(total_time[0],XCO2_melt[k],'r')
    pylab.semilogx(total_time[0],XH2O_melt[k],'b')
pylab.xlabel('Time (yrs)')
pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9]) 


pylab.figure()
pylab.subplot(3,1,1)
pylab.loglog(total_time[0],confidence_y[1][35],'r',label='40K Mantle') 
pylab.fill_between(total_time[0],confidence_y[0][35],confidence_y[2][35], color='red', alpha='0.4')  
pylab.loglog(total_time[0],confidence_y[1][36],'b',label='40Ar Mantle') 
pylab.fill_between(total_time[0],confidence_y[0][36],confidence_y[2][36], color='blue', alpha='0.4')  
pylab.loglog(total_time[0],confidence_y[1][37],'g',label='40K lid') 
pylab.fill_between(total_time[0],confidence_y[0][37],confidence_y[2][37], color='green', alpha='0.4')  

con_total_Ar40 = scipy.stats.scoreatpercentile(total_Ar40 ,[int1,int2,int3], interpolation_method='fraction',axis=0)
con_total_K40 = scipy.stats.scoreatpercentile(total_K40 ,[int1,int2,int3], interpolation_method='fraction',axis=0)

pylab.legend()       
pylab.subplot(3,1,2)
pylab.ylabel('40Ar atmo')
pylab.loglog(total_time[0],confidence_y[1][38],'g')#
pylab.fill_between(total_time[0],confidence_y[0][38],confidence_y[2][38], color='green', alpha='0.4')  

pylab.subplot(3,1,3)
pylab.ylabel('40Ar atmo / Ar40 total')
pylab.semilogx(total_time[0],con_total_Ar40[1],'g')#
pylab.fill_between(total_time[0],con_total_Ar40[0],con_total_Ar40[2], color='green', alpha='0.4')  

pylab.figure()
pylab.subplot(3,1,1)
pylab.loglog(total_time[0],confidence_y[1][43],'r',label='238U Mantle') 
pylab.fill_between(total_time[0],confidence_y[0][43],confidence_y[2][43], color='red', alpha='0.4')  
pylab.loglog(total_time[0],confidence_y[1][44],'b',label='235U Mantle') 
pylab.fill_between(total_time[0],confidence_y[0][44],confidence_y[2][44], color='blue', alpha='0.4')  
pylab.loglog(total_time[0],confidence_y[1][45],'g',label='Th Mantle') 
pylab.fill_between(total_time[0],confidence_y[0][45],confidence_y[2][45], color='green', alpha='0.4')  
pylab.legend()

pylab.subplot(3,1,2)
pylab.loglog(total_time[0],confidence_y[1][46],'r',label='238U Lid') 
pylab.fill_between(total_time[0],confidence_y[0][46],confidence_y[2][46], color='red', alpha='0.4')  
pylab.loglog(total_time[0],confidence_y[1][47],'b',label='235U Lid') 
pylab.fill_between(total_time[0],confidence_y[0][47],confidence_y[2][47], color='blue', alpha='0.4')  
pylab.loglog(total_time[0],confidence_y[1][48],'g',label='Th Lid') 
pylab.fill_between(total_time[0],confidence_y[0][48],confidence_y[2][48], color='green', alpha='0.4')  
pylab.legend()

pylab.subplot(3,1,3)
pylab.loglog(total_time[0],confidence_y[1][49],'r',label='4He Mantle') 
pylab.fill_between(total_time[0],confidence_y[0][49],confidence_y[2][49], color='red', alpha='0.4')  
pylab.loglog(total_time[0],confidence_y[1][50],'b',label='4He Atmo')
pylab.fill_between(total_time[0],confidence_y[0][50],confidence_y[2][50], color='blue', alpha='0.4')  
pylab.legend()

pylab.figure()
pylab.subplot(3,1,1)
pylab.ylabel('Power, TW')
pylab.loglog(total_time[0],confidence_y[1][32]/1e12,'g',label='Convective loss') 
pylab.loglog(total_time[0],confidence_y[1][33]/1e12,'k',label='Volanic loss') 
pylab.loglog(total_time[0],confidence_y[1][30]/1e12,'r',label='Mantle heat prod')
pylab.loglog(total_time[0],confidence_y[1][32]/1e12+confidence_y[1][33]/1e12,'b--',label='Total loss')
pylab.legend()

pylab.subplot(3,1,2)
pylab.ylabel('Equivalent surface flux mW/m2')
pylab.loglog(total_time[0],1000*confidence_y[1][32]/(4*np.pi*rp**2),'g',label='Convective loss') 
pylab.fill_between(total_time[0],1000*confidence_y[0][32]/(4*np.pi*rp**2),1000*confidence_y[2][32]/(4*np.pi*rp**2), color='green', alpha='0.4')  
pylab.loglog(total_time[0],1000*confidence_y[1][33]/(4*np.pi*rp**2),'k',label='Volanic loss') 
pylab.fill_between(total_time[0],1000*confidence_y[0][33]/(4*np.pi*rp**2),1000*confidence_y[2][33]/(4*np.pi*rp**2), color='grey', alpha='0.4') 
pylab.loglog(total_time[0],1000*confidence_y[1][30]/(4*np.pi*rp**2),'r',label='Mantle heat prod')
pylab.fill_between(total_time[0],1000*confidence_y[0][30]/(4*np.pi*rp**2),1000*confidence_y[2][30]/(4*np.pi*rp**2), color='red', alpha='0.4') 
pylab.loglog(total_time[0],1000*confidence_y[1][32]/(4*np.pi*rp**2)+1000*confidence_y[1][33]/(4*np.pi*rp**2),'b--',label='Total loss')
pylab.fill_between(total_time[0],1000*confidence_y[0][32]/(4*np.pi*rp**2)+1000*confidence_y[0][33]/(4*np.pi*rp**2),1000*confidence_y[2][32]/(4*np.pi*rp**2)+1000*confidence_y[2][33]/(4*np.pi*rp**2), color='blue', alpha='0.4') 
pylab.legend()

pylab.subplot(3,1,3)
pylab.ylabel('Crustal thickness (km)')
pylab.semilogx(total_time[0],confidence_y[1][27]/(4*np.pi*rp**2*1000),'r')
pylab.fill_between(total_time[0],confidence_y[0][27]/(4*np.pi*rp**2*1000),confidence_y[2][27]/(4*np.pi*rp**2*1000), color='red', alpha='0.4')  
pylab.xlabel('Time (yrs)')
pylab.xticks([1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9])

pylab.figure()
pylab.subplot(2,2,1)
pylab.loglog(total_time[0],conf_DH_atmo[1],'g',label='D/H surface')
pylab.fill_between(total_time[0],conf_DH_atmo[0],conf_DH_atmo[2], color='green', alpha='0.4')  
pylab.loglog(total_time[0],conf_DH_solid[1],'r',label='D/H interior')
pylab.fill_between(total_time[0],conf_DH_solid[0],conf_DH_solid[2], color='red', alpha='0.4')
pylab.legend()
   
pylab.subplot(2,2,3)
pylab.loglog(total_time[0],0.5*confidence_y[1][41],'g',label='D surface')
pylab.fill_between(total_time[0],0.5*confidence_y[0][41],0.5*confidence_y[2][41], color='green', alpha='0.4')  
pylab.loglog(total_time[0],0.5*confidence_y[1][42],'r',label='D interior')
pylab.fill_between(total_time[0],0.5*confidence_y[0][42],0.5*confidence_y[2][42], color='red', alpha='0.4')
pylab.legend()

pylab.subplot(2,2,2)
for k in range(0,len(inputs)):
    pylab.loglog(total_time[0],0.5*total_y[k][41]/total_y[k][1],'g')
    pylab.loglog(total_time[0],0.5*total_y[k][42]/total_y[k][0],'r')
pylab.legend()
   
pylab.subplot(2,2,4)
for k in range(0,len(inputs)):
    pylab.loglog(total_time[0],total_y[k][41],'g')
    pylab.loglog(total_time[0],total_y[k][42],'r')

pylab.legend()

pylab.figure()
for k in range(0,len(new_inputs)):
    pylab.plot(total_y[k][16][-50]/1000,total_y[k][15][-50],'.')
pylab.title(total_time[0][-50])
pylab.xlabel('Crust thickness')
pylab.ylabel('CO2 outgas (Tmol/yr)')

pylab.figure()
for k in range(0,len(new_inputs)):
    pylab.semilogx(total_y[k][13][-50]/mantle_mass,total_y[k][15][-50],'.')
pylab.title(total_time[0][-50])
pylab.xlabel('Mantle CO2')
pylab.ylabel('CO2 outgas (Tmol/yr)')

pylab.figure()
pylab.ylabel("MMW")
pylab.semilogx(total_time[0],confidence_y[1][24],'g')
pylab.fill_between(total_time[0],confidence_y[0][24],confidence_y[2][24], color='green', alpha='0.4')  
pylab.xlabel('Time (yrs)')

pylab.figure()
pylab.ylabel("Solid mantle Fe3+/Fe2+")
pylab.semilogx(total_time[0],confidence_iron_ratio[1],'k')
pylab.fill_between(total_time[0],confidence_iron_ratio[0],confidence_iron_ratio[2], color='grey', alpha='0.4')  
pylab.xlabel('Time (yrs)')

pylab.tight_layout()

#######################################################################
#######################################################################
### Begin parameter space plots 
######## First, need to fill input arrays from input files
#######################################################################

pylab.figure()
pylab.subplot(4,1,1)
O2_final_ar = []
H2O_final_ar = []
CO2_final_ar = []
Total_P_ar = []
atmo_H2O_ar = []


for k in range(0,len(inputs)):
    addO2 = Mass_O_atm[k][-1]*g/(4*np.pi*(0.032/total_y[k][24][-1])*rp**2*1e5)
    if (addO2 <= 0 ) or np.isnan(addO2):
        addO2 = 1e-8
    addH2O = Pressre_H2O[k][-1]/1e5
    atmo_H2O = water_frac[k][-1]
    if (addH2O <= 0 ) or np.isnan(addH2O):
        addH2O = 1e-8
    if (atmo_H2O<=0) or np.isnan(atmo_H2O):
        atmo_H2O = 0
    addCO2 = CO2_Pressure_array[k][-1]/1e5
    if (addCO2 <= 0 ) or np.isnan(addCO2):
        addCO2 = 1e-8
    O2_final_ar.append(np.log10(addO2))
    H2O_final_ar.append(np.log10(addH2O))
    CO2_final_ar.append(np.log10(addCO2))
    atmo_H2O_ar.append(atmo_H2O)
    Total_P_ar.append(np.log10(addO2+addH2O+addCO2))

pylab.hist(O2_final_ar,bins = 50,color = 'g')
pylab.xlabel('log(pO2)')
pylab.subplot(4,1,2)
pylab.hist(H2O_final_ar,color = 'b',bins = 50)
pylab.xlabel('log(pH2O)')
pylab.subplot(4,1,3)
pylab.hist(CO2_final_ar,color = 'r',bins = 50)
pylab.xlabel('log(pCO2)')
pylab.subplot(4,1,4)
pylab.hist(Total_P_ar,color = 'c',bins = 50)
pylab.xlabel('Log10(Pressure (bar))')

init_CO2_H2O = []
init_CO2_ar =[]
Final_O2 = []
Final_CO2 = []
Final_H2O = []
Surface_T_ar = []
H2O_upper_ar = []
Weathering_limit = []
tsat_array = []
epsilon_array = []

Ca_array = []
Omega_ar = []
init_H2O_ar=[]
offset_ar = []
Te_ar =[]
expCO2_ar = []
Mfrac_hydrated_ar= []
dry_frac_ar = []
wet_OxFrac_ar = []
Radiogenic= []
Init_fluid_O_ar = []
albedoH_ar = []
albedoC_ar = []
MaxMantleH2O=[]
imp_coef_ar = []
imp_slope_ar = []
hist_total_imp_mass=[]
mult_ar= []
mix_epsilon_ar=[]
Transition_time = []
Final_40Ar = []
Ar40_ratio = []
beta_array = []
completion_time = []
surface_magma_frac_array = []
Tstrat_ar = []

for k in range(0,len(inputs)):
    Tstrat_ar.append(inputs_for_MC[k][5].Tstrat)
    init_CO2 = total_y[k][12][0]+total_y[k][13][0]
    init_H2O = total_y[k][0][0]+total_y[k][1][0]
    surface_magma_frac_array.append(inputs_for_MC[k][5].surface_magma_frac)
    init_H2O_ar.append(inputs_for_MC[k][2].Init_fluid_H2O)
    Init_fluid_O_ar.append(inputs_for_MC[k][2].Init_fluid_O)
    albedoC_ar.append(inputs_for_MC[k][1].albedoC)
    albedoH_ar.append(inputs_for_MC[k][1].albedoH)
    init_CO2_H2O.append(init_CO2/init_H2O)
    init_CO2_ar.append(inputs_for_MC[k][2].Init_fluid_CO2)
    Final_O2.append(total_y[k][22][-1]/1e5)
    Final_CO2.append(CO2_Pressure_array[k][-1]/1e5) 
    Final_H2O.append(water_frac[k][-1]*Pressre_H2O[k][-1]/1e5)
    H2O_upper_ar.append(total_y[k][14][-1])
    Surface_T_ar.append(total_y[k][8][-1])
    Weathering_limit.append(inputs_for_MC[k][5].supp_lim)
    tsat_array.append(inputs_for_MC[k][4].tsat_XUV)
    epsilon_array.append(inputs_for_MC[k][4].epsilon)
    beta_array.append(inputs_for_MC[k][4].beta0)
    Ca_array.append(inputs_for_MC[k][5].ocean_a)
    Omega_ar.append(inputs_for_MC[k][5].ocean_b)
    offset_ar.append(inputs_for_MC[k][5].interiora)
    Mfrac_hydrated_ar.append(inputs_for_MC[k][5].interiorb)
    Te_ar.append(inputs_for_MC[k][5].ccycle_a)
    expCO2_ar.append(inputs_for_MC[k][5].ccycle_b) 
    dry_frac_ar.append(inputs_for_MC[k][5].interiorc)
    wet_OxFrac_ar.append(inputs_for_MC[k][5].interiord)
    Radiogenic.append(inputs_for_MC[k][5].interiore)
    MaxMantleH2O.append(inputs_for_MC[k][5].interiorf)
    Transition_time.append(inputs_for_MC[k][5].interiorg)
    Final_40Ar.append(total_y[k][38][-1])
    Ar40_ratio.append(total_Ar40[k][-1])
    imp_coef_ar.append(inputs_for_MC[k][5].esc_a)
    imp_slope_ar.append(inputs_for_MC[k][5].esc_b)
    mult_ar.append(inputs_for_MC[k][5].esc_c)
    mix_epsilon_ar.append(inputs_for_MC[k][5].esc_d)
    t_ar = np.linspace(0,1,1000)
    y = np.copy(t_ar)
    for i in range(0,len(t_ar)):
        y[i] = inputs_for_MC[k][5].esc_a*np.exp(-t_ar[i]/inputs_for_MC[k][5].esc_b)
    hist_total_imp_mass.append(np.trapz(y,t_ar*1e9))
    completion_time.append(total_y[k][41][-1])


Ca_array = np.array(Ca_array)
Omega_ar = np.array(Omega_ar)


pylab.figure()
pylab.subplot(2,2,1)
pylab.loglog(dry_frac_ar,Final_O2,'.')
pylab.xlabel('Efficiency dry crustal oxidation, $f_{dry-oxid}$')
pylab.ylabel('Final O$_2$ (bar)')
pylab.xlim([1e-4,1e-1])

pylab.subplot(2,2,2)
pylab.semilogy(epsilon_array,Final_O2,'.')
pylab.xlabel('Low XUV escape efficiency, $\epsilon$$_{lowXUV}$ ')
pylab.ylabel('Final O$_2$ (bar)')

pylab.subplot(2,2,3)
pylab.loglog(np.array(init_H2O_ar)/1.4e21,np.array(init_CO2_ar)*g/(1e5*4*np.pi*rp**2),'.')
pylab.xlabel('Initial H$_2$O inventory (Earth oceans)')
pylab.ylabel('Initial CO$_2$ inventory (bar)')

pylab.subplot(2,2,4)
pylab.loglog(surface_magma_frac_array,Final_O2,'.')
pylab.xlabel('Extrusive lava fraction, $f_{lava}$')
pylab.ylabel('Final O$_2$ (bar)')
pylab.xlim([1e-4,1])

pylab.figure()
pylab.semilogy(Tstrat_ar,Final_O2,'b.') 
pylab.xlabel('T$_{stratosphere}$ (K)')
pylab.ylabel('Final O$_2$ (bar)')
pylab.xlim([150,250])

pylab.figure()
pylab.title('completion time (min)')

pylab.subplot(2,2,1)
pylab.plot(Radiogenic,completion_time,'x')
pylab.xlabel('Radiongenic')

pylab.subplot(2,2,2)
pylab.semilogx(offset_ar,completion_time,'x')
pylab.xlabel('offset_ar')

pylab.subplot(2,2,3)
pylab.semilogx(init_CO2_ar,completion_time,'x')
pylab.xlabel('init_CO2_ar')

pylab.subplot(2,2,4)
pylab.semilogx(init_H2O_ar,completion_time,'x')
pylab.xlabel('init_H2O_ar')

pylab.figure()
pylab.semilogy(np.log10(hist_total_imp_mass),Final_O2,'.')
pylab.xlabel('Total impactor mass, log$_{10}$(kg)')
pylab.ylabel('Final O$_2$ (bar)')

pylab.figure()
pylab.subplot(2,1,1)
pylab.plot(Transition_time,Final_40Ar,'.')
pylab.xlabel('Transition_time (yrs)')
pylab.ylabel('Final_40Ar')
pylab.subplot(2,1,2)
pylab.plot(Transition_time,Ar40_ratio,'.')
pylab.xlabel('Transition_time (yrs)')
pylab.ylabel('Final_40Ar_ratio')

pylab.figure()
pylab.subplot(2,3,1)
pylab.loglog(np.array(init_H2O_ar)/1.4e21,np.array(init_CO2_ar)*g/(1e5*4*np.pi*rp**2),'.')
pylab.xlabel('Initial H2O inventory (Earth oceans)')
pylab.ylabel('Initial CO2 inventory (bar)')

pylab.subplot(2,3,2)
pylab.loglog(init_H2O_ar,init_CO2_ar,'.')
pylab.xlabel('Initial H2O inventory')
pylab.ylabel('Initial CO2 inventory')

pylab.subplot(2,3,3)
pylab.loglog(Radiogenic,offset_ar,'.')
pylab.xlabel('Radiogenic')
pylab.ylabel('offset_ar')

pylab.subplot(2,3,4)
pylab.loglog(Init_fluid_O_ar,wet_OxFrac_ar,'.')
pylab.xlabel('Init_fluid_O')
pylab.ylabel('wet_OxFrac_ar')

pylab.subplot(2,3,6)
pylab.loglog(MaxMantleH2O,Final_O2,'.')
pylab.xlabel('MaxMantleH2O')
pylab.ylabel('Final_O2')

pylab.subplot(2,3,5)
pylab.loglog(albedoC_ar,albedoH_ar,'.')
pylab.xlabel('AlbedoC_ar')
pylab.ylabel('AlbedoH_ar')

pylab.figure()
pylab.subplot(2,3,1)
pylab.semilogy(Te_ar,Final_O2,'.')
pylab.xlabel('Te_ar (K)')
pylab.ylabel('Final O2 (bar)')

pylab.subplot(2,3,2)
pylab.semilogy(expCO2_ar,Final_O2,'.')
pylab.xlabel('expCO2_ar')
pylab.ylabel('Final O2 (bar)')

pylab.subplot(2,3,3)
pylab.loglog(Mfrac_hydrated_ar,Final_O2,'.')
pylab.xlabel('Mfrac_hydrated_ar')
pylab.ylabel('Final O2 (bar)')

pylab.subplot(2,3,4)
pylab.loglog(dry_frac_ar,Final_O2,'.')
pylab.xlabel('dry_frac_ar')
pylab.ylabel('Final O2 (bar)')

pylab.subplot(2,3,5)
pylab.loglog(wet_OxFrac_ar,Final_O2,'.')
pylab.xlabel('wet_OxFrac_ar')
pylab.ylabel('Final O2 (bar)')


pylab.subplot(2,3,6)
pylab.loglog(Radiogenic,Final_O2,'.')
pylab.xlabel('Radiogenic')
pylab.ylabel('Final O2 (bar)')

pylab.figure()
pylab.subplot(1,2,1)
pylab.loglog(dry_frac_ar,Final_O2,'.')
pylab.xlabel('Efficiency dry crustal oxidation, $f_{dry-oxid}$')
pylab.ylabel('Final O$_2$ (bar)')

pylab.subplot(1,2,2)
pylab.loglog(init_CO2_H2O,Final_O2,'.')
pylab.xlabel('Initial CO$_2$:H$_2$O')
pylab.ylabel('Final O$_2$ (bar)')

pylab.figure()
pylab.loglog(init_H2O_ar,Final_O2,'.')
pylab.xlabel('Initial H$_2$O (kg)')
pylab.ylabel('Final O$_2$ (bar)')

pylab.figure()
pylab.loglog(np.array(init_H2O_ar)/1.4e21,Final_O2,'.')
pylab.xlabel('Initial H$_2$O (Earth oceans)')
pylab.ylabel('Final O$_2$ (bar)')

pylab.figure()
pylab.subplot(1,3,1)
pylab.loglog(dry_frac_ar,Final_O2,'.')
pylab.xlabel('Efficiency dry crustal oxidation, $f_{dry-oxid}$')
pylab.ylabel('Final O$_2$ (bar)')

pylab.subplot(1,3,2)
pylab.semilogy(epsilon_array,Final_O2,'.')
pylab.xlabel('Low XUV escape efficiency, $\epsilon$$_{lowXUV}$ ')
pylab.ylabel('Final O$_2$ (bar)')

pylab.subplot(1,3,3)
pylab.loglog(np.array(init_H2O_ar)/1.4e21,np.array(init_CO2_ar)*g/(1e5*4*np.pi*rp**2),'.')
pylab.xlabel('Initial H$_2$O inventory (Earth oceans)')
pylab.ylabel('Initial CO$_2$ inventory (bar)')

pylab.figure()
pylab.subplot(2,3,1)
pylab.semilogy(epsilon_array,Final_O2,'.')
pylab.xlabel('epsilon (for XUV)')
pylab.ylabel('Final O2 (bar)')

pylab.subplot(2,3,2)
pylab.loglog(Omega_ar/Ca_array,Final_O2,'.')
pylab.xlabel('omega/Ca ~ CO3')
pylab.ylabel('Final O2 (bar)')

pylab.subplot(2,3,3)
pylab.loglog(offset_ar,Final_O2,'.')
pylab.xlabel('offset')
pylab.ylabel('Final O2 (bar)')

pylab.subplot(2,3,4)
pylab.loglog(init_H2O_ar,Final_O2,'.')
pylab.xlabel('init_H2O')
pylab.ylabel('Final O2 (bar)')

pylab.subplot(2,3,5)
pylab.loglog(mult_ar,Final_O2,'.')
pylab.xlabel('mult_ar')
pylab.ylabel('Final O2 (bar)')

pylab.subplot(2,3,6)
pylab.semilogy(mix_epsilon_ar,Final_O2,'.')
pylab.xlabel('mix_epsilon_ar')
pylab.ylabel('Final O2 (bar)')

pylab.figure()
pylab.subplot(2,2,1)
pylab.plot(Surface_T_ar,H2O_upper_ar,'.')
pylab.xlabel('SurfT')
pylab.ylabel('y14 H2O upper atmo frac')

pylab.subplot(2,2,2)
pylab.semilogy(Surface_T_ar,Final_H2O,'.')
pylab.xlabel('SurfT')
pylab.ylabel('atmo H2o pressure (bar)')

pylab.subplot(2,2,3)
pylab.loglog(init_CO2_H2O,Final_O2,'.')
pylab.xlabel('CO2/H2O init')
pylab.ylabel('final pO2 (bar)')

pylab.subplot(2,2,4)
pylab.loglog(init_CO2_ar,Final_O2,'.')
pylab.xlabel('CO2 init')
pylab.ylabel('final pO2 (bar)')

pylab.figure()
pylab.subplot(3,2,1)
pylab.loglog(Final_H2O,Final_O2,'.')
pylab.xlabel('Final_H2O (bar)')
pylab.ylabel('final pO2 (bar)')

pylab.subplot(3,2,2)
pylab.loglog(Final_CO2,Final_O2,'.')
pylab.xlabel('final CO2 (bar)')
pylab.ylabel('final pO2 (bar)')


pylab.subplot(3,2,3)
pylab.loglog(Weathering_limit,Final_O2,'.')
pylab.xlabel('Weathering limit (kg/s)')
pylab.ylabel('final pO2 (bar)')

pylab.subplot(3,2,4)
pylab.loglog(tsat_array,Final_O2,'.')
pylab.xlabel('tsat XUV (Gyr)')
pylab.ylabel('final pO2 (bar)')

pylab.subplot(3,2,5)
pylab.semilogy(beta_array,Final_O2,'.')
pylab.xlabel('beta0')
pylab.ylabel('final pO2 (bar)')

pylab.subplot(3,2,6)
pylab.loglog(Final_H2O,Final_CO2,'.')
pylab.xlabel('Final_H2O (bar)')
pylab.ylabel('final CO2 (bar)')

pylab.figure()
pylab.subplot(6,1,1)
pylab.ylabel('Mass H2O solid, kg')

for k in range(0,len(inputs)):
    pylab.semilogx(total_time[k],total_y[k][0])
pylab.subplot(6,1,2)
pylab.ylabel('H2O reservoir (kg)')
for k in range(0,len(inputs)):
    pylab.semilogx(total_time[k],total_y[k][1],'r',label='Mass H2O, magma ocean + atmo'  if k == 0 else "")
    pylab.semilogx(total_time[k],MH2O_liq[k],'b',label='Mass H2O, magma ocean'  if k == 0 else "")
    pylab.semilogx(total_time[k],Pressre_H2O[k] *4 * np.pi * (0.018/total_y[k][24])* (rp**2/g) ,label='Mass H2O atmosphere'  if k == 0 else "")
pylab.subplot(6,1,3)
for k in range(0,len(inputs)):
    pylab.ylabel('H2O fraction solid')
    pylab.semilogx(total_time[k],total_y[k][0]/(total_y[k][0]+total_y[k][1]))
pylab.subplot(6,1,4)
for k in range(0,len(inputs)):
    pylab.ylabel('CO2 fraction solid')
    pylab.semilogx(total_time[k],total_y[k][13]/(total_y[k][13]+total_y[k][12]))
pylab.subplot(6,1,5)
pylab.semilogx(total_time[0],confidence_mantle_CO2_fraction[1],'k')
pylab.fill_between(total_time[0],confidence_mantle_CO2_fraction[0],confidence_mantle_CO2_fraction[2], color='grey', alpha='0.4')  
pylab.ylabel('CO2 fraction solid')
pylab.xlabel('Time (yrs)')
pylab.subplot(6,1,6)
pylab.semilogx(total_time[0],confidence_mantle_H2O_fraction[1],'k')
pylab.fill_between(total_time[0],confidence_mantle_H2O_fraction[0],confidence_mantle_H2O_fraction[2], color='grey', alpha='0.4')  
pylab.ylabel('H2O fraction solid')
pylab.xlabel('Time (yrs)')
pylab.show()

