import numpy as np
from scipy.interpolate import interp1d

def main_sun_fun(time,stellar_mass,tsat_XUV,beta_XUV,fsat):
    
    if stellar_mass == 1.0: 
        stellar_data = np.loadtxt('Baraffe3.txt',skiprows=31) # for reproducing sun exactly
    else:
        print ("This version of code is only set up for solar mass stars")
        return [time*0,time*0,time*0,time*0]
    
    stellar_array=[]
    for i in range(0,len(stellar_data[:,0])):
        if stellar_data[i,0] == stellar_mass:
            stellar_array.append(stellar_data[i,:])
            
    stellar_array=np.array(stellar_array)

    min_time = np.min(stellar_array[:,1])
    max_time = np.max(stellar_array[:,1])
    
    if (min_time>np.min(time) ) or (max_time<np.max(time)):
        print ("Problem: exceeding time range for stellar data")
    
    time_array = stellar_array[:,1]
    
    Total_Lum = (10**stellar_array[:,4])
    
    ratio_out = [] # For XUV ratio
    for i in range(0,len(time_array)):
        if time_array[i]<tsat_XUV:
            ratio= fsat
        else:
            ratio = fsat*(time_array[i]/tsat_XUV)**beta_XUV  
        ratio_out.append(ratio)
  
    XUV_Lum = ratio_out*Total_Lum
    
    Total_fun = interp1d(time_array,Total_Lum)
    XUV_fun = interp1d(time_array,XUV_Lum)
    Relative_total_Lum = Total_fun(time)
    Relative_XUV_lum = XUV_fun(time)
    Absolute_total_Lum = Relative_total_Lum*3.828e26 
    Absolute_XUV_Lum = Relative_XUV_lum*3.828e26
    
    return [Relative_total_Lum,Relative_XUV_lum,Absolute_total_Lum,Absolute_XUV_Lum ]
