class Switch_Inputs:
  def __init__(self, print_switch, speedup_flag, start_speed , fin_speed,heating_switch,C_cycle_switch,Start_time):
    self.print_switch = print_switch
    self.speedup_flag = speedup_flag
    self.start_speed = start_speed
    self.fin_speed = fin_speed
    self.heating_switch = heating_switch
    self.C_cycle_switch = C_cycle_switch
    self.Start_time = Start_time

class Planet_inputs:
  def __init__(self, RE,ME,rc,pm,Total_Fe_mol_fraction,Planet_sep,albedoC,albedoH):
    self.RE = RE
    self.ME = ME
    self.rc = rc
    self.pm = pm
    self.Total_Fe_mol_fraction = Total_Fe_mol_fraction
    self.Planet_sep = Planet_sep
    self.albedoC = albedoC
    self.albedoH = albedoH
    
class Init_conditions:
  def __init__(self, Init_solid_H2O,Init_fluid_H2O,Init_solid_O,Init_fluid_O,Init_solid_FeO1_5,Init_solid_FeO,Init_solid_CO2,Init_fluid_CO2):
    self.Init_solid_H2O = Init_solid_H2O
    self.Init_fluid_H2O = Init_fluid_H2O
    self.Init_solid_O = Init_solid_O
    self.Init_fluid_O = Init_fluid_O     
    self.Init_solid_FeO1_5 = Init_solid_FeO1_5
    self.Init_solid_FeO = Init_solid_FeO
    self.Init_solid_CO2 = Init_solid_CO2
    self.Init_fluid_CO2 = Init_fluid_CO2


class Numerics:
  def __init__(self, total_steps,step0,step1,step2,step3,step4,tfin0,tfin1,tfin2,tfin3,tfin4):
    self.total_steps = total_steps
    self.step0 = step0
    self.step1 = step1
    self.step2 = step2
    self.step3 = step3
    self.step4 = step4
    self.tfin0 = tfin0
    self.tfin1 = tfin1
    self.tfin2 = tfin2
    self.tfin3 = tfin3
    self.tfin4 = tfin4


class Stellar_inputs:
  def __init__(self, tsat_XUV, Stellar_Mass,fsat, beta0 , epsilon ):
    self.tsat_XUV = tsat_XUV
    self.Stellar_Mass = Stellar_Mass
    self.fsat = fsat
    self.beta0 = beta0
    self.epsilon  = epsilon 
    
class MC_inputs:
  def __init__(self, esc_a, esc_b, esc_c, esc_d,ccycle_a , ccycle_b ,supp_lim, interiora , interiorb,interiorc,interiord,interiore,interiorf,interiorg,ocean_a,ocean_b,K_over_U,Tstrat,surface_magma_frac):
    self.esc_a = esc_a
    self.esc_b = esc_b
    self.esc_c = esc_c
    self.esc_d = esc_d
    self.ccycle_a = ccycle_a
    self.ccycle_b  = ccycle_b 
    self.supp_lim = supp_lim
    self.interiora = interiora
    self.interiorb = interiorb
    self.interiorc = interiorc
    self.interiord = interiord
    self.interiore = interiore
    self.interiorf = interiorf
    self.interiorg = interiorg
    self.ocean_a = ocean_a
    self.ocean_b = ocean_b
    self.K_over_U = K_over_U
    self.Tstrat = Tstrat
    self.surface_magma_frac = surface_magma_frac

class Model_outputs:
  def __init__(self, total_time,total_y,FH2O_array,FCO2_array,MH2O_liq,MH2O_crystal,MCO2_liq,Pressre_H2O,CO2_Pressure_array,fO2_array,Mass_O_atm,Mass_O_dissolved,water_frac,Ocean_depth,Max_depth,Ocean_fraction):
    self.total_time = total_time
    self.total_y = total_y
    self.FH2O_array = FH2O_array
    self.FCO2_array = FCO2_array     
    self.MH2O_liq = MH2O_liq
    self.MH2O_crystal = MH2O_crystal
    self.MCO2_liq = MCO2_liq
    self.Pressre_H2O = Pressre_H2O
    self.CO2_Pressure_array = CO2_Pressure_array
    self.fO2_array = fO2_array
    self.Mass_O_atm = Mass_O_atm
    self.Mass_O_dissolved = Mass_O_dissolved
    self.water_frac = water_frac
    self.Ocean_depth = Ocean_depth
    self.Max_depth = Max_depth
    self.Ocean_fraction = Ocean_fraction
