PACMAN Venus - Version 1.0

This set of python scripts runs the Planetary Atmosphere, Crust, and MANtle (PACMAN) geochemical evolution model for Venus, as described in Krissansen-Totton et al. (2021) "Was Venus ever habitable? Constraints from a coupled interior-atmosphere-redox evolution model", Planetary Science Journal. As a matter of courtesy, we request that people using this code please cite Krissansen-Totton et al. (2021). We also request that authors who use and modify the code, please send a copy of papers to the lead author (jkt@ucsc.edu). This version of the code (V1.0) will be archived upon publication. For updated versions, check Github.com/joshuakt/Venus-evolution.

REQUIREMENTS: Python 3.0, including numpy, pylab, scipy, joblib, and numba modules. Additionally, the Volcgases outgassing module (Wogan et al. 2020;PSJ) must be installed in python prior to running the PACMAN code: https://github.com/Nicholaswogan/VolcGases

HOW TO RUN CODE:
(1) Put all the python scripts in the same directory, and ensure python is working in this directory.
(2) Open MC_calc.py and check desired parameter ranges, number of iterations, and number of cores for parallelization. Run MC_calc.py to execute Monte Carlo calculations over chosen parameter ranges.
(3) When complete, run Plot_MC_output.py to plot results from Monte Carlo calculations.

Note that default input values in MC_calc.py runs the forward model 720 times (num_runs = 720), parallelized over 60 independent threads (num_cores = 60). This takes approximately 1 hour. The number of cores and total forward model calls should be adjusted as needed. Output files from Monte Carlo calculations are large (~9 Gb for 720 model runs)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
EXPLANATION OF CODE STRUCTURE:

%% MC_calc.py
This python script perfoms the Monte Carlo calculations to reproduce results from the main text. It also contains nominal parameter ranges, which can be altered to reproduce different scenarios and sensitivity tests. Model outputs are saved as "Venus_ouputs_revisions.npy", whereas corresponding input parameters are saved as "Venus_inputs_revisions.npy". Note that these will be overwritten every time MC_Calc.py finishes running, unless file names are changed manually. These two files are subsequently used by the script "Plot_MC_output.py" to plot results. The file Venus_inputs has dimensions ITERATIONS X 6, where the second dimension contains the six classes that define all input parameters: inputs, Planet_inputs, Init_conditions, Numerics, Stellar_inputs, MC_inputs
The file Venus_outputs has dimensions ITERATIONS, where each iteration contains a python class with outputs from "Main_code_callable.py":
total_time, total_y, FH2O_array, FCO2_array, MH2O_liq, MH2O_crystal, MCO2_liq, Pressre_H2O, CO2_Pressure_array, fO2_array, Mass_O_atm, Mass_O_dissolved, water_frac, Ocean_depth, Max_depth, Ocean_fraction

%% Plot_MC_output.py
This script loads the outputs from "MC_calc.py", specifically "Venus_outputs_revisions.npy" and corresponding input parameters "Venus_inputs_revisions.npy". Within the function "use_one_output", the user can choose which outputs to plot e.g. all successful outputs, habitable Venus outputs, never habitable Venus outputs, all Venus outputs. Change selected outputs by commenting out unwanted outputs.

%% Main_code_callable.py
This python script contains the forward model. Typically, it should not need to be altered for reproducing nominal results from the main text. Various sensitivity tests require modification of Main_code_callable, however.

%% radiative_functions.py
This script contains the functions for interpolating the pre-computed Outgoing Longwave Radiation (OLR) grid, the atmosphere-ocean partitioning grid, and the stratospheric water vapor grid. Two different versions of the grids are available. For stratospheric temperature sensitivity tests, use the following:
"OLR_200_FIX_flat.npy", "Atmo_frac_200_FIX_flat.npy" and "fH2O_200_FIX_flat.npy"
For all other nominal calculations, use the following (this is the default):
"OLR_200_FIX_cold.npy", "Atmo_frac_200_FIX_cold.npy", and "fH2O_200_FIX_cold.npy"
The script radiative_functions.py also contains the function "correction", which is used to incorporate atmosphere-ocean partitioning of CO2 into the OLR calculation.

%% Albedo_module.py
Contains function for calculating bond albedo following parameterization in Pluriel et al. (2019).

%% outgassing_module_fast.py
Contains functions that call the magmatic outgassing module described in Wogan et al. (2020).

%% outgassing_module.py
Redundant - non-optimized version of outgassing functions.

%% other_functions.py.py
Contains a variety of functions including radiogenic heat production ("qr"), mantle viscosity ("viscosity_fun"), partitioning of water between magma ocean and atmosphere ("H2O_partition_function"), partitioning of CO2 between magma ocean and atmosphere ("CO2_partition_function"), magma ocean mass calculation ("Mliq_fun"), analytic calculations for the solidification radius evolution ("rs_term_fun"), mantle adiabatic temperature profile ("adiabat"), solidus calculation ("sol_liq"), solidus radius calculation ("find_r"), and the mantle melt fraction integration ("temp_meltfrac"). Both numba optimized and nominal versions of some functions are included. 

%% carbon_cycle_model.py
Contains function for calculation continental and seafloor silicate weathering fluxes.

%% all_classes.py
Defines classes used for input parameters.

%% escape_functions.py
Contains functions for atmospheric escape parameterizations. The function "better_diffusion" calculates diffusion-limited escape of water through a background gas of N2 and CO2, whereas "better_diffusion_atomic" calculates diffusion-limited escape of atomic hydrogen (H) through a background gas of O, N2 and CO2. The function "Odert_three" calculates XUV-driven escape of H given a H-O-CO2-N2 atmosphere, as described in Odert et al. (2018). Drag of O and CO2 are also computed. The function "find_epsilon" calculates escape efficiency, epsilon, as a function of the XUV flux.

%% thermodynamic_variables.py
The function "Sol_prod" calculates the temperature dependent solubility product of carbonate and calicum, Ksp(T) = [CO3(2-)][Ca(2+)] / Omega.

%% numba_nelder_mead.py
Contains numba version of the nelder_mead optimization routine.

%% stellar_funs.py
Loads stellar evolution parameterizations from Baraffe et al. ("Baraffe3.txt"), and returns total luminosity and XUV lumionsity as a function of time. See "Baraffe_readme.txt" for description of luminosity evolution parameterizations. See main text for expressions for XUV evolution relative to bolometric luminosity evolution

%% switch_garbage
This folder contains the switch parameters that keep track of whether each iteration is currently in the magma ocean or solid mantle phase. Can be emptied between model runs - is not needed after Monte Carlo outputs have been saved.

%% switch_garbage3
This folder is used to temporarily store Monte Carlo input values. It is deleted at the end of each successful execution of MC_calc.py. Therefore, if the code is interrupted before completion, switch_garbage3 must be manually deleted before MC_calc.py can be run again.

END EXPLANATION OF CODE STRUCTURE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
DESCRIPTION OF INPUT/OUTPUT VARIABLES that are used in Plot_MC_output.py

Consider the following example:
inputs = np.load('Venus_outputs_revisions.npy',allow_pickle = True) #Time-evolution of model variables
MCinputs = np.load('Venus_inputs_revisions.npy',allow_pickle = True) #Corresponding parameter values sampled in Monte Carlo analysis

inputs:
    • Total_time
        ◦ Time the model runs over [“default” for nominal Venus = 0-4.5 Gyr]
    • Total_y [time evolution of 55 explicit model variables]
        ◦ 0) Abundance of H2O in the solid interior, (kg)
        ◦ 1) Abundance of H2O in fluid phases (magma ocean plus atmosphere-ocean), (kg)
        ◦ 2) Radius of solidification, rs (m)
        ◦ 3) Abundance of free O in the solid interior, (kg)
        ◦ 4) Abundance of free O in fluid phases (magma ocean plus atmosphere-ocean) (kg)
        ◦ 5) Abundance of FeO1.5 in the solid interior (kg)
        ◦ 6) Abundance of FeO in the solid interior (kg)
        ◦ 7) Mantle potential temperature, Tp (K)
        ◦ 8) Surface temperature, Ts (K)
        ◦ 9) Outgoing longwave radiation, OLR (W/m2)
        ◦ 10) Absorbed shortwave radiation, ASR (W/m2)
        ◦ 11) Heat from interior, qmantle (W/m2)
        ◦ 12) Abundance of CO2 in fluid phases (magma ocean plus atmosphere-ocean), (kg)
        ◦ 13) Abundance of CO2 in the solid interior (kg)
        ◦ 14) Silicate weathering flux (Tmol C/yr) 
        ◦ 15) Carbon outgassing flux (Tmol C/yr)
        ◦ 16) Crustal thickness (m)
        ◦ 17) Crystal mass fraction in magma ocean
        ◦ 18) Direct (dry) oxidation of surface crust by atmospheric oxygen (kg O2/s); converted in Tmol/yr in plotting
        ◦ 19) Oxygen source flux from net escape of H (kg O2/s); converted in Tmol/yr in plotting
        ◦ 20) Wet oxidation of surface crust by serpentinizing reactions that generate H2 (kg O2/s); converted in Tmol/yr in plotting
        ◦ 21) Oxygen sink flux from outgassing of reduced species (kg O2/yr); converted in Tmol/yr in plotting
        ◦ 22) Partial pressure of O2 in atmosphere, pO2 (Pa)
        ◦ 23) Partial pressure of CO2 in atmosphere, pCO2 (Pa)
        ◦ 24) Mean molecular weight of atmosphere (kg/mol)
        ◦ 25) Global melt production (m3/s) 
        ◦ 26) Stagnant lid thickness (m)
        ◦ 27) Crustal volume (m3)
        ◦ 28) Normalized radiogenic heat production per unit volume in mantle (W/m3)
        ◦ 29) Normalized radiogenic heat production per unit volume in crust (W/m3)
        ◦ 30) Radiogenic heat production in mantle (W)
        ◦ 31) Radiogenic heat production in crust (W)
        ◦ 32) Convective heat loss from mantle to surface (W)
        ◦ 33) Advective heat loss from mantle to surface (W)
        ◦ 34) Depth at which partial melting beings (m)
        ◦ 35) 40K in mantle (kg)
        ◦ 36) 40Ar in mantle (kg)
        ◦ 37) 40K in stagnant lid (kg)
        ◦ 38) 40Ar in atmosphere (kg)
        ◦ 39) H2O in crust (kg)
        ◦ 40) CO2 in crust (kg), redundant in this version of model
        ◦ 41) Deuterium in surface reservoirs (not fully implemented)
        ◦ 42) Deuterium in interior reservoirs (not fully implemented)
        ◦ 43) 238U in mantle (kg)
        ◦ 44) 235U in mantle (kg)
        ◦ 45) 232Th in mantle (kg)
        ◦ 46) 238U in stagnant lid (kg)
        ◦ 47) 235U in stagnant lid (kg)
        ◦ 48) 232Th in stagnant lid (kg)
        ◦ 49) 4He in mantle (kg)
        ◦ 50) 4He in atmosphere (kg)
        ◦ 51) Mixing ratio of CO2 in upper atmosphere (for upper atmosphere cooling sensitivity test)
        ◦ 52) Depleted mantle fraction (for mantle depletion sensitivity test)
        ◦ 53) Water loss flux from surface reservoirs (kg/s)
        ◦ 54) Water flux surface to interior (kg/s)
        ◦ 55) Water outgassing flux (kg/s)
    • FH2O_array
        ◦ H2O mass fraction in magma ocean partial melt
    • FCO2_array
        ◦ CO2 mass fraction in magma ocean partial melt
    • MH2O_liq
        ◦ Liquid H2O mass (kg) in magma ocean
    • MH2O_crystal
        ◦ Crystal H2O mass (kg) in magma ocean
    • MCO2_liq
        ◦  Liquid CO2 mass (kg) in magma ocean
    • Pressre_H2O
        ◦ Partial pressure of H2O at surface (Pa)
    • CO2_Pressure_array
        ◦ Partial pressure of CO2 at surface (Pa)
    • fO2_array
        ◦ Oxygen fugacity (Pa)
    • Mass_O_atm
        ◦ Mass of oxygen in atmosphere (kg)
    • Mass_O_dissolved
        ◦ Mass of oxygen in melt (kg)
    • Water_frac
        ◦ Fraction of surface water in atmosphere (1-this = fraction of surface water in ocean)
    • Ocean_depth
        ◦ Depth of water ocean (m)
    • Max_depth
        ◦ Max depth of water ocean where land is permitted (m)
    • Ocean_fraction
        ◦ Approximate surface area ocean fraction (parameterized hypsometric curve)

The Monte Carlo parameter file, MCinputs, contains six classes: Switch_Inputs, Planet_inputs, Init_conditions, Numerics, Stellar_inputs, MC_inputs. Each contains a number of input parameters.

Switch_Inputs
    • print_switch
        ◦ Option for orint outputs, y/n
    • Speedup_flag
        ◦ redundant in this version of the code
    • Start_speed
        ◦ redundant in this version of the code
    • Fin_speed
        ◦ redundant in this version of the code
    • Heating_switch
        ◦ Controls locus of internal heating
    • C_cycle_switch
        ◦ Carbon cycle on/off
    • Start_time
        ◦ Calculation start time, in yrs (relative to stellar evolution track)

Planet_inputs
    • RE
        ◦ Planet radius (Earth Radii)
    • ME
        ◦ Planet mass (Earth Masses)
    • rc
        ◦ (metallic) Core radius (m)
    • pm
        ◦ ρm = Average (silicate) mantle density (kg/m^3)
    • Total_Fe_mol_fraction
        ◦ xFe = Iron mol of mantle (silicate mole fraction)
    • Planet_sep
        ◦ Planet-star distance (AU)
    • albedoC
        ◦ Cold state bond albedo 
    • albedoH
        ◦ Hot state bond albedo 

Init_conditions
    • Init_solid_H2O
        ◦ Endowment of water in solid silicate interior (kg), typically zero for magma ocean initialization
    • Init_fluid_H2O
        ◦ Endowment of water in fluid phases (kg)
    • Init_solid_O
        ◦ Endowment of free O in silicate interior (kg), typically zero for magma ocean initialization
    • Init_fluid_O     
        ◦ Endowment of free O in fluid phases (kg)
    • Init_solid_FeO1_5
        ◦ Endowment of FeO1_5 in mantle (kg)
    • Init_solid_FeO
        ◦ Endowment of FeO in mantle (kg)
    • Init_solid_CO2
        ◦ Endowment of CO2 in solid silicate interior (kg), typically zero for magma ocean initialization
    • Init_fluid_CO2
        ◦ Endowment of CO2 in fluid phases (kg)
    
Numerics #This is for choosing different maximum timesteps for different portions of the model to minimize numerical failures.
    • Total_steps
        ◦ Sets the total number of time-step divisons in each model run
    • Step0
        ◦ Maximum time step for zeroth interval (yrs)
    • Step1
        ◦ Maximum time step for first interval (yrs)
    • Step2
        ◦ Maximum time step for second interval (yrs)
    • Step3
        ◦ Maximum time step for third interval (yrs)
    • Step4
        ◦ Maximum time step for fourth interval (yrs), redundant in nominal model
    • Tfin0
        ◦ End time for zeroth interval (yrs)
    • Tfin1
        ◦ End time for first interval (yrs)
    • Tfin2
        ◦ End time for second interval (yrs)
    • Tfin3
        ◦ End time for third interval (yrs)
    • Tfin4
        ◦ End time for fourth interval (yrs), redundant in nominal model

Stellar_inputs
    • tsat_XUV
        ◦ tsat = XUV saturation time (Myr) (see equation 10 in supplemental)
    • Stellar_Mass
        ◦ Mass of star (solar masses)
    • Fsat
        ◦ Saturation of star’s XUV luminosity (see equation 11 in supplemental)
    • Beta0
        ◦ β = Decay exponent to ensure modern solar XUV flux
    • Epsilon
        ◦ εlowXUV, low XUV flux escape efficiency 

MC_inputs
    • Esc_a
        ◦ = Impactor flux coefficient (kg/yr)
    • Esc_b
        ◦ tdecay = Impactor flux decay time (Gyr)
    • Esc_c
        ◦ λtra = Transition abundance coefficient (diffusion-limited H escape for low stratospheric abundances to XUV-driven escape as upper atmosphere becomes steam dominated)
    • Esc_d
        ◦ ζ = Efficiency factor for O-drag during hydrodynamic escape 
    • Ccycle_a
        ◦ Temperature dependence of continental weathering, Te (K) (e-folding temperature)
    • Ccycle_b
        ◦  CO2 dependence of continental weathering
    • Supp_lim
        ◦ Supply limit of continental weathering (kg/s) 
    • Interiora
        ◦ Vcoef = Mantle viscosity scalar (Pa s)
    • Interiorb
        ◦ frhydr-frac = Hydration efficiency 
    • Interiorc
        ◦ fdry-oxid = Dry oxidation efficiency 
    • Interiord
        ◦ fwet-oxid = Wet oxidation efficiency 
    • Interiore
        ◦ Radiogenic inventory relative to Earth 
    • Interiorf
        ◦ Msolid-H2O-max = Max mantle in solid water (Earth oceans)
    • Interiorg
        ◦ Transition time from plate tectonics to stagnant lid (yrs after magma ocean solidification)
    • Ocean_a
        ◦ Ocean calcium concentration, [Ca2+]
    • Ocean_b
        ◦ Ω = Ocean saturation state
    • K_over_U
        ◦ K_over_U = Observed silicate K/U ratio
    • Tstrat
        ◦ Tstrat = Stratospheric temperature [K]
    • Surface_magma_frac
        ◦ flava = Max molten surface area fraction

END EXPLANATION OF INPUT/OUTPUT VARIABLES 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

-------------------------
Contact e-mail: jkt@ucsc.edu
