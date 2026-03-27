from su2dataminer.config import Config_NICFD
from su2dataminer.generate_data import DataGenerator_CoolProp
from su2dataminer.manifold import SU2TableGenerator_NICFD
import os

# Define SU2 DataMiner configuration for NICFD problems
config = Config_NICFD()
# Select the equation of state and the fluid
config.SetEquationOfState("REFPROP") # Available CoolProp with "HEOS" or REFPROP with "REFPROP"
config.SetFluid("MM")

MainFolder="MM/LUT_2PH_TET4_EXP_Adapt_Ref_Add_Ref_V9" # folder where all the data are saved
PlotFolder="CompPlot_2PH_TET4_EXP_Adapt_Ref_Add_Ref_V9" # Folder where the fluid data plots are saved
PlotFolderLuT="CompLuT_2PH_TET4_EXP_Adapt_Ref_Add_Ref_V9" # Folder where the LUT plots are saved
outpath="LUT_2PH_TET4_EXP_Adapt_Ref_Add_Ref_V9.vtk" # Name of the file where the LuT are saved to be opened by Paraview
LuTName="LUT_2PH_TET4_EXP_Adapt_Ref_Add_Ref_V9.drg"      # # Name of the file where the LuT are saved as .drg
RefFile="LuT_Ref_Data.csv"

PlotCompData=False # If True plot the data computed by the DataMining operation
PlotLuTData=True # If True plot the data saved in the LuT

"""
Variables that can be printed are (in P-s diagram):
Density, Energy, T, c2, X, dpdrho_e, dpde_rho, dhdrho_e, dhde_rho, dhdp_rho, dhdrho_p, dsdrho_e, dsde_rho, dsdp_rho, dsdrho_p, dTdrho_e, dTde_rho, cp, cv
"""
Variables=["Density","Energy", "T", "c2", "X", "dpdrho_e", "dpde_rho", "dhdrho_e", "dhde_rho", "dhdp_rho", "dhdrho_p", "dsdrho_e", "dsde_rho",\
            "dsdp_rho", "dsdrho_p", "dTdrho_e", "dTde_rho", "cp", "cv"]
Unit=["kg/m3", "J/kg", "K", "m/s", "-", "J/kg", "kg/m3", "m5/(kgs2)", "-", "m3/kg", "m5/(kgs2)", "m5/(kgs2K)", "1/K", "m3/(kgK)","m5/(kgs2K)",\
      "Kkg/J", "Km3/kg", "J/(kgK)","J/(kgK)"]

"""
Variables that can be printed (in LuT) are (in P-s diagram):
Density, Energy, T, c2, X, dpdrho_e, dpde_rho, dhdrho_e, dhde_rho, dhdp_rho, dhdrho_p, dsdrho_e, dsde_rho, dsdp_rho, dsdrho_p, dTdrho_e, dTde_rho, cp, cv, Enthalpy
"""
Variables_LuT=["Density","Energy", "T", "c2", "X", "dpdrho_e", "dpde_rho", "dhdrho_e", "dhde_rho", "dhdp_rho", "dhdrho_p", "dsdrho_e", "dsde_rho",\
                "dsdp_rho", "dsdrho_p","dTdrho_e", "dTde_rho", "cp", "cv", "Enthalpy"]
Unit_LuT=["kg/m3", "J/kg", "K", "m/s", "-", "J/kg", "kg/m3", "m5/(kgs2)", "-", "m3/kg", "m5/(kgs2)", "m5/(kgs2K)", "1/K", "m3/(kgK)","m5/(kgs2K)",\
          "Kkg/J", "Km3/kg","J/(kgK)","J/(kgK)", "J/kg"]

# Configure the LuT Creation
config.UsePTGrid(False) # If True use P-T grid, if False use rho-e grid
config.UseAutoRange(False) # If True all the thermodynamic space modeled by the thermodynamic library is reproduced in the LUT

# Select the right input based on the values selected in UsePTGrid and UseAutoRange
if config.GetPTGrid(): 

    # Data set resolution (does not affect table resolution)
    config.SetNpPressure(400)
    config.SetNpTemp(200)

    if not config.GetAutoRange():
        
        config.SetPressureBounds(1e5,21e5)
        config.SetTemperatureBounds(373.15, 523.15)

else:

    # Data set resolution (does not affect table resolution)
    config.SetNpDensity(200)
    config.SetNpEnergy(200)

    if not config.GetAutoRange():
        
        config.SetDensityBounds(0.5,450)
        config.SetEnergyBounds(200e3, 365e3)

config.SetdPFD(1) # Pa
config.SetdhFD(1) # J/kg
config.SetdrhoMultFD(1e-4) # Density_plus=Density*(1+drho_mult_FD)
config.SetMainFolder(MainFolder)

# Create the main SaveFolder
if os.path.isdir(MainFolder) is False:
    os.mkdir(MainFolder)

# Save the config file in the main folder
config.SaveConfig(MainFolder)

# Generate and save fluid data
dgen = DataGenerator_CoolProp(config)
dgen.PreprocessData()
dgen.ComputeData()

if PlotCompData:
    dgen.PlotContours(Variables,Unit,MainFolder,PlotFolder)

dgen.SaveData()

# Initiate table generator
lut = SU2TableGenerator_NICFD(config)

# Apply table refinement where the speed of sound is low (near the critical point)
# and at low density, where the fluid is close to an ideal gas
#lut.AddRefinementCriterion("c2", norm_val_max=0.25, norm_val_min=0.0)
#lut.AddRefinementCriterion("Density", norm_val_max=0.01, norm_val_min=0.0)
#lut.AddRefinementCriterion("p", norm_val_max=0.05, norm_val_min=0.0)

# Save the main inputs in a .txt file
lut.write_TxT_config(config, MainFolder)

# Generate and save table
LoadRef=f"{MainFolder}/{RefFile}"
lut.GenerateTable(LoadRef, MainFolder, config)

if PlotLuTData:
    lut.PlotContoursLuT(config, Variables_LuT, Unit_LuT, MainFolder,PlotFolderLuT)

lut.WriteTableFile(MainFolder,LuTName)

if getattr(config, "_Config_NICFD__use_PT"):
    x_vars="p"
    y_vars="T"

else:
    x_vars="Density"
    y_vars="Energy"

lut.WriteOutParaview(lut._table_connectivity, lut._table_nodes, MainFolder, outpath, x_vars, y_vars, variables=None)