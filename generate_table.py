from su2dataminer.config import Config_NICFD
from su2dataminer.generate_data import DataGenerator_CoolProp
from su2dataminer.manifold import SU2TableGenerator_NICFD

# Define SU2 DataMiner configuration for NICFD problems
config = Config_NICFD()
# Select the equation of state and the fluid
config.SetEquationOfState("REFPROP") # Available CoolProp with "HEOS" or REFPROP with "REFPROP"
config.SetFluid("MM")

PlotFolder="CompData_Plots" # Folder where the plots are saved
PlotFolderLuT="LuTPlots"

PlotCompData=True # If True plot the data computed by the DataMining operation
PlotLuTData=True # If True plot the data saved in the LuT

outpath="Lut_Data.vtk" # Name of the file where the LuT are saved to be opened by Paraview

"""
Variables that can be printed are (in P-s diagram):
Density, Energy, T, c2, X, dpdrho_e, dpde_rho, cp
"""
Variables=["Density","Energy", "T", "c2", "X", "dpdrho_e", "dpde_rho", "cp"]
Unit=["kg/m3", "J/kg", "K", "m/s", "-", "J/kg", "kg/m3", "J/(kgK)"]

"""
Variables that can be printed (in LuT) are (in P-s diagram):
Density, Energy, T, c2, X, dpdrho_e, dpde_rho, cp, h
"""
Variables_LuT=["Density","Energy", "T", "c2", "X", "dpdrho_e", "dpde_rho", "cp", "Enthalpy"]
Unit_LuT=["kg/m3", "J/kg", "K", "m/s", "-", "J/kg", "kg/m3", "J/(kgK)", "J/kg"]

# Configure the LuT Creation
config.UsePTGrid(False) # If True use P-T grid, if False use rho-e grid
config.UseAutoRange(False) # If True all the thermodynamic space modeled by the thermodynamic library is reproduced in the LUT

# Select the right input based on the values selected in UsePTGrid and UseAutoRange
if getattr(config, "_Config_NICFD__use_PT"): 

    # Data set resolution (does not affect table resolution)
    config.SetNpPressure(400)
    config.SetNpTemp(200)

    if not getattr(config, "_Config_NICFD__use_auto_range"):
        
        config.SetPressureBounds(1e5,21e5)
        config.SetTemperatureBounds(373.15, 523.15)

else:

    # Data set resolution (does not affect table resolution)
    config.SetNpDensity(400)
    config.SetNpEnergy(300)

    if not getattr(config, "_Config_NICFD__use_auto_range"):
        
        config.SetDensityBounds(1,500)
        config.SetEnergyBounds(200e3, 500e3)

config.SaveConfig()

# Generate and save fluid data
dgen = DataGenerator_CoolProp(config)
dgen.PreprocessData()
dgen.ComputeData()

if PlotCompData:
    dgen.PlotContours(config, Variables,Unit,PlotFolder)

dgen.SaveData()

# Initiate table generator
lut = SU2TableGenerator_NICFD(config)

# Apply table refinement where the speed of sound is low (near the critical point)
# and at low density, where the fluid is close to an ideal gas
lut.AddRefinementCriterion("c2", norm_val_max=0.3, norm_val_min=0.0)
lut.AddRefinementCriterion("Density", norm_val_max=0.5, norm_val_min=0.0)

# Generate and save table
lut.GenerateTable()

if PlotLuTData:
    lut.PlotContoursLuT(config, Variables_LuT, Unit_LuT, PlotFolderLuT)

lut.WriteTableFile("LUT_test.drg")

if getattr(config, "_Config_NICFD__use_PT"):
    x_vars="p"
    y_vars="T"

else:
    x_vars="Density"
    y_vars="Energy"

lut.WriteOutParaview(lut._table_connectivity, lut._table_nodes, outpath, x_vars, y_vars, variables=None)