from su2dataminer.config import Config_NICFD
from su2dataminer.generate_data import DataGenerator_CoolProp
from su2dataminer.manifold import SU2TableGenerator_NICFD

# Define SU2 DataMiner configuration for NICFD problems
config = Config_NICFD()
# Select the equation of state and the fluid
config.SetEquationOfState("REFPROP") #Available CoolProp with "HEOS" or REFPROP with "REFPROP"
config.SetFluid("MM")

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
    config.SetNpDensity(800)
    config.SetNpEnergy(400)

    if not getattr(config, "_Config_NICFD__use_auto_range"):
        
        config.SetDensityBounds(5.24,312)
        config.SetEnergyBounds(207.5e3, 371e3)

config.SaveConfig()

# Generate and save fluid data
dgen = DataGenerator_CoolProp(config)
dgen.PreprocessData()
dgen.ComputeData()
dgen.SaveData()

# Initiate table generator
lut = SU2TableGenerator_NICFD(config)

# Apply table refinement where the speed of sound is low (near the critical point)
# and at low density, where the fluid is close to an ideal gas
lut.AddRefinementCriterion("c2", norm_val_max=0.01, norm_val_min=0.0)
lut.AddRefinementCriterion("Density", norm_val_max=0.15, norm_val_min=0.0)

# Generate and save table
lut.GenerateTable()
lut.WriteTableFile("LUT_test.drg")