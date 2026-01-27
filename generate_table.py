from su2dataminer.config import Config_NICFD
from su2dataminer.generate_data import DataGenerator_CoolProp
from su2dataminer.manifold import SU2TableGenerator_NICFD

# Define SU2 DataMiner configuration for NICFD problems
config = Config_NICFD()
# Calculate the fluid properties of Siloxane MM with the Helmholtz equation of state
config.SetEquationOfState("HEOS")
config.SetFluid("MM")

# Generate fluid data for densities between 0.01 and 300 kg/m3 and for static 
# energy values between 2e5 and 5e5 J/kg
config.UsePTGrid(False)
config.SetDensityBounds(0.01, 150)
config.SetEnergyBounds(2e5, 5e5)

# Data set resolution (does not affect table resolution)
config.SetNpDensity(400)
config.SetNpEnergy(200)
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