#!/usr/bin/env python3

import os 
import sys
from su2dataminer.config import Config_NICFD
from su2dataminer.generate_data import DataGenerator_CoolProp 

Config = Config_NICFD()

Config.SetFluid("MM")
Config.SetEquationOfState("HEOS")
Config.UseAutoRange(True)
Config.UsePTGrid(False)
Config.SetNpDensity(50)
Config.SetNpEnergy(50)
Config.SetOutputDir(os.getcwd())

D = DataGenerator_CoolProp(Config_in=Config)

D.PreprocessData()

D.ComputeData()

D.SaveData()