#!/usr/bin/env python3

import os 
from su2dataminer.manifold import Train_Entropic_PINN,TrainMLP_NICFD
from su2dataminer.generate_data import DataGenerator_CoolProp
from Common.DataDrivenConfig import Config_NICFD 

C = Config_NICFD()
C.SetFluid("MM")
C.SetEquationOfState("HEOS")
C.UseAutoRange(True)
C.UsePTGrid(False)
C.SetNpDensity(100)
C.SetNpEnergy(50)
C.SetAlphaExpo(-1.8269e+00)
C.SetLRDecay(+9.8959e-01)
C.SetOutputDir(os.getcwd()+"/")
C.SetStateVars(["s","T","p","c2"])
C.SetHiddenLayerArchitecture([10])
C.SetActivationFunction("exponential")
C.SetBatchExpo(4)
C.SaveConfig()

D = DataGenerator_CoolProp(C)
D.PreprocessData()
D.ComputeData()
D.SaveData()

M = TrainMLP_NICFD(C)
M.SetTrainHardware("CPU",0)
M.SetNEpochs(100)
M.SetScaler("minmax")
M.CommenceTraining()
C.UpdateMLPHyperParams(M)
C.WriteSU2MLP("SU2_MLP")
