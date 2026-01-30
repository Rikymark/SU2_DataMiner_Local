###############################################################################################
#       #      _____ __  _____      ____        __        __  ____                   #        #  
#       #     / ___// / / /__ \    / __ \____ _/ /_____ _/  |/  (_)___  ___  _____   #        #  
#       #     \__ \/ / / /__/ /   / / / / __ `/ __/ __ `/ /|_/ / / __ \/ _ \/ ___/   #        #      
#       #    ___/ / /_/ // __/   / /_/ / /_/ / /_/ /_/ / /  / / / / / /  __/ /       #        #  
#       #   /____/\____//____/  /_____/\__,_/\__/\__,_/_/  /_/_/_/ /_/\___/_/        #        #
#       #                                                                            #        #
###############################################################################################

############################ FILE NAME: DataGenerator_NICFD.py ################################
#=============================================================================================#
# author: Evert Bunschoten                                                                    |
#    :PhD Candidate ,                                                                         |
#    :Flight Power and Propulsion                                                             |
#    :TU Delft,                                                                               |
#    :The Netherlands                                                                         |
#                                                                                             |
#                                                                                             |
# Description:                                                                                |
#  Class for generating fluid data for NI-CFD data mining operations.                         |                                                               
#                                                                                             |  
# Version: 3.0.0                                                                              |
#                                                                                             |
#=============================================================================================#

#---------------------------------------------------------------------------------------------#
# Importing general packages
#---------------------------------------------------------------------------------------------#
import CoolProp.CoolProp as CP
import CoolProp as CoolP
import numpy as np 
from tqdm import tqdm
import csv 
import matplotlib.pyplot as plt 
import os
np.random.seed(2)

#---------------------------------------------------------------------------------------------#
# Importing DataMiner classes and functions
#---------------------------------------------------------------------------------------------#
from Common.DataDrivenConfig import Config_NICFD
from Common.Properties import DefaultSettings_NICFD, EntropicVars
from Data_Generation.DataGenerator_Base import DataGenerator_Base


class DataGenerator_CoolProp(DataGenerator_Base):
    """Class for generating fluid data using CoolProp
    """
    _Config:Config_NICFD
    fluid = None 
    __accepted_phases:list[int] = [CoolP.iphase_gas, CoolP.iphase_supercritical_gas, CoolP.iphase_supercritical, CoolP.iphase_liquid, CoolP.iphase_supercritical_liquid,CoolP.iphase_twophase]
    # Pressure and temperature limits
    __use_PT:bool = DefaultSettings_NICFD.use_PT_grid
    __T_min:float = DefaultSettings_NICFD.T_min
    __T_max:float = DefaultSettings_NICFD.T_max
    __Np_Y:int = DefaultSettings_NICFD.Np_temp

    __P_min:float = DefaultSettings_NICFD.P_min
    __P_max:float = DefaultSettings_NICFD.P_max
    __Np_X:int = DefaultSettings_NICFD.Np_p

    __auto_range:bool=False
    # Density and static energy limits
    __rho_min:float = DefaultSettings_NICFD.Rho_min
    __rho_max:float = DefaultSettings_NICFD.Rho_max
    __e_min:float = DefaultSettings_NICFD.Energy_min 
    __e_max:float = DefaultSettings_NICFD.Energy_max 
    __X_grid:np.ndarray[float] = None
    __Y_grid:np.ndarray[float] = None 

    # Entropy derivatives.
    __StateVars_fluid:np.ndarray[float] = None 
    __StateVars_additional:np.ndarray[float] = None 

    __success_locations:np.ndarray[bool] = None 
    __mixture:bool = False 

    def __init__(self, Config_in:Config_NICFD=None):
        DataGenerator_Base.__init__(self, Config_in=Config_in)

        if Config_in is None:
            print("Initializing NICFD data generator with default settings.")
            self._Config = Config_NICFD()
        else:
            # Load configuration and set default properties.
            self.__use_PT = self._Config.GetPTGrid()
            if len(self._Config.GetFluidNames()) > 1:
                self.__mixture = True 

            self.fluid = CP.AbstractState(self._Config.GetEquationOfState(), self._Config.GetFluid())
            self.fluid_sCrit=CP.PropsSI("S","P",self.fluid.p_critical(),"T",self.fluid.T_critical(),self._Config.GetEquationOfState()+"::"+self._Config.GetFluid())
            self.fluid_vap = CP.AbstractState(self._Config.GetEquationOfState(), self._Config.GetFluid())
            self.fluid_liq = CP.AbstractState(self._Config.GetEquationOfState(), self._Config.GetFluid())
            self.fluid_FD = CP.AbstractState(self._Config.GetEquationOfState(), self._Config.GetFluid())
            self.__auto_range = self._Config.GetAutoRange()

            if len(self._Config.GetFluidNames()) > 1:
                mole_fractions = self._Config.GetMoleFractions()
                self.fluid.set_mole_fractions(mole_fractions)
            
            self.__use_PT = self._Config.GetPTGrid()
            P_bounds = self._Config.GetPressureBounds()
            T_bounds = self._Config.GetTemperatureBounds()
            rho_bounds = self._Config.GetDensityBounds()
            e_bounds = self._Config.GetEnergyBounds()

            self.__P_min, self.__P_max = P_bounds[0], P_bounds[1]
            self.__rho_min, self.__rho_max = rho_bounds[0], rho_bounds[1]
            self.__Np_X = self._Config.GetNpPressure()

            self.__T_min, self.__T_max = T_bounds[0], T_bounds[1]
            self.__e_min, self.__e_max = e_bounds[0], e_bounds[1]
            self.__Np_Y = self._Config.GetNpTemp()
            self.__dP_FD = self._Config.GetdPFD()

        return 
    
    def UseAutoRange(self, use_auto_range:bool=True):
        """Automatically set controlling variable ranges depending on the fluid triple point and critical point."""
        self.__auto_range = use_auto_range
        return 
    
    def PreprocessData(self):
        """Generate density and static energy grid at which to evaluate fluid properties.
        """
        if self.__auto_range:
            pmin = CP.PropsSI("PTRIPLE", self._Config.GetFluid())
            pmax = self.fluid.pmax()
            Tmin = self.fluid.Tmin()
            Tmax = self.fluid.Tmax()
            p_range = np.linspace(pmin, pmax, self.__Np_X)
            T_range = np.linspace(Tmin, Tmax, self.__Np_Y)
            pp, TT = np.meshgrid(p_range, T_range)
            dd = np.zeros(np.shape(pp))
            uu = np.zeros(np.shape(TT))
            for i in range(len(T_range)):
                for j in range(len(p_range)):
                    try:
                        self.fluid.update(CP.PT_INPUTS, pp[i,j], TT[i,j])
                        if self.fluid.phase() in self.__accepted_phases:
                            dd[i,j] = self.fluid.rhomass()
                            uu[i,j] = self.fluid.umass()
                        else:
                            dd[i,j] = float("nan")
                            uu[i,j] = float("nan")
                    except:
                        dd[i,j] = float("nan")
                        uu[i,j] = float("nan")
            idx_valid = np.invert(np.isnan(dd))
            if self.__use_PT:
                X_min, X_max = np.min(pp[idx_valid]), np.max(pp[idx_valid])
                Y_min, Y_max = np.min(TT[idx_valid]), np.max(TT[idx_valid])
                self.__P_min, self.__P_max = X_min, X_max
                self.__T_min, self.__T_max = Y_min, Y_max
            else:
                X_min, X_max = np.min(dd[idx_valid]), np.max(dd[idx_valid])
                Y_min, Y_max = np.min(uu[idx_valid]), np.max(uu[idx_valid])
                self.__rho_min, self.__rho_max = X_min, X_max
                self.__e_min, self.__e_max = Y_min, Y_max
            self.UpdateConfig()
        else:
            if self.__use_PT:
                X_min = self.__P_min
                X_max = self.__P_max
                Y_min = self.__T_min
                Y_max = self.__T_max
            else:
                X_min = self.__rho_min
                X_max = self.__rho_max
                Y_min = self.__e_min
                Y_max = self.__e_max 
        
        X_range = (X_min - X_max) * np.cos(np.linspace(0, 0.5*np.pi, self.__Np_X)) + X_max
        Y_range = np.linspace(Y_min, Y_max, self.__Np_Y)
        
        self.__X_grid, self.__Y_grid = np.meshgrid(X_range, Y_range)

        fig,ax=plt.subplots()
        Psat=np.linspace(0.5e5,CP.PropsSI("Pcrit","REFPROP::"+self._Config.GetFluid()),200)
        rhoLiq=CP.PropsSI("D","P",Psat,"Q",0,"REFPROP::"+self._Config.GetFluid())
        ULiq=CP.PropsSI("Umass","P",Psat,"Q",0,"REFPROP::"+self._Config.GetFluid())
        rhoVap=CP.PropsSI("D","P",Psat,"Q",1,"REFPROP::"+self._Config.GetFluid())
        UVap=CP.PropsSI("Umass","P",Psat,"Q",1,"REFPROP::"+self._Config.GetFluid())
        ax.plot(X_range,np.min(Y_range)*np.ones(len(X_range)),color="orange")
        ax.plot(X_range,max(Y_range)*np.ones(len(X_range)),color="orange")
        ax.plot(np.min(X_range)*np.ones(len(Y_range)),Y_range,color="orange")
        ax.plot(np.max(X_range)*np.ones(len(Y_range)),Y_range,color="orange")

        ax.plot(rhoLiq,ULiq,"k")
        ax.plot(rhoVap,UVap,"k")
        ax.set_xlabel("rho [kg/m3]")
        ax.set_ylabel("e [J/kg]")
        plt.show(block=True)
        return 
    
    def UpdateConfig(self):
        if self.__use_PT:
            self._Config.SetPressureBounds(self.__P_min, self.__P_max)
            self._Config.SetTemperatureBounds(self.__T_min, self.__T_max)
        else:
            self._Config.SetDensityBounds(self.__rho_min, self.__rho_max)
            self._Config.SetEnergyBounds(self.__e_min, self.__e_max)
        self._Config.SaveConfig()
        return 
    
    def VisualizeDataGrid(self):
        """Visualize query points at which fluid data are evaluated.
        """
        fig = plt.figure(figsize=[10,10])
        ax = plt.axes()
        ax.plot(self.__X_grid.flatten(), self.__Y_grid.flatten(), 'k.')
        if self. __use_PT:
            ax.set_xlabel(r"Pressure $(p)[Pa]",fontsize=20)
            ax.set_ylabel(r"Temperature $(T)[K]",fontsize=20)
        else:
            ax.set_xlabel(r"Density $(\rho)[kg m^{-3}]$",fontsize=20)
            ax.set_ylabel(r"Static energy $(e)[J kg^{-1}]$",fontsize=20)
        ax.tick_params(which='both',labelsize=18)
        ax.grid()
        
        plt.show()
        return 
    
    def SetTemperatureBounds(self, T_lower:float=DefaultSettings_NICFD.T_min, T_upper:float=DefaultSettings_NICFD.T_max):
        """Set the upper and lower temperature limits for the fluid data grid.

        :param T_lower: lower temperature limit in Kelvin.
        :type T_lower: float
        :param T_upper: upper temperature limit in Kelvin.
        :type T_upper: float
        :raises Exception: if lower temperature limit exceeds upper temperature limit.
        """
        if (T_lower >= T_upper):
            raise Exception("Lower temperature should be lower than upper temperature.")
        else:
            self.__T_min = T_lower
            self.__T_max = T_upper
        return
    
    def GetTemperatureBounds(self):
        """Get fluid temperature limits.

        :return: list with minimum and maximum temperature.
        :rtype: list[float]
        """
        return [self.__T_min, self.__T_max]
    
    def SetNpDensity(self, Np_density:int=DefaultSettings_NICFD.Np_p):
        self.SetNpPressure(Np_P=Np_density)
        return 

    def GetNpDensity(self):
        return self.GetNpPressure()
    
    def SetDensityBounds(self, Density_lower:float=DefaultSettings_NICFD.Rho_min, Density_upper:float=DefaultSettings_NICFD.Rho_max):
        self.__rho_min = Density_lower
        self.__rho_max = Density_upper
        return 
    
    def SetNpEnergy(self, Np_energy:int=DefaultSettings_NICFD.Np_temp):
        self.SetNpTemp(Np_Temp=Np_energy)
        return 
    
    def GetNpEnergy(self):
        return self.GetNpTemp()
    
    def SetEnergyBounds(self, Energy_lower:float=DefaultSettings_NICFD.Energy_min, Energy_upper:float=DefaultSettings_NICFD.Energy_max):
        self.__e_min = Energy_lower
        self.__e_max = Energy_upper
        return 
    
    
    def SetNpTemp(self, Np_Temp:int=DefaultSettings_NICFD.Np_temp):
        """
        Set number of divisions for the temperature grid.

        :param Np_Temp: Number of divisions for the temperature range.
        :type Np_Temp: int
        :rase: Exception: If the number of divisions is lower than one.

        """
        if (Np_Temp <= 0):
            raise Exception("Number of unburnt temperature samples should be higher than one.")
        else:
            self.__Np_Y = Np_Temp
        return 
    
    def GetNpTemp(self):
        """
        Get the number of divisions for the fluid temperature range.

        :return: Number of divisions for the fluid temperature range.
        :rtype: int

        """
        return self.__Np_Y
    

    
    def SetNpPressure(self, Np_P:int=DefaultSettings_NICFD.Np_p):
        """
        Set number of divisions for the fluid pressure grid.

        :param Np_Temp: Number of divisions for the fluid pressure.
        :type Np_Temp: int
        :rase: Exception: If the number of divisions is lower than one.

        """
        if (Np_P <= 0):
            raise Exception("Number of unburnt temperature samples should be higher than one.")
        else:
            self.__Np_X = Np_P 
        return 
    
    def GetNpPressure(self):
        return self.__Np_X
    
    def SetPressureBounds(self, P_lower:float=DefaultSettings_NICFD.P_min, P_upper:float=DefaultSettings_NICFD.P_max):
        """Set the upper and lower limits for the fluid pressure.

        :param P_lower: lower pressure limit in Pa.
        :type P_lower: float
        :param P_upper: upper pressure limit in Pa.
        :type P_upper: float
        :raises Exception: if lower pressure limit exceeds upper pressure limit.
        """
        if (P_lower >= P_upper):
            raise Exception("Lower temperature should be lower than upper temperature.")
        else:
            self.__P_min = P_lower
            self.__P_max = P_upper
        return 
    
    def GetPressureBounds(self):
        """Get minimum and maximum pressure.

        :return: list with minimum and maximum fluid pressure.
        :rtype: list[float]
        """
        return [self.__P_min, self.__P_max]
    

    def ComputeData(self):
        super().ComputeData()
        """Evaluate fluid properties on the density-energy grid or pressure-temperature grid.
        """

        # Initiate empty fluid property arrays.
        
        self.__StateVars_fluid = np.zeros([self.__Np_X, self.__Np_Y, EntropicVars.N_STATE_VARS.value])
        self.__success_locations = np.ones([self.__Np_X, self.__Np_Y],dtype=bool)
        
        # Loop over density-based or pressure-based grid.
        for i in tqdm(range(self.__Np_X)):
            for j in tqdm(range(self.__Np_Y)):
                try:
                    if self.__use_PT:
                        self.fluid.update(CP.PT_INPUTS, self.__X_grid[i,j], self.__Y_grid[i,j])
                    else:
                        self.fluid.update(CP.DmassUmass_INPUTS, self.__X_grid[j,i], self.__Y_grid[j,i])
                        #p = self.fluid.p()
                        #T = self.fluid.T()
                        #self.fluid.update(CP.PT_INPUTS, p, T)
                    # Check if fluid phase is not vapor or liquid
                    self.__StateVars_fluid[i,j,:], self.__success_locations[i,j] = self.GetStateVector()
                except:
                    self.__success_locations[i,j] = False 
                    self.__StateVars_fluid[i, j, :] = None
        
        # self.__AddIdealGasData()

        return 
    
    def var_index(self, var: "EntropicVars | str") -> int:
        if isinstance(var, str):
            var = EntropicVars[var]  
        return var.value


    def PlotContours(self, config, Variables, Unit, PlotFolder):

        if os.path.isdir(PlotFolder) is False:
            os.mkdir(PlotFolder)

        EOS=config._Config_NICFD__EOS_type
        fluid=config._Config_NICFD__fluid_names[0]

        iP=self.var_index("p")
        iS=self.var_index("s")
                          
        #rhoMin=config._Config_NICFD__Rho_lower
        #rhoMax=config._Config_NICFD__Rho_upper

        sMin=np.nanmin(self._DataGenerator_CoolProp__StateVars_fluid[:,:,iS]*1e-3)
        sMax=np.nanmax(self._DataGenerator_CoolProp__StateVars_fluid[:,:,iS]*1e-3)

        PMin=np.nanmin(self._DataGenerator_CoolProp__StateVars_fluid[:,:,iP]*1e-5)
        PMax=np.nanmax(self._DataGenerator_CoolProp__StateVars_fluid[:,:,iP]*1e-5)

        #eMin=config._Config_NICFD__Energy_lower
        #eMax=config._Config_NICFD__Energy_upper

        Psat=np.linspace(CP.PropsSI("PTRIPLE",EOS+"::"+fluid), CP.PropsSI("PCRIT",EOS+"::"+fluid),2000)
        sLiq=CP.PropsSI("S","P",Psat,"Q",0,EOS+"::"+fluid)
        sVap=CP.PropsSI("S","P",Psat,"Q",1,EOS+"::"+fluid)

        

        #rhoLiq=CP.PropsSI("D","P",Psat,"Q",0,EOS+"::"+fluid)
        #rhoVap=CP.PropsSI("D","P",Psat,"Q",1,EOS+"::"+fluid)
        #eLiq=CP.PropsSI("Umass","P",Psat,"Q",0,EOS+"::"+fluid)
        #eVap=CP.PropsSI("Umass","P",Psat,"Q",1,EOS+"::"+fluid)

        for i in range(len(Variables)):
            ivar=self.var_index(Variables[i])

            fig,ax=plt.subplots()
            # Plot saturation dome
            #ax.plot(rhoLiq,eLiq*1e-3,"-k")
            #ax.plot(rhoVap,eVap*1e-3,"-k")
            ax.plot(sLiq*1e-3,Psat*1e-5,"-k")
            ax.plot(sVap*1e-3,Psat*1e-5,"-k")

            # Contour
            #X_Data=self._DataGenerator_CoolProp__X_grid
            #Y_Data=self._DataGenerator_CoolProp__Y_grid

            X_Data=self._DataGenerator_CoolProp__StateVars_fluid[:,:,iS]*1e-3
            Y_Data=self._DataGenerator_CoolProp__StateVars_fluid[:,:,iP]*1e-5

            if Variables[i]!="c2":
                Z_Data=self._DataGenerator_CoolProp__StateVars_fluid[:,:,ivar]
            else:
                Z_Data=self._DataGenerator_CoolProp__StateVars_fluid[:,:,ivar]**0.5

            #print(np.nanmin(Z_Data))
            #print(np.nanmax(Z_Data))
            cont=ax.contourf(X_Data,Y_Data,Z_Data,vmin=np.nanmin(Z_Data),vmax=np.nanmax(Z_Data))

            if Variables[i]!="c2":
                plt.colorbar(cont,ax=ax,location='right',label=Variables[i]+f" [{Unit[i]}]")
            else:
                plt.colorbar(cont,ax=ax,location='right',label=f"c [{Unit[i]}]")

            #ax.set_xlabel("rho [kg/m3]")
            #ax.set_ylabel("e [kJ/kg]")
            ax.set_xlabel("s [kJ/kg]")
            ax.set_ylabel("P [bar]")

            #ax.set_xlim(rhoMin,rhoMax)
            #ax.set_ylim(eMin*1e-3,eMax*1e-3)

            ax.set_xlim(sMin,sMax)
            ax.set_ylim(PMin,PMax)

            ax.grid(ls=":", c="lightgray")

            plt.show(block=True)

            fig.savefig(f"{PlotFolder}/P-s_diagram+{Variables[i]}_contour.pdf", dpi=600)
            fig.savefig(f"{PlotFolder}/P-s_diagram+{Variables[i]}_contour.svg", dpi=600)

        return

    def __AddIdealGasData(self):
        state_flattened = np.vstack(self.__StateVars_fluid)[self.__success_locations.flatten(), :]
        rho_data = state_flattened[:, EntropicVars.Density.value]
        e_data = state_flattened[:, EntropicVars.Energy.value]
        p_data = state_flattened[:, EntropicVars.p.value]
        T_data = state_flattened[:, EntropicVars.T.value]
        


        R_gas = self.fluid.gas_constant()/self.fluid.molar_mass()
        compressibility_factor = p_data / (R_gas * rho_data * T_data)
        idealgas_loc = compressibility_factor > 0.9

        rho_min, rho_max = min(rho_data), max(rho_data)
        e_min, e_max = min(e_data), max(e_data)
        
        Np = 10

        rho_idealgas = rho_data[idealgas_loc]
        e_idealgas = e_data[idealgas_loc]

        self.__StateVars_additional = np.zeros([len(rho_idealgas), EntropicVars.N_STATE_VARS.value])
        success_locations = np.ones(len(rho_idealgas),dtype=bool)
        for i in tqdm(range(len(rho_idealgas))):
            for _ in range(Np):
                try:
                    theta = 2*np.pi*np.random.rand()
                    radius = 0.05*np.random.rand()
                    rho_delta = rho_idealgas[i] + radius * (rho_max - rho_min)*np.cos(theta)
                    e_delta = e_idealgas[i] + radius*(e_max - e_min)*np.sin(theta)
                    rho_delta = max(rho_min, min(rho_max, rho_delta))
                    e_delta = max(e_min, min(e_max, e_delta))
                    
                    self.fluid.update(CP.DmassUmass_INPUTS, rho_delta, e_delta)
                    self.__StateVars_additional[i, :], success_locations[i] = self.__GetStateVector()
                except:
                    self.__StateVars_additional[i, :] = None
                    success_locations[i] = False
        self.__StateVars_additional = self.__StateVars_additional[success_locations, :]
        
        return 
    
    def GetStateVector(self):
        state_vector_vals = np.ones(EntropicVars.N_STATE_VARS.value)
        correct_phase = True 
        if self.fluid.phase() in self.__accepted_phases:
            state_vector_vals[EntropicVars.s.value] = self.fluid.smass()
            state_vector_vals[EntropicVars.Density.value] = self.fluid.rhomass()
            state_vector_vals[EntropicVars.Energy.value] = self.fluid.umass()
            state_vector_vals[EntropicVars.T.value] = self.fluid.T()
            state_vector_vals[EntropicVars.p.value] = self.fluid.p()
            X = self.fluid.Q()
            P=state_vector_vals[EntropicVars.p.value]

            if X<=0 or X>=1:
                state_vector_vals[EntropicVars.c2.value] = self.fluid.speed_sound()**2
                #state_vector_vals[EntropicVars.dTde_rho.value] = self.fluid.first_partial_deriv(CP.iT, CP.iUmass, CP.iDmass)
                #state_vector_vals[EntropicVars.dTdrho_e.value] = self.fluid.first_partial_deriv(CP.iT, CP.iDmass, CP.iUmass)
                state_vector_vals[EntropicVars.dpde_rho.value] = self.fluid.first_partial_deriv(CP.iP, CP.iUmass, CP.iDmass)
                state_vector_vals[EntropicVars.dpdrho_e.value] = self.fluid.first_partial_deriv(CP.iP, CP.iDmass, CP.iUmass)
                state_vector_vals[EntropicVars.cp.value] = self.fluid.cpmass()

                S=state_vector_vals[EntropicVars.s.value]
                T=state_vector_vals[EntropicVars.T.value]
                if S>=self.fluid_sCrit:
                    state_vector_vals[EntropicVars.X.value]=1
                elif P>=self.fluid.p_critical():
                    state_vector_vals[EntropicVars.X.value]=0
                else:
                    self.fluid_vap.update(CoolP.QT_INPUTS, 1, T)
                    Pvap=self.fluid_vap.p()
                    if P<=Pvap:
                        state_vector_vals[EntropicVars.X.value]=1
                    else:
                        state_vector_vals[EntropicVars.X.value]=0
            
            else:
                state_vector_vals[EntropicVars.X.value]=X
                #print(state_vector_vals[EntropicVars.X.value])
                dP_FD=self.__dP_FD

                #SOS^2
                self.fluid_FD.update(CoolP.PSmass_INPUTS, P+dP_FD, self.fluid.smass())
                state_vector_vals[EntropicVars.c2.value]=dP_FD/(self.fluid_FD.rhomass()-self.fluid.rhomass())

                #dpde_rho
                self.fluid_FD.update(CoolP.DmassP_INPUTS, self.fluid.rhomass(), P+dP_FD)
                state_vector_vals[EntropicVars.dpde_rho.value]=dP_FD/(self.fluid_FD.umass()-self.fluid.umass())

                #dpdrho_e
                self.fluid_FD.update(CoolP.PUmass_INPUTS, P+dP_FD, self.fluid.umass())
                state_vector_vals[EntropicVars.dpdrho_e.value]=dP_FD/(self.fluid_FD.rhomass()-self.fluid.rhomass())

                #Cp
                self.fluid_vap.update(CoolP.PQ_INPUTS, P, 1)
                self.fluid_liq.update(CoolP.PQ_INPUTS, P, 1)
                alpha=X*self.fluid.rhomass()/self.fluid_vap.rhomass()
                state_vector_vals[EntropicVars.cp.value] =alpha*self.fluid_vap.cpmass()+(1-alpha)*self.fluid_liq.cpmass()
                        
        else:
            correct_phase = False
            state_vector_vals[:] = None 
        return state_vector_vals, correct_phase
    
    def UpdateFluid(self, val_rho, val_e):
        self.fluid.update(CP.DmassUmass_INPUTS, val_rho, val_e)
        return 
    
    def VisualizeFluidData(self):
        """Visualize computed fluid data.
        """

        fig = plt.figure(figsize=[27, 9])
        ax0 = fig.add_subplot(1, 3, 1, projection='3d')
        ax0.plot_surface(self.__StateVars_fluid[:, :, EntropicVars.Density.value],\
                         self.__StateVars_fluid[:, :, EntropicVars.Energy.value],\
                         self.__StateVars_fluid[:, :, EntropicVars.p.value])
        ax0.set_xlabel("Density [kg/m3]",fontsize=20)
        ax0.set_ylabel("Static Energy [J/kg]",fontsize=20)
        ax0.set_zlabel("Pressure [Pa]",fontsize=20)
        ax0.tick_params(which='both',labelsize=18)
        ax0.grid()
        ax0.set_title("Fluid pressure data",fontsize=22)

        ax1 = fig.add_subplot(1, 3, 2, projection='3d')
        ax1.plot_surface(self.__StateVars_fluid[:, :, EntropicVars.Density.value],\
                         self.__StateVars_fluid[:, :, EntropicVars.Energy.value],\
                         self.__StateVars_fluid[:, :, EntropicVars.T.value])
        ax1.set_xlabel("Density [kg/m3]",fontsize=20)
        ax1.set_ylabel("Static Energy [J/kg]",fontsize=20)
        ax1.set_zlabel("Temperature [K]",fontsize=20)
        ax1.tick_params(which='both',labelsize=18)
        ax1.grid()
        ax1.set_title("Fluid temperature data",fontsize=22)

        ax2 = fig.add_subplot(1, 3, 3, projection='3d')
        ax2.plot_surface(self.__StateVars_fluid[:, :, EntropicVars.Density.value],\
                         self.__StateVars_fluid[:, :, EntropicVars.Energy.value],\
                         np.sqrt(self.__StateVars_fluid[:, :, EntropicVars.c2.value]))
        ax2.set_xlabel("Density [kg/m3]",fontsize=20)
        ax2.set_ylabel("Static Energy [J/kg]",fontsize=20)
        ax2.set_zlabel("Speed of sound [m/s]",fontsize=20)
        ax2.tick_params(which='both',labelsize=18)
        ax2.grid()
        ax2.set_title("Fluid speed of sound data",fontsize=22)

        plt.show()
        return 
    
    def SaveData(self):
        """Save fluid data in separate files for train, test and validation.
        """

        # Define output files for all, train, test, and validation data.
        full_file = self.GetOutputDir() + "/" + self.GetConcatenationFileHeader() + "_full.csv"
        train_file = self.GetOutputDir() + "/" + self.GetConcatenationFileHeader() + "_train.csv"
        test_file = self.GetOutputDir() + "/" + self.GetConcatenationFileHeader() + "_test.csv"
        val_file = self.GetOutputDir() + "/" + self.GetConcatenationFileHeader() + "_val.csv"

        # Append controlling and training variable data.
        controlling_vars = [EntropicVars.Density, EntropicVars.Energy]
        if self.__mixture:
            entropic_vars=[EntropicVars.s]
        else:
            #entropic_vars = [EntropicVars.s, EntropicVars.dsdrho_e, EntropicVars.dsde_rho, EntropicVars.d2sdrho2, EntropicVars.d2sdedrho, EntropicVars.d2sde2]
            entropic_vars = [EntropicVars.s]
        
        TD_vars = [EntropicVars.T, EntropicVars.p, EntropicVars.c2]

        """
        secondary_vars = [EntropicVars.dTdrho_e, EntropicVars.dTde_rho, EntropicVars.dpdrho_e, EntropicVars.dpde_rho,\
                          EntropicVars.dhdrho_e, EntropicVars.dhde_rho, EntropicVars.dhdrho_p, EntropicVars.dhdp_rho,\
                          EntropicVars.dsdp_rho, EntropicVars.dsdrho_p,EntropicVars.cp]
        """
        secondary_vars = [EntropicVars.dpdrho_e, EntropicVars.dpde_rho, EntropicVars.cp]
        all_vars = controlling_vars + entropic_vars + TD_vars + secondary_vars

        
        CV_data = np.vstack(self.__StateVars_fluid[:, :, [v.value for v in controlling_vars]])
        entropic_data = np.vstack(self.__StateVars_fluid[:, :, [v.value for v in entropic_vars]])
        secondary_data = np.vstack(self.__StateVars_fluid[:, :, [v.value for v in secondary_vars]])
        TD_data = np.vstack(self.__StateVars_fluid[:, :, [v.value for v in TD_vars]])

        full_data = np.hstack((CV_data, entropic_data, TD_data, secondary_data))
        full_data = full_data[self.__success_locations.flatten(), :]
  
        # remove inf values
        full_data = full_data[~np.isinf(full_data).any(axis=1), :]
        full_data = full_data[~np.isnan(full_data).any(axis=1), :]
        # Shuffle data array.
        #np.random.shuffle(full_data)
        full_data_n = full_data.copy()
        np.random.shuffle(full_data_n)
        # print(",".join("%+.16e" % f for f in full_data[0,:]))
        # print(np.shape(full_data))
        # Define number of training and test data points.
        Np_full = np.shape(full_data_n)[0]
        Np_train = int(self.GetTrainFraction()*Np_full)
        Np_test = int(self.GetTestFraction()*Np_full)

        train_data = full_data_n[:Np_train, :]
        test_data = full_data_n[Np_train:Np_train+Np_test, :]
        val_data = full_data_n[Np_train+Np_test:, :]
        # Write output data files.
        with open(full_file,"w+") as fid:
            fid.write(",".join(v.name for v in all_vars) + "\n")
            csvWriter = csv.writer(fid)
            csvWriter.writerows(full_data)
        
        with open(train_file,"w+") as fid:
            fid.write(",".join(v.name for v in all_vars) + "\n")
            csvWriter = csv.writer(fid)
            csvWriter.writerows(train_data)

        with open(test_file,"w+") as fid:
            fid.write(",".join(v.name for v in all_vars) + "\n")
            csvWriter = csv.writer(fid)
            csvWriter.writerows(test_data)

        with open(val_file,"w+") as fid:
            fid.write(",".join(v.name for v in all_vars) + "\n")
            csvWriter = csv.writer(fid)
            csvWriter.writerows(val_data)
            
        return 
    
    def GetStateData(self):
        return self.__StateVars_fluid, self.__success_locations
    
    def GetFluidDataGrid(self):
        return self.__X_grid, self.__Y_grid