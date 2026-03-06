###############################################################################################
#       #      _____ __  _____      ____        __        __  ____                   #        #
#       #     / ___// / / /__ \    / __ \____ _/ /_____ _/  |/  (_)___  ___  _____   #        #
#       #     \__ \/ / / /__/ /   / / / / __ `/ __/ __ `/ /|_/ / / __ \/ _ \/ ___/   #        #
#       #    ___/ / /_/ // __/   / /_/ / /_/ / /_/ /_/ / /  / / / / / /  __/ /       #        #
#       #   /____/\____//____/  /_____/\__,_/\__/\__,_/_/  /_/_/_/ /_/\___/_/        #        #
#       #                                                                            #        #
###############################################################################################

######################### FILE NAME: FlameletTableGenerator.py ################################
#=============================================================================================#
# author: Evert Bunschoten                                                                    |
#    :PhD Candidate ,                                                                         |
#    :Flight Power and Propulsion                                                             |
#    :TU Delft,                                                                               |
#    :The Netherlands                                                                         |
#                                                                                             |
#                                                                                             |
# Description:                                                                                |
#   Table generator class for generating SU2-supported tables of flamelet data.               |
# Version: 3.0.0                                                                              |
#                                                                                             |
#=============================================================================================#

import numpy as np 
import CoolProp.CoolProp as CP
from Common.Properties import EntropicVars
from su2dataminer.generate_data import DataGenerator_CoolProp
from scipy.spatial import ConvexHull, Delaunay
from sklearn.preprocessing import MinMaxScaler,RobustScaler,StandardScaler, QuantileTransformer
import matplotlib.pyplot as plt 
from tqdm import tqdm
import sys,os
from Common.DataDrivenConfig import Config_NICFD
import cantera as ct
import gmsh 
import pickle
from multiprocessing import Pool 
from sklearn.metrics import mean_squared_error
from Common.Interpolators import Invdisttree 
from random import sample 
from concave_hull import concave_hull, concave_hull_indexes
from scipy.interpolate import griddata
import matplotlib.tri as mtri
import meshio

class SU2TableGenerator_NICFD:

    _Config:Config_NICFD = None # Config_FGM class from which to read settings.
    _DataGenerator:DataGenerator_CoolProp = None 
    _savedir:str

    _Fluid_Variables:list[str] = None  # Variable names in the concatenated flamelet data file.

    _custom_table_limits_set:bool = False 
    _mixfrac_min_table:float = None     # Lower mixture fraction limit of the table.
    _mixfrac_max_table:float = None     # Upper mixture fraction limit of the table.

    refinement_vars = []
    refinement_norm_min = []
    refinement_norm_max = []

    _base_cell_size:float = 2e-2      # Table level base cell size.

    _refined_cell_size:float = 1.25e-3 #2.5e-3#1.5e-3   # Table level refined cell size. # old value is 5e-3, standard for adapted ref 2e-3
    _refinement_radius:float = 3.5e-2 #5e-2     # Table level radius within which refinement is applied. # original value is 1e-2

    _table_nodes = []       # Progress variable, total enthalpy, and mixture fraction node values for each table level.
    _table_nodes_norm = []  # Normalized table nodes for each level.
    _table_connectivity = []    # Table node connectivity per table level.
    _table_hullnodes = []   # Hull node indices per table level.

    _controlling_variables:list[str]=["Density",\
                                      "Energy"]  # FGM controlling variables
    _fluid_data_scaler:MinMaxScaler = None   # Scaler for flamelet data controlling variables.

    def __init__(self, Config:Config_NICFD, load_file:str=None):
        """
        Initiate table generator class.

        :param Config: Config_FGM object.
        :type Config: Config_FGM
        """
        self._Config = Config 
        self._DataGenerator = DataGenerator_CoolProp(self._Config)
        self.__LoadFluidData()
        return 
    
    def SetCellSize_Coarse(self, cell_size_coarse:float=1e-2):
        """Specify the coarse level cell size of the table

        :param cell_size_coarse: coarse cell size, defaults to 1e-2
        :type cell_size_coarse: float, optional
        :raises Exception: if specified cell size is negative or zero
        """
        if cell_size_coarse <= 0:
            raise Exception("Cell size value should be positive")
        self._base_cell_size = cell_size_coarse 
        return 
    
    def SetCellSize_Refined(self, cell_size_ref:float=5e-3):
        """Specify the refined level cell size of the table

        :param cell_size_ref: refined cell size, defaults to 1e-2
        :type cell_size_ref: float, optional
        :raises Exception: if specified cell size is negative or zero
        """
        if cell_size_ref <= 0:
            raise Exception("Cell size value should be positive")
        self._refined_cell_size = cell_size_ref 
        return 
    
    def SetRefinement_Radius(self, refinement_radius:float=1e-2):
        """Specify the radius around each refinement point within which the refined cell size is applied

        :param refinement_radius: refinement radius, defaults to 1e-2
        :type refinement_radius: float, optional
        :raises Exception: if specified value is negative or zero
        """
        if refinement_radius <= 0:
            raise Exception("Refinement radius should be positive")
        self._refinement_radius = refinement_radius
        return 
    
    def __LoadFluidData(self):
        fluid_data_file = self._Config.GetOutputDir() + "/" + self._Config.GetConcatenationFileHeader() + "_full.csv"
        with open(fluid_data_file, 'r') as fid:
            vars = fid.readline().strip().split(',')
        D = np.loadtxt(fluid_data_file,delimiter=',',skiprows=1)
        entropic_vars = [a.name for a in EntropicVars][:-1]
        self.table_vars = entropic_vars.copy()
        fluid_data_out = np.zeros([len(D), EntropicVars.N_STATE_VARS.value])
        for ivar, x in enumerate(vars):
            fluid_data_out[:, entropic_vars.index(x)] = D[:, ivar]
        self._fluid_data_scaler = MinMaxScaler()
        fluid_data_norm = self._fluid_data_scaler.fit_transform(fluid_data_out)
   
        
        return fluid_data_norm
        
    def __Compute2DMesh(self, points:np.ndarray[float], ref_pts:np.ndarray[float]=[],show:bool=False):
        
        # Create concave hull of normalized table coordinates.
        XY_hull = concave_hull(np.unique(points,axis=0), length_threshold=1e-1)
        
        # Filter concave hull to remove nodes that are too close together.
        hull_pts = []
        i = 0
        hull_indices = [i]
        while i < (len(XY_hull)-1):
            i_next = i+1
            found_next_pt = False 
            while not found_next_pt:
                dist = np.sqrt(np.sum(np.power(XY_hull[i_next, :] - XY_hull[i, :], 2)))
                if (dist >= self._base_cell_size) or (i_next == len(XY_hull)-1):
                    found_next_pt = True 
                else:
                    i_next += 1
            i = i_next
            hull_indices.append(i_next)
        XY_hull = XY_hull[hull_indices, :]

        # Initiate gmsh
        gmsh.initialize() 
        gmsh.model.add("table_level")
        factory = gmsh.model.geo

        # Create hull points
        for i in range(int(len(XY_hull))):
            hull_pts.append(factory.addPoint(XY_hull[i, 0], XY_hull[i, 1], 0, self._base_cell_size))
        
        # Connect hull points to a closed multi-component curve
        hull_lines = []
        for i in range(len(hull_pts)-1):
            hull_lines.append(factory.addLine(hull_pts[i], hull_pts[i+1]))
        hull_lines.append(factory.addLine(hull_pts[-1], hull_pts[0]))

        # Create a 2D plane of the enclosed space
        curvloop = factory.addCurveLoop(hull_lines)
        factory.addPlaneSurface([curvloop])

        # Apply refinement points
        ref_pt_ids = []
        if len(ref_pts)>0:
            for i in range(len(ref_pts)):
                ref_pt_ids.append(factory.addPoint(ref_pts[i,0], ref_pts[i, 1], 0.0))

        # Apply conditional refinement, where the refined cell size is applied in proximity to the refinement points
        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "PointsList", ref_pt_ids)
        gmsh.model.mesh.field.setNumber(1, "Sampling", 100)
        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "InField", 1)
        gmsh.model.mesh.field.setNumber(2, "SizeMin", self._refined_cell_size)
        gmsh.model.mesh.field.setNumber(2, "SizeMax", self._base_cell_size)
        gmsh.model.mesh.field.setNumber(2, "DistMin", 0.5*self._refinement_radius)
        gmsh.model.mesh.field.setNumber(2, "DistMax", 1.5*self._refinement_radius)

        gmsh.model.mesh.field.add("Min", 7)
        gmsh.model.mesh.field.setNumbers(7, "FieldsList", [2])
        gmsh.model.mesh.field.setAsBackgroundMesh(7)

        factory.synchronize()

        # Generate 2D mesh and extract table nodes
        gmsh.model.mesh.generate(2)
        if show:
            gmsh.fltk.run()
        nodes = gmsh.model.mesh.getNodes(dim=2, tag=-1, includeBoundary=True, returnParametricCoord=False)[1]
        gmsh.finalize()
        MeshPoints = np.array([nodes[::3], nodes[1::3]]).T
        return MeshPoints
    
    
    def __CalcMeshData(self, fluid_data_mesh:np.ndarray[float]):
        """Calculate the fluid thermodynamic state variables for the table nodes

        :param fluid_data_mesh: table mesh nodes of density and static energy
        :type fluid_data_mesh: np.ndarray[float]
        :return: filtered thermodynamic state data at the table nodes
        :rtype: np.ndarray[float]
        """
        fluid_data_out = fluid_data_mesh.copy()
        for i in range(len(fluid_data_mesh)):
            try:
                self._DataGenerator.UpdateFluid(fluid_data_mesh[i, EntropicVars.Density.value], fluid_data_mesh[i, EntropicVars.Energy.value])
                state_vector, correct_phase = self._DataGenerator.GetStateVector()
                if correct_phase:
                    fluid_data_out[i, :] = state_vector
                else:
                    fluid_data_out[i, :] = None
            except:
                fluid_data_out[i, :] = None
        fluid_data_out = fluid_data_out[~np.isnan(fluid_data_out[:,0]),:]
        return fluid_data_out
    
    def GenerateTable(self, LoadRef, MainFolder):
        """Initiate table generation process
        """

        # Load initial fluid data and scale it
        fluid_data_norm = self.__LoadFluidData()
        rhoe_norm = fluid_data_norm[:, [EntropicVars.Density.value, EntropicVars.Energy.value]]

        # Generate initial coarse table of fluid data
        rhoe_mesh_norm_coarse = self.__Compute2DMesh(rhoe_norm)

        # Calculate thermodynamic state variables of initial table nodes
        fluid_data_norm_coarse = np.zeros([len(rhoe_mesh_norm_coarse), EntropicVars.N_STATE_VARS.value])
        fluid_data_norm_coarse[:, EntropicVars.Density.value] = rhoe_mesh_norm_coarse[:,0]
        fluid_data_norm_coarse[:, EntropicVars.Energy.value] = rhoe_mesh_norm_coarse[:,1]
        fluid_data_coarse = self._fluid_data_scaler.inverse_transform(fluid_data_norm_coarse)
        fluid_data_coarse = self.__CalcMeshData(fluid_data_coarse)

        # Identify refinement locations
        fluid_data_norm = self._fluid_data_scaler.transform(fluid_data_coarse)
        ix_ref_user = self.__ApplyRefinement(fluid_data_norm)

        # add refinment near the expected thermodynamic path
        RefPoints=np.loadtxt(LoadRef,skiprows=1,usecols=(0,1),delimiter=",",dtype=float)
        mask = np.isfinite(RefPoints).all(axis=1) 
        RefPoints_clean=RefPoints[mask, :]  
        ix_ref_TH_transf=self.__ApplyRefinement_exp(fluid_data_coarse, RefPoints_clean)

        # Regenerate table including refinement locations
        rhoe_norm_mesh = fluid_data_norm[:, [EntropicVars.Density.value, EntropicVars.Energy.value]]
        ix_ref=np.union1d(ix_ref_user,ix_ref_TH_transf).astype(np.int64)
        rhoe_norm_ref = rhoe_norm_mesh[ix_ref, :]
        rhoe_mesh_norm = self.__Compute2DMesh(rhoe_norm, ref_pts=rhoe_norm_ref,show=True)

        # Extract thermodynamic state variables of refined table
        fluid_data_norm_ref = np.zeros([len(rhoe_mesh_norm), EntropicVars.N_STATE_VARS.value])
        fluid_data_norm_ref[:, EntropicVars.Density.value] = rhoe_mesh_norm[:,0]
        fluid_data_norm_ref[:, EntropicVars.Energy.value] = rhoe_mesh_norm[:,1]
        fluid_data_ref = self._fluid_data_scaler.inverse_transform(fluid_data_norm_ref)
        fluid_data_ref = self.__CalcMeshData(fluid_data_ref)

        # Create triangulation of filtered thermodynamic state data
        fluid_data_norm_ref = self._fluid_data_scaler.transform(fluid_data_ref)
        DT = Delaunay(fluid_data_norm_ref[:, [EntropicVars.Density.value,EntropicVars.Energy.value]])

        # Extract triangulation, hull nodes, and table data
        Tria = DT.simplices 
        HullNodes = concave_hull_indexes(fluid_data_norm_ref[:, [EntropicVars.Density.value,EntropicVars.Energy.value]])

        self._table_nodes = fluid_data_ref 
        self._table_connectivity = Tria 
        self._table_hullnodes = HullNodes
        
        # Add static enthalpy and the specific heat at constant volume
        self.table_vars.append("Enthalpy")
        h = self._table_nodes[:, EntropicVars.Energy.value] + self._table_nodes[:, EntropicVars.p.value] / self._table_nodes[:, EntropicVars.Density.value]
        self._table_nodes = np.hstack((self._table_nodes, h[:,np.newaxis]))

        # Plot the mesh + expected expansion
        fig, ax = plt.subplots()
        i_rho=self.LuT_var_index("Density")
        i_e=self.LuT_var_index("Energy")

        X_Data=self._table_nodes[:,i_rho]
        Y_Data=self._table_nodes[:,i_e]*1e-3
        
        ax.scatter(X_Data, Y_Data, s=5, c="blue", label="Table nodes")
        ax.plot(RefPoints_clean[:,0], RefPoints_clean[:,1]*1e-3, "-r", label="Ref exp")

        ax.set_xlabel("rho [kg/m3]")
        ax.set_ylabel("e [kJ/kg]")

        ax.legend(loc="lower right")

        ax.grid(ls=":", c="lightgray")

        plt.show(block=True)

        fig.savefig(f"{MainFolder}/Mesh_vs_Ref_Exp.pdf", dpi=600)
        fig.savefig(f"{MainFolder}/Mesh_vs_Ref_Exp.svg", dpi=600)


        return

    def LuT_var_index(self, var: str) -> int:

        try:
            return self.table_vars.index(var)
        except ValueError:
            raise KeyError(f"'{var}' not found in self.table_vars") from None


    def build_triangulation(self, x, y, connectivity, hullnodes=None):
        x = np.asarray(x)
        y = np.asarray(y)

        conn = np.asarray(connectivity, dtype=np.int64)
        if conn.shape[1] != 3:
            raise ValueError(f"The connettivity is not triangular: shape={conn.shape}")

        # chck if gmsh is 1-based
        if conn.min() == 1:
            conn = conn - 1

        self.x = x
        self.y = y
        self.tri = mtri.Triangulation(x, y, triangles=conn)

        self.hullnodes = None
        if hullnodes is not None:
            hn = np.asarray(hullnodes, dtype=np.int64)
            if hn.min() == 1:
                hn = hn - 1
            self.hullnodes = hn

    def PlotContoursLuT(self, config, Variables, Unit, MainFolder, PlotFolderLuT):
        """
        Plot the LuT contours in a P-s diagram
        
        :param config: item of class Config_NICFD containing the configuration data
        :param Variables: array of string containing the name of the properties that will be plotted
        :param Unit: array of string containing the unit of measurements of the properties that will be plotted
        :param MainFolder: string indicating the folder where all the outputs are saved
        :param PlotFolderLuT: string indicating the folder where all the plots are saved, it is contained in MainFolder
        """

        if os.path.isdir(MainFolder+"/"+PlotFolderLuT) is False:
            os.mkdir(MainFolder+"/"+PlotFolderLuT)

        EOS=config._Config_NICFD__EOS_type
        fluid=config._Config_NICFD__fluid_names[0]

        iP=self.LuT_var_index("p")
        iS=self.LuT_var_index("s")
                          
        #rhoMin=config._Config_NICFD__Rho_lower
        #rhoMax=config._Config_NICFD__Rho_upper

        sMin=np.nanmin(self._table_nodes[:,iS]*1e-3)
        sMax=np.nanmax(self._table_nodes[:,iS]*1e-3)

        PMin=np.nanmin(self._table_nodes[:,iP]*1e-5)
        PMax=np.nanmax(self._table_nodes[:,iP]*1e-5)

        #eMin=config._Config_NICFD__Energy_lower
        #eMax=config._Config_NICFD__Energy_upper

        Psat=np.linspace(CP.PropsSI("PTRIPLE",EOS+"::"+fluid), CP.PropsSI("PCRIT",EOS+"::"+fluid),2000)
        sLiq=CP.PropsSI("S","P",Psat,"Q",0,EOS+"::"+fluid)
        sVap=CP.PropsSI("S","P",Psat,"Q",1,EOS+"::"+fluid)

        #rhoLiq=CP.PropsSI("D","P",Psat,"Q",0,EOS+"::"+fluid)
        #rhoVap=CP.PropsSI("D","P",Psat,"Q",1,EOS+"::"+fluid)
        #eLiq=CP.PropsSI("Umass","P",Psat,"Q",0,EOS+"::"+fluid)
        #eVap=CP.PropsSI("Umass","P",Psat,"Q",1,EOS+"::"+fluid)

        X_Data=self._table_nodes[:,iS]*1e-3
        Y_Data=self._table_nodes[:,iP]*1e-5
        
        #hn=np.asarray(self._table_hullnodes, dtype=np.int64)
        self.build_triangulation(X_Data, Y_Data, self._table_connectivity, hullnodes=self._table_hullnodes)

        for i in range(len(Variables)):
            ivar=self.LuT_var_index(Variables[i])

            fig,ax=plt.subplots()
            # Plot saturation dome
            #ax.plot(rhoLiq,eLiq*1e-3,"-k")
            #ax.plot(rhoVap,eVap*1e-3,"-k")
            ax.plot(sLiq*1e-3,Psat*1e-5,"-k")
            ax.plot(sVap*1e-3,Psat*1e-5,"-k")

            # Contour
            if Variables[i]!="c2":
                Z_Data=self._table_nodes[:,ivar]
            else:
                Z_Data=self._table_nodes[:,ivar]**0.5

            #Z_Plot=griddata((X_Data, Y_Data), Z_Data, (X_Data[None,:], Y_Data[:,None]), method='linear')

            #print(np.nanmin(Z_Data))
            #print(np.nanmax(Z_Data))
            #print("masked?", np.ma.isMaskedArray(Z_Data), "min/max", np.nanmin(Z_Data), np.nanmax(Z_Data))
            levels = np.linspace(np.nanmin(Z_Data),np.nanmax(Z_Data)*1.000001, 11)
            cont=ax.tricontourf(self.tri,Z_Data,levels=levels)
            #cont=ax.tricontourf(self.tri,Z_Data)
            #cont=ax.scatter(X_Data, Y_Data, c=Z_Data, s=2)

            """"""
            if Variables[i]!="c2":
                plt.colorbar(cont,ax=ax,ticks=levels, location='right',label=Variables[i]+f" [{Unit[i]}]")
            else:
                plt.colorbar(cont,ax=ax,ticks=levels,location='right',label=f"c [{Unit[i]}]")

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

            fig.savefig(f"{MainFolder}/{PlotFolderLuT}/LuT_P-s_diagram+{Variables[i]}_contour.pdf", dpi=600)
            fig.savefig(f"{MainFolder}/{PlotFolderLuT}/LuT_P-s_diagram+{Variables[i]}_contour.svg", dpi=600)

        return
    
    def AddRefinementCriterion(self, TD_variable:str, norm_val_min:float=np.inf, norm_val_max:float=-np.inf):
        """Apply refinement in the table where the normalized value of the thermodynamic variable lies between the specified bounds.

        :param TD_variable: name of the thermodynamic variable for which to apply refinement
        :type TD_variable: str
        :param norm_val_min: lower bound of the normalized thermodynamic variable, defaults to np.inf
        :type norm_val_min: float, optional
        :param norm_val_max: upper bound of the normalized thermodynamic variable, defaults to -np.inf
        :type norm_val_max: float, optional
        :raises Exception: if thermodynamic state variable is unknown to SU2 DataMiner
        """
        if TD_variable not in self.table_vars:
            raise Exception("%s is not present in fluid data" % TD_variable)
        
        self.refinement_vars.append(TD_variable)
        self.refinement_norm_min.append(norm_val_min)
        self.refinement_norm_max.append(norm_val_max)
        return 
    
    def __ApplyRefinement(self, fluid_data_norm_ref:np.ndarray[float]):
        ix_ref = np.array([],dtype=np.int64)
        fluid_vars = [a.name for a in EntropicVars][:-1]
        for TD_var, val_min, val_max in zip(self.refinement_vars, self.refinement_norm_min, self.refinement_norm_max):
            norm_data_var = fluid_data_norm_ref[:, fluid_vars.index(TD_var)]

            ix = np.argwhere(np.logical_and(norm_data_var>=val_min, norm_data_var<=val_max))[:,0]
            ix_ref = np.append(ix_ref, ix)
        if len(ix_ref) > 0:
            return np.unique(ix_ref)
        else:
            return []

    def __ApplyRefinement_exp(self, fluid_data_coarse:np.ndarray[float], ref_points:np.ndarray[float]):
        ix_ref = np.array([],dtype=np.int64)
        fluid_vars = [a.name for a in EntropicVars][:-1]
        Density_Data = fluid_data_coarse[:, fluid_vars.index("Density")]
        Energy_Data = fluid_data_coarse[:, fluid_vars.index("Energy")]
        for TH in zip(ref_points[:,0], ref_points[:,1]):
            if TH[0]<=9:
                ix = np.argwhere(np.logical_and(np.logical_and(Density_Data>=TH[0]*0.1, Density_Data<=1.5*TH[0]), \
                             np.logical_and(Energy_Data>=TH[1]*0.98, Energy_Data<=1.025*TH[1])))[:,0]

            else:
                ix = np.argwhere(np.logical_and(np.logical_and(Density_Data>=TH[0]*0.98, Density_Data<=1.02*TH[0]), \
                             np.logical_and(Energy_Data>=TH[1]*0.98, Energy_Data<=1.02*TH[1])))[:,0]
            ix_ref = np.append(ix_ref, ix)
        if len(ix_ref) > 0:
            return np.unique(ix_ref)
        else:
            return []

    def WriteTableFile(self, MainFolder:str=None,output_filepath:str=None):
        """
        Save the table data and connectivity as a Dragon library file. If no file name is provided, the table file will be named according to the Config_FGM class name.

        :param output_filepath: optional output filepath for table file.
        :type output_filepath: str
        """

        if output_filepath:
            file_out = MainFolder+"/"+output_filepath
        else:
            file_out = self._savedir + "/LUT_"+self._Config.GetConfigName()+".drg"

        print("Writing LUT file with name " + file_out)
        fid = open(file_out, "w+")
        fid.write("Dragon library\n\n")
        fid.write("<Header>\n\n")
        fid.write("[Version]\n1.0.1\n\n")

        fid.write("[Number of points]\n")
        fid.write("%i\n" % np.shape(self._table_nodes)[0])
        fid.write("\n")

        fid.write("[Number of triangles]\n")
        fid.write("%i\n" % np.shape(self._table_connectivity)[0])
        fid.write("\n")

        fid.write("[Number of hull points]\n")
        fid.write("%i\n" % np.shape(self._table_hullnodes)[0])
        fid.write("\n")

        fid.write("[Number of variables]\n%i\n\n" % (len(self.table_vars)))
        fid.write("[Variable names]\n")
        for iVar, Var in enumerate(self.table_vars):
            fid.write(str(iVar + 1)+":"+Var+"\n")
        fid.write("\n")

        fid.write("</Header>\n\n")

        print("Writing table data...")
        fid.write("<Data>\n")
        for iNode in range(len(self._table_nodes)):
            for ivar in range(len(self.table_vars)):
                fid.write("\t%+.14e" % self._table_nodes[iNode, ivar])
            fid.write("\n")
        fid.write("</Data>\n\n")
        print("Done!")

        print("Writing table connectivity...")
        fid.write("<Connectivity>\n")
        for iCell in range(len(self._table_connectivity)):
            fid.write("\t".join("%i" % c for c in self._table_connectivity[iCell, :]+1) + "\n")
        fid.write("</Connectivity>\n\n")
        print("Done!")

        print("Writing hull nodes...")
        fid.write("<Hull>\n")
        for iCell in range(len(self._table_hullnodes)):
            fid.write(("%i" % (self._table_hullnodes[iCell]+1)) + "\n")
        fid.write("</Hull>\n\n")
        print("Done!")

        fid.close()

        return
    
    def WriteOutParaview(self, connectivity, data_nodes_2d, MainFolder,outpath, x_vars, y_vars, variables=None):
        """
        write a file containing all the LuT data that can be opened with Paraview
        
        :param connectivity: contains the node index of the created LuT
        :param data_nodes_2d: contains the LuT nodes
        :param MainFolder: string indicating the folder where all the outputs are saved
        :param outpath: string indicating the name and extension of the saved file
        :param x_vars: name of the variable that varies along the mesh x direction
        :param y_vars: name of the variable that varies along the mesh y direction
        :param variables: list of the saved variables, if None all the available variables are saved
        """

        data_nodes_2d = np.asarray(data_nodes_2d)
        ix=self.LuT_var_index(x_vars)
        iy=self.LuT_var_index(y_vars)
        # coordinate del dominio: rho, e
        x = data_nodes_2d[:, ix]

        if y_vars=="Energy":
            y = data_nodes_2d[:, iy]*1e-3

        else:
            y = data_nodes_2d[:, iy]
            
        pts = np.column_stack([x, y, np.zeros_like(x)])  # z=0

        conn = np.asarray(connectivity, dtype=np.int64)
        if conn.min() == 1:
            conn = conn - 1

        if variables is None:
            variables = list(self.table_vars)

        point_data = {}
        for name in variables:
            j = self.LuT_var_index(name)
            point_data[name] = np.asarray(data_nodes_2d[:, j])

        mesh = meshio.Mesh(
            points=pts,
            cells=[("triangle", conn)],
            point_data=point_data
        )
        mesh.write(MainFolder+"/"+outpath)

        return
    
    def write_TxT_config(self,config, MainFolder):
        """
        Write a txt file with the main configuration inputs
        
        :param config: item of class Config_NICFD containing the configuration data
        :param MainFolder: string indicating the folder where all the outputs are saved
        """

        # Fluid Input
        # ICEM txt file
        output_file = open(f"{MainFolder}/Config.txt", "w")
        output_file.write("%s%s \n" %("EOS=" , config.GetEquationOfState()))
        output_file.write("%s%s \n" %("Fluid=",config.GetFluid()))

        if config.GetPTGrid():
            output_file.write("%s%.0f \n" %("Np=",config.GetNpPressure()))
            output_file.write("%s%.0f \n" %("NT=",config.GetNpTemp()))

            if config.GetAutoRange():
                output_file.write("%s \n" %("Autorange activated"))

            else:

                PRange=config.GetPressureBounds()
                TRange=config.GetTemperatureBounds()
                output_file.write("%s%.0f, %.0f%s \n" %("PRange=[",PRange[0],PRange[1],"] Pa"))
                output_file.write("%s%.0f, %.0f%s \n" %("TRange=[",TRange[0],TRange[1],"] K"))

        else:
            output_file.write("%s%.0f \n" %("Nrho=",config.GetNpDensity()))
            output_file.write("%s%.0f \n" %("Ne=",config.GetNpEnergy()))

            if config.GetAutoRange():
                output_file.write("%s \n" %("Autorange activated"))

            else:

                rhoRange=config.GetDensityBounds()
                eRange=config.GetEnergyBounds()
                output_file.write("%s%.3f, %.0f%s \n" %("rhoRange=[",rhoRange[0],rhoRange[1],"] kg/m3"))
                output_file.write("%s%.0f, %.0f%s \n" %("eRange=[",eRange[0],eRange[1],"] J/kg"))

        # Steps Input
        output_file.write("%s%.0f %s \n" %("dP=",config.GetdPFD(), "Pa"))
        output_file.write("%s%.0f %s \n" %("dh=",config.GetdhFD(), "J/kg"))
        output_file.write("%s%.6f \n" %("drho multiplier=",config.GetdrhoMultFD()))

        # Local refinment options
        RefMaxCell=self._refined_cell_size
        RefRadius=self._refinement_radius

        output_file.write("%s%.9f \n" %("Maximum refined cell size=",RefMaxCell))
        output_file.write("%s%.9f \n" %("Refinment radius=",RefRadius))
        # Local refinment
        output_file.write("%s \n" %("List of refinments applied to the LuT"))

        VarRefinment=self.refinement_vars
        MinRefinment=self.refinement_norm_min
        MaxRefinment=self.refinement_norm_max

        nVarRef=len(VarRefinment)

        for i in range(nVarRef):
            output_file.write("%s%s, %s%.4f, %s%.4f \n" %("Var: ",VarRefinment[i],"Min normal ref=", MinRefinment[i], "Max normal ref=", MaxRefinment[i]))

        output_file.close()

        return
