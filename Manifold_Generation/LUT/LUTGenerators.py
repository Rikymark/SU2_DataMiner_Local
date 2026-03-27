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

    ######## START USER-INPUT ########
    _base_cell_size:float = 2e-2      # Table level base cell size.

    _refined_cell_size:float = 2e-3 #2.5e-3#1.5e-3   # Table level refined cell size. # old value is 5e-3, standard for adapted ref 2e-3
    _finer_refined_cell_size:float = 2e-4 # Refinment for finer zones
    _finer_sat_refined_cell_size:float = 7.5e-5 # Refinment for finer zones near saturation curve
    _refinement_radius:float = 4e-2 #5e-2     # Table level radius within which refinement is applied. # original value is 1e-2
    _refinement_radius_sat_curve:float = 1.5e-3 #5e-2  # Table level radius within which refinement is applied for the points along the sat curve. 

    _LowMult_Density:float=0.9985  # Adaptive refinment density lower bound is computed as _LowMult_Density*Density[i]
    _HighMult_Density:float=1.015 # Adaptive refinment density upper bound is computed as _HighMult_Density*Density[i]

    _LowMult_Energy:float=0.985 # Adaptive refinment energy lower bound is computed as _LowMult_Energy*Energy[i]
    _HighMult_Energy:float=1.015 # Adaptive refinment energy upper bound is computed as _HighMult_Energy*Energy[i]

    _LowMult_Density_ref:float=0.1 # Adaptive refinment density lower bound is computed as _LowMult_Density_ref*Density[i] when density is lower than _Density_ref_value
    _HighMult_Density_ref:float=1.1 # Adaptive refinment density upper bound is computed as _HighMult_Density_ref*Density[i] when density is lower than _Density_ref_value

    _Density_ref_value:float=5 # Density value below which _LowMult_Density_ref and _HighMult_Density_ref are employed as multiplier
    _Density_finer_ref_value:float=8 # Density value below the finer refinment is activated
    _Sat_Curve_Discretization:float=0.5 # Define the saturation curve spacing as _Sat_Curve_Discretization*_base_cell_size
    ######## END USER-INPUT ########

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

    def map_tags(self,tags,nodeTags_sorted,order):
        tags = np.asarray(tags, dtype=np.int64).ravel()
        pos = np.searchsorted(nodeTags_sorted, tags)
        ok = (pos < len(nodeTags_sorted)) & (nodeTags_sorted[pos] == tags)
        if not np.all(ok):
            missing = np.unique(tags[~ok])
            raise RuntimeError(f"Node tags non trovati in getNodes(): {missing[:20]} (tot missing={len(missing)})")
        return order[pos]
       
    def __Compute2DMesh(self, points:np.ndarray[float],sat_curve_norm_clipped:np.ndarray[float]=[], ref_pts:np.ndarray[float]=[],ref_pts_add:np.ndarray[float]=[],ref_pts_add_sat:np.ndarray[float]=[] , show:bool=False):
        
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

        gmsh.option.setNumber("Mesh.RecombineAll", 0)
        gmsh.option.setNumber("Mesh.Recombine3DAll", 0)

        # Force triangular algorithm (Frontal-Delaunay)
        gmsh.option.setNumber("Mesh.Algorithm", 6)

        # Avoid change in the element type
        gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 0)

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
        SurfaceTag=factory.addPlaneSurface([curvloop])

        # Apply refinement points
        ref_pt_ids = []
        if len(ref_pts)>0:
            for i in range(len(ref_pts)):
                ref_pt_ids.append(factory.addPoint(ref_pts[i,0], ref_pts[i, 1], 0.0))

        # Points where to apply finer refinment
        ref_pt_ids_add = []
        if len(ref_pts_add)>0:
            for i in range(len(ref_pts_add)):
                ref_pt_ids_add.append(factory.addPoint(ref_pts_add[i,0], ref_pts_add[i, 1], 0.0))

        # Points where to apply finer refinment near the saturation curve
        ref_pt_ids_add_sat = []
        if len(ref_pts_add_sat)>0:
            for i in range(len(ref_pts_add_sat)):
                ref_pt_ids_add_sat.append(factory.addPoint(ref_pts_add_sat[i,0], ref_pts_add_sat[i, 1], 0.0))

        # Add the saturation curve as a mesh curve
        #occ = gmsh.model.occ

        sat_tags = []

        for sat in sat_curve_norm_clipped:
            sat_resampled=self.resample_polyline_by_step(sat, self._Sat_Curve_Discretization*self._base_cell_size)
            pt_tags = [factory.addPoint(float(rho), float(e), 0.0) for rho, e in sat_resampled]

            sat_line=[factory.addLine(pt_tags[i], pt_tags[i+1]) for i in range(len(pt_tags)-1)]
            sat_tags.extend(sat_line)

        factory.synchronize()

        gmsh.model.mesh.embed(1, sat_tags, 2, SurfaceTag)

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

        # Apply finer refinement where needed
        gmsh.model.mesh.field.add("Distance", 3)
        gmsh.model.mesh.field.setNumbers(3, "PointsList", ref_pt_ids_add)
        gmsh.model.mesh.field.setNumber(3, "Sampling", 100)

        gmsh.model.mesh.field.add("Threshold", 4)
        gmsh.model.mesh.field.setNumber(4, "InField", 3)
        gmsh.model.mesh.field.setNumber(4, "SizeMin", self._finer_refined_cell_size)
        gmsh.model.mesh.field.setNumber(4, "SizeMax", self._base_cell_size)
        gmsh.model.mesh.field.setNumber(4, "DistMin", 0.5*self._refinement_radius)
        gmsh.model.mesh.field.setNumber(4, "DistMax", 1.5*self._refinement_radius)

        # Apply finer refinement around the saturation curve
        gmsh.model.mesh.field.add("Distance", 5)
        gmsh.model.mesh.field.setNumbers(5, "PointsList", ref_pt_ids_add_sat)
        gmsh.model.mesh.field.setNumber(5, "Sampling", 100)

        gmsh.model.mesh.field.add("Threshold", 6)
        gmsh.model.mesh.field.setNumber(6, "InField", 5)
        gmsh.model.mesh.field.setNumber(6, "SizeMin", self._finer_sat_refined_cell_size)
        gmsh.model.mesh.field.setNumber(6, "SizeMax", self._base_cell_size)
        gmsh.model.mesh.field.setNumber(6, "DistMin", 0.5*self._refinement_radius_sat_curve)
        gmsh.model.mesh.field.setNumber(6, "DistMax", 1.5*self._refinement_radius_sat_curve)

        gmsh.model.mesh.field.add("Min", 7)
        gmsh.model.mesh.field.setNumbers(7, "FieldsList", [2, 4, 6])
        gmsh.model.mesh.field.setAsBackgroundMesh(7)
        
        factory.synchronize()

        # Generate 2D mesh and extract table nodes+connectivity
        gmsh.model.mesh.generate(2)
        gmsh.write("gmsh_mesh.vtk")   # o .msh / .vtu
        if show:
            gmsh.fltk.run()

        # Global nodes
        nodeTags, coords, _ = gmsh.model.mesh.getNodes()  
        nodeTags = np.asarray(nodeTags, dtype=np.int64)
        MeshPoints = np.asarray(coords, dtype=float).reshape(-1, 3)[:, :2]

        order = np.argsort(nodeTags)
        nodeTags_sorted = nodeTags[order]

        # 2) 2D elements
        if SurfaceTag is None:
            elemTypes, _, elemNodeTags = gmsh.model.mesh.getElements(2)
        else:
            elemTypes, _, elemNodeTags = gmsh.model.mesh.getElements(2, SurfaceTag)

        tris = []
        quads = []

        for et, nodes_flat in zip(elemTypes, elemNodeTags):
            if et == 2:  # triangles with 3 nodes
                tri_tags = np.asarray(nodes_flat, dtype=np.int64).reshape(-1, 3)
                tris.append(self.map_tags(tri_tags, nodeTags_sorted,order).reshape(-1, 3))
            elif et == 3:  # quad with 4 nodes
                quad_tags = np.asarray(nodes_flat, dtype=np.int64).reshape(-1, 4)
                quads.append(self.map_tags(quad_tags, nodeTags_sorted,order).reshape(-1, 4))

        tris = np.vstack(tris) if tris else np.zeros((0, 3), dtype=np.int64)

        if quads:
            quads = np.vstack(quads)
            # split quad -> 2 tri: (0,1,2) + (0,2,3)
            tris = np.vstack([
                tris,
                quads[:, [0, 1, 2]],
                quads[:, [0, 2, 3]],
            ])

        #nodes = gmsh.model.mesh.getNodes(dim=2, tag=-1, includeBoundary=True, returnParametricCoord=False)[1]
        gmsh.finalize()
        #MeshPoints = np.array([nodes[::3], nodes[1::3]]).T

        return MeshPoints, tris
        
    
    def __CalcMeshData(self, fluid_data_mesh:np.ndarray[float]):
        """Calculate the fluid thermodynamic state variables for the table nodes

        :param fluid_data_mesh: table mesh nodes of density and static energy
        :type fluid_data_mesh: np.ndarray[float]
        :return: filtered thermodynamic state data at the table nodes
        :rtype: np.ndarray[float]
        """
        fluid_data_out = fluid_data_mesh.copy()
        i_None=np.array([],dtype=np.int64)
        for i in range(len(fluid_data_mesh)):
            try:
                self._DataGenerator.UpdateFluid(fluid_data_mesh[i, EntropicVars.Density.value], fluid_data_mesh[i, EntropicVars.Energy.value])
                state_vector, correct_phase = self._DataGenerator.GetStateVector()
                if correct_phase:
                    fluid_data_out[i, :] = state_vector
                else:
                    fluid_data_out[i, :] = None
            except:
                print(f"The properties computation has failed in the point rho={fluid_data_mesh[i, EntropicVars.Density.value]} kg/m3, \
                      e={fluid_data_mesh[i, EntropicVars.Energy.value]} J/kg")
                i_None=np.append(i_None,i)
                fluid_data_out[i, :] = None
        fluid_data_out = fluid_data_out[~np.isnan(fluid_data_out[:,0]),:]
        return fluid_data_out,i_None
    
    def Normalize_Sat_Curve(self, sat_curve,config,eps=1e-6):
        
        sat_curve_norm=np.copy(sat_curve)
        rho_bound=config.GetDensityBounds()
        e_bound=config.GetEnergyBounds()
        sat_curve_norm[:,0]=[(rho-rho_bound[0])/(rho_bound[1]-rho_bound[0]) for rho in sat_curve[:,0]]
        sat_curve_norm[:,1]=[(e-e_bound[0])/(e_bound[1]-e_bound[0])  for e in sat_curve[:,1]]
        sat_curve_clipped=self.clip_polyline_to_rect(sat_curve, rho_bound[0]+eps, rho_bound[1]-eps, e_bound[0]+eps, e_bound[1]-eps)
        sat_curve_norm_clipped=self.clip_polyline_to_rect(sat_curve_norm, 0+eps, 1-eps, 0+eps, 1-eps)

        return sat_curve_norm, sat_curve_norm_clipped, sat_curve_clipped

    def clip_segment_liang_barsky(self, p0, p1, xmin, xmax, ymin, ymax, eps=1e-12):
        """
        Clip the saturation curve when it is outside the bounds
        Return None if outside, if inside (q0, q1) are the clipped boundaries.
        """
        x0, y0 = p0
        x1, y1 = p1
        dx = x1 - x0
        dy = y1 - y0

        p = np.array([-dx, dx, -dy, dy], dtype=float)
        q = np.array([x0 - xmin, xmax - x0, y0 - ymin, ymax - y0], dtype=float)

        u1, u2 = 0.0, 1.0
        for pi, qi in zip(p, q):
            if abs(pi) < eps:
                if qi < 0:
                    return None  
                continue
            u = qi / pi
            if pi < 0:
                if u > u1: u1 = u
            else:
                if u < u2: u2 = u
            if u1 - u2 > eps:
                return None

        q0 = np.array([x0 + u1*dx, y0 + u1*dy])
        q1 = np.array([x0 + u2*dx, y0 + u2*dy])
        return q0, q1

    def clip_polyline_to_rect(self, pts, xmin, xmax, ymin, ymax, tol=1e-9):
        """
        Clip a polyline pts (n,2) at the 2D mesh.
        Return a list of internal polylines (each (k,2), k>=2).
        Can manage numerous intersections (2, 4 or more).
        """
        pts = np.asarray(pts, dtype=float)
        if len(pts) < 2:
            return []

        out_segments = []
        current = None  # lista di punti del tratto in costruzione

        for i in range(len(pts)-1):
            p0 = pts[i]
            p1 = pts[i+1]
            clipped = self.clip_segment_liang_barsky(p0, p1, xmin, xmax, ymin, ymax)
            if clipped is None:
                # close a path under construction
                if current is not None and len(current) >= 2:
                    out_segments.append(np.array(current))
                current = None
                continue

            q0, q1 = clipped

            if current is None:
                current = [q0, q1]
            else:
                # Avoid duplicates when the new segment is clipped
                if np.linalg.norm(current[-1] - q0) > tol:
                    current.append(q0)
                current.append(q1)

        if current is not None and len(current) >= 2:
            out_segments.append(np.array(current))

        # remove the equal/near-equal consecutive points
        cleaned = []
        for seg in out_segments:
            keep = [seg[0]]
            for k in range(1, len(seg)):
                if np.linalg.norm(seg[k] - keep[-1]) > tol:
                    keep.append(seg[k])
            if len(keep) >= 2:
                cleaned.append(np.array(keep))
        return cleaned

    def resample_polyline_by_step(self, P, ds):
        """Resample a polyline P (n,2) with step ~ ds. Return (m,2)."""
        P = np.asarray(P, float)
        if len(P) < 2:
            return P.copy()

        seg = P[1:] - P[:-1]
        L = np.linalg.norm(seg, axis=1)
        s = np.concatenate([[0.0], np.cumsum(L)])
        total = s[-1]
        if total == 0:
            return P[[0]].copy()

        # points along the curvilinear abscissa
        n = int(np.floor(total / ds))
        samp = np.linspace(0.0, total, n + 2)  

        out = []
        j = 0
        for u in samp:
            while j < len(L) - 1 and s[j+1] < u:
                j += 1
            t = (u - s[j]) / (L[j] + 1e-30)
            out.append(P[j] + t * seg[j])
        return np.array(out)

    def remove_invalid_nodes_from_mesh(self, connectivity, valid_mask, rhoe_mesh_norm):

        conn = np.asarray(connectivity, dtype=np.int64)

        tri_keep = np.all(valid_mask[conn], axis=1)
        conn2 = conn[tri_keep]

        keep_nodes = np.flatnonzero(valid_mask)
        old_to_new = -np.ones(len(rhoe_mesh_norm), dtype=np.int64)
        old_to_new[keep_nodes] = np.arange(len(keep_nodes), dtype=np.int64)

        conn2 = old_to_new[conn2]

        return conn2
    def GenerateTable(self, LoadRef, MainFolder, config):
        """Initiate table generation process
        """

        # Compute the saturation curves
        EOS=config._Config_NICFD__EOS_type
        fluid=config._Config_NICFD__fluid_names[0]

        Psat=np.linspace(CP.PropsSI("PTRIPLE",EOS+"::"+fluid), CP.PropsSI("PCRIT",EOS+"::"+fluid),2000)

        rhoLiq=CP.PropsSI("D","P",Psat,"Q",0,EOS+"::"+fluid)
        rhoVap=CP.PropsSI("D","P",Psat,"Q",1,EOS+"::"+fluid)

        eLiq=CP.PropsSI("U","P",Psat,"Q",0,EOS+"::"+fluid)
        eVap=CP.PropsSI("U","P",Psat,"Q",1,EOS+"::"+fluid)

        rho_sat=np.concatenate((rhoLiq[:-1],np.flip(rhoVap)))
        e_sat=np.concatenate((eLiq[:-1],np.flip(eVap)))
        sat_curve=np.column_stack((rho_sat,e_sat))
        sat_curve_norm, sat_curve_norm_clipped, sat_curve_clipped=self.Normalize_Sat_Curve(sat_curve, config)

        # Load initial fluid data and scale it
        fluid_data_norm = self.__LoadFluidData()
        rhoe_norm = fluid_data_norm[:, [EntropicVars.Density.value, EntropicVars.Energy.value]]

        # Generate initial coarse table of fluid data
        rhoe_mesh_norm_coarse, Tria_coarse = self.__Compute2DMesh(rhoe_norm,sat_curve_norm_clipped=sat_curve_norm_clipped)

        # Calculate thermodynamic state variables of initial table nodes
        fluid_data_norm_coarse = np.zeros([len(rhoe_mesh_norm_coarse), EntropicVars.N_STATE_VARS.value])
        fluid_data_norm_coarse[:, EntropicVars.Density.value] = rhoe_mesh_norm_coarse[:,0]
        fluid_data_norm_coarse[:, EntropicVars.Energy.value] = rhoe_mesh_norm_coarse[:,1]
        fluid_data_coarse = self._fluid_data_scaler.inverse_transform(fluid_data_norm_coarse)
        fluid_data_coarse, _ = self.__CalcMeshData(fluid_data_coarse)

        # Identify refinement locations
        fluid_data_norm = self._fluid_data_scaler.transform(fluid_data_coarse)
        ix_ref_user = self.__ApplyRefinement(fluid_data_norm)

        # add refinment near the expected thermodynamic path
        RefPoints=np.loadtxt(LoadRef,skiprows=1,usecols=(0,1),delimiter=",",dtype=float)
        mask = np.isfinite(RefPoints).all(axis=1) 
        RefPoints_clean=RefPoints[mask, :]  

        if len(sat_curve_norm_clipped)==2:
            sat_curve_norm_clipped_merged=np.vstack((sat_curve_norm_clipped[0],sat_curve_norm_clipped[1]))
            #sat_curve_clipped_merged=np.vstack((sat_curve_clipped[0],sat_curve_clipped[1]))
        elif len(sat_curve_norm_clipped)==1:
            sat_curve_norm_clipped_merged=np.copy(sat_curve_norm_clipped[0])
            #sat_curve_clipped_merged=np.copy(sat_curve_clipped[0])
        else:
            sat_curve_norm_clipped_merged=np.copy(sat_curve_norm)
            #sat_curve_clipped_merged=np.copy(sat_curve)

        ix_ref_TH_transf, ix_ref_TH_transf_add, ix_ref_TH_transf_add_sat=self.__ApplyRefinement_exp(fluid_data_coarse, RefPoints_clean, sat_curve_clipped)

        # Regenerate table including refinement locations
        rhoe_norm_mesh = fluid_data_norm[:, [EntropicVars.Density.value, EntropicVars.Energy.value]]
        ix_ref=np.union1d(ix_ref_user,ix_ref_TH_transf).astype(np.int64)
        rhoe_norm_ref = rhoe_norm_mesh[ix_ref, :]
        rhoe_norm_ref_add = rhoe_norm_mesh[ix_ref_TH_transf_add, :]

        sat_curve_ref= sat_curve_norm_clipped_merged[ix_ref_TH_transf_add_sat, :]

        rhoe_mesh_norm, Tria = self.__Compute2DMesh(rhoe_norm, sat_curve_norm_clipped=sat_curve_norm_clipped, ref_pts=rhoe_norm_ref,ref_pts_add=rhoe_norm_ref_add, ref_pts_add_sat=sat_curve_ref, show=True)

        # Extract thermodynamic state variables of refined table
        fluid_data_norm_ref = np.zeros([len(rhoe_mesh_norm), EntropicVars.N_STATE_VARS.value])
        fluid_data_norm_ref[:, EntropicVars.Density.value] = rhoe_mesh_norm[:,0]
        fluid_data_norm_ref[:, EntropicVars.Energy.value] = rhoe_mesh_norm[:,1]
        fluid_data_ref = self._fluid_data_scaler.inverse_transform(fluid_data_norm_ref)

        fluid_data_ref, i_None = self.__CalcMeshData(fluid_data_ref)
        fluid_data_norm_ref = self._fluid_data_scaler.transform(fluid_data_ref)

        # Save in a mask the index of the invalid nodes 
        valid_mask = np.ones(len(rhoe_mesh_norm), dtype=bool)
        valid_mask[i_None] = False

        # Create triangulation of filtered thermodynamic state data
        
        #DT = Delaunay(fluid_data_norm_ref[:, [EntropicVars.Density.value,EntropicVars.Energy.value]])

        # Extract triangulation, hull nodes, and table data
        #Tria = DT.simplices 
        if Tria.max() + 1!=len(fluid_data_ref ):
            print("WARNING: COOLPROP HAS FAILED IN AT LEAST ONE NODE, NEW TRIANGULATION WILL BE COMPUTED")
            Tria=self.remove_invalid_nodes_from_mesh(Tria, valid_mask, rhoe_mesh_norm)

        HullNodes = concave_hull_indexes(fluid_data_norm_ref[:, [EntropicVars.Density.value,EntropicVars.Energy.value]])

        self._table_nodes = fluid_data_ref 
        self._table_connectivity = Tria 
        self._table_hullnodes = HullNodes
        
        """
        if self._table_connectivity.max() + 1!=len(self._table_nodes):
            
            print("len(rhoe_mesh_norm) =", len(rhoe_mesh_norm))
            print("len(self._table_nodes) =", len(self._table_nodes))
            print("len(fluid_data_ref) =", len(fluid_data_ref[:,0]))
            print("conn max =", self._table_connectivity.max())
            print("expected npts from conn =", self._table_connectivity.max() + 1)
        """

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

        ax.plot(sat_curve[:,0],sat_curve[:,1]*1e-3,"k", label="Sat")
        #ax.plot(rhoVap,eVap,"k")

        ax.set_xlabel("rho [kg/m3]")
        ax.set_ylabel("e [kJ/kg]")

        ax.set_xlim(np.min(X_Data),np.max(X_Data))
        ax.set_ylim(np.min(Y_Data),np.max(Y_Data))

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

    

    def __cross2(self, a, b):
        return a[0]*b[1] - a[1]*b[0]

    def __seg_intersect(self, p, p2, q, q2, eps=1e-12):
        """
        Intersection between two segments p->p2 e q->q2.
        """
        r = p2 - p
        s = q2 - q
        rxs = self.__cross2(r, s)
        qp = q - p

        if abs(rxs) < eps:
            return False, None, None, None  

        t = self.__cross2(qp, s) / rxs
        u = self.__cross2(qp, r) / rxs

        if -eps <= t <= 1+eps and -eps <= u <= 1+eps:
            pt = p + t*r
            return True, t, u, pt
        return False, None, None, None

    def __polyline_intersections(self, A, B, eps=1e-12, dedup_tol=1e-9):
        """
        Return the dict list: {iA,iB,tA,tB,P}
        where iA is the index of the segment A[iA]->A[iA+1], the same for B.
        """
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)

        hits = []
        for iA in range(len(A)-1):
            p, p2 = A[iA], A[iA+1]
            for iB in range(len(B)-1):
                q, q2 = B[iB], B[iB+1]
                ok, t, u, pt = self.__seg_intersect(p, p2, q, q2, eps=eps)
                if ok:
                    hits.append({"iA": iA, "iB": iB, "tA": t, "tB": u, "P": pt})

        # Avoid to count an index two times if the intersection correspond to one vertex
        if not hits:
            return []

        pts = np.vstack([h["P"] for h in hits])
        keep = []
        used = np.zeros(len(hits), dtype=bool)

        for k in range(len(hits)):
            if used[k]:
                continue
            d = np.linalg.norm(pts - pts[k], axis=1)
            same = d < dedup_tol
            used[same] = True
            keep.append(hits[k])

        return keep

    def __ApplyRefinement_exp(self, fluid_data_coarse:np.ndarray[float], ref_points:np.ndarray[float], sat_curve_clipped:np.ndarray[float]):

        rho_low_mult=self._LowMult_Density
        rho_up_mult=self._HighMult_Density

        e_low_mult=self._LowMult_Energy
        e_up_mult=self._HighMult_Energy

        rho_low_mult_ref=self._LowMult_Density_ref
        rho_up_mult_ref=self._HighMult_Density_ref

        rho_limit=self._Density_ref_value
        rho_limit_finer=self._Density_finer_ref_value

        ix_ref = np.array([],dtype=np.int64)
        ix_ref_add = np.array([],dtype=np.int64)
        ix_ref_add_sat = np.array([],dtype=np.int64)
        fluid_vars = [a.name for a in EntropicVars][:-1]

        Density_Data = fluid_data_coarse[:, fluid_vars.index("Density")]
        Energy_Data = fluid_data_coarse[:, fluid_vars.index("Energy")]

        for TH in zip(ref_points[:,0], ref_points[:,1]):
            if TH[0]<=rho_limit:
                ix = np.argwhere(np.logical_and(np.logical_and(Density_Data>=TH[0]*rho_low_mult_ref, Density_Data<=rho_up_mult_ref*TH[0]), \
                             np.logical_and(Energy_Data>=TH[1]*e_low_mult, Energy_Data<=e_up_mult*TH[1])))[:,0]
                ix_ref_add = np.append(ix_ref_add, ix)
            
            elif TH[0]>rho_limit and TH[0]<=rho_limit_finer:
                ix = np.argwhere(np.logical_and(np.logical_and(Density_Data>=TH[0]*rho_low_mult, Density_Data<=rho_up_mult*TH[0]), \
                             np.logical_and(Energy_Data>=TH[1]*e_low_mult, Energy_Data<=e_up_mult*TH[1])))[:,0]
                ix_ref_add = np.append(ix_ref_add, ix)

            else:
                ix = np.argwhere(np.logical_and(np.logical_and(Density_Data>=TH[0]*rho_low_mult, Density_Data<=rho_up_mult*TH[0]), \
                             np.logical_and(Energy_Data>=TH[1]*e_low_mult, Energy_Data<=e_up_mult*TH[1])))[:,0]
                
                ix_ref = np.append(ix_ref, ix)

        if len(sat_curve_clipped)!=0:
            for i in range(len(sat_curve_clipped)):
                sat_curve=sat_curve_clipped[i]
                ints = self.__polyline_intersections(ref_points, sat_curve)
                ix_sat = [d["iA"] + (d["tA"] > 0.5) for d in ints]

                ix = np.argwhere(np.logical_and(np.logical_and(sat_curve[:,0]>=ref_points[ix_sat[0],0]*0.975, sat_curve[:,0]<=1.025*ref_points[ix_sat[0],0]), \
                                    np.logical_and(sat_curve[:,1]>=ref_points[ix_sat[0],1]*0.975, sat_curve[:,1]<=1.025*ref_points[ix_sat[0],1])))[:,0]
                if i>0:
                    ix+=len(sat_curve_clipped[i-1])       
                ix_ref_add_sat=np.append(ix_ref_add_sat,ix)
        """
        ints = self.__polyline_intersections(ref_points, sat_curve)
        ix_sat = [d["iA"] + (d["tA"] > 0.5) for d in ints]

        if len(ix_sat)==2:
            ix = np.argwhere(np.logical_and(np.logical_and(sat_curve[:,0]>=ref_points[ix_sat[0],0]*0.99, sat_curve[:,0]<=1.01*ref_points[ix_sat[0],0]), \
                            np.logical_and(sat_curve[:,1]>=ref_points[ix_sat[0],1]*0.99, sat_curve[:,1]<=1.01*ref_points[ix_sat[0],1])))[:,0]
                
            ix_ref_add_sat = np.append(ix_ref_add_sat, ix)

            ix = np.argwhere(np.logical_and(np.logical_and(sat_curve[:,0]>=ref_points[ix_sat[1],0]*0.99, sat_curve[:,0]<=1.01*ref_points[ix_sat[1],0]), \
                            np.logical_and(sat_curve[:,1]>=ref_points[ix_sat[1],1]*0.99, sat_curve[:,1]<=1.001*ref_points[ix_sat[1],1])))[:,0]
                
            ix_ref_add_sat = np.append(ix_ref_add_sat, ix)

        elif len(ix_sat)==1:
            ix = np.argwhere(np.logical_and(np.logical_and(Density_Data>=ref_points[ix_sat[0],0]*0.975, Density_Data<=1.025*ref_points[ix_sat[0],0]), \
                            np.logical_and(Energy_Data>=ref_points[ix_sat[0],1]*0.975, Energy_Data<=1.025*ref_points[ix_sat[0],1])))[:,0]
                
            ix_ref_add_sat = np.append(ix_ref_add_sat, ix)
        """
        if len(ix_ref) > 0 and len(ix_ref_add) > 0 and len(ix_ref_add_sat) > 0:
            ix_ref_add = np.setdiff1d(np.unique(ix_ref_add), np.unique(ix_ref_add_sat), assume_unique=True)
            ix_ref = np.setdiff1d(np.unique(ix_ref), np.union1d(ix_ref_add, ix_ref_add_sat), assume_unique=True)
            return np.unique(ix_ref), np.unique(ix_ref_add), np.unique(ix_ref_add_sat)
        
        elif len(ix_ref) > 0 and len(ix_ref_add)>0 and len(ix_ref_add_sat)==0:
            return np.unique(ix_ref), np.unique(ix_ref_add), []
        
        elif len(ix_ref) == 0 and len(ix_ref_add)>0 and len(ix_ref_add_sat)>0:
            return [], np.unique(ix_ref_add), np.unique(ix_ref_add_sat)

        elif len(ix_ref) > 0 and len(ix_ref_add)==0 and len(ix_ref_add_sat)>0:
            return np.unique(ix_ref), [], np.unique(ix_ref_add_sat)
        
        elif len(ix_ref) > 0 and len(ix_ref_add)==0 and len(ix_ref_add_sat)==0:
            return np.unique(ix_ref), [], []
        
        elif len(ix_ref) == 0 and len(ix_ref_add) > 0 and len(ix_ref_add_sat)==0:
           return [],np.unique(ix_ref_add), []

        elif len(ix_ref) == 0 and len(ix_ref_add)== 0 and len(ix_ref_add_sat)>0:
            return [],[], np.unique(ix_ref_add_sat)

        else:
            return [], [], []

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
        RefMaxCell_finer=self._finer_refined_cell_size
        RefMaxCell_finer_sat=self._finer_sat_refined_cell_size
        RefRadius=self._refinement_radius
        RefRadiusSatCurve=self._refinement_radius_sat_curve
        SatCurveDiscr=self._Sat_Curve_Discretization

        output_file.write("%s%.9f \n" %("Maximum refined cell size=",RefMaxCell))
        output_file.write("%s%.9f \n" %("Maximum finer refined cell size=",RefMaxCell_finer))
        output_file.write("%s%.9f \n" %("Maximum refined cell size near saturated curve=",RefMaxCell_finer_sat))
        output_file.write("%s%.9f \n" %("Refinment radius=",RefRadius))
        output_file.write("%s%.9f \n" %("Refinment radius for points along sat curve=",RefRadiusSatCurve))
        output_file.write("%s%.9f \n" %("Multiplier that defines the saturation curve discretization=",SatCurveDiscr))

        # Local refinment
        output_file.write("%s \n" %("List of refinments applied to the LuT"))

        VarRefinment=self.refinement_vars
        MinRefinment=self.refinement_norm_min
        MaxRefinment=self.refinement_norm_max

        nVarRef=len(VarRefinment)

        for i in range(nVarRef):
            output_file.write("%s%s, %s%.4f, %s%.4f \n" %("Var: ",VarRefinment[i],"Min normal ref=", MinRefinment[i], "Max normal ref=", MaxRefinment[i]))

        # Adaptive refinment options
        rho_low_mult=self._LowMult_Density
        rho_up_mult=self._HighMult_Density

        e_low_mult=self._LowMult_Energy
        e_up_mult=self._HighMult_Energy

        rho_low_mult_ref=self._LowMult_Density_ref
        rho_up_mult_ref=self._HighMult_Density_ref

        rho_limit=self._Density_ref_value
        rho_lim_finer=self._Density_finer_ref_value

        output_file.write("%s%.9f \n" %("Density lower bound multiplier=",rho_low_mult))
        output_file.write("%s%.9f \n" %("Density upper bound multiplier=",rho_up_mult))

        output_file.write("%s%.9f \n" %("Energy lower bound multiplier=",e_low_mult))
        output_file.write("%s%.9f \n" %("Energy upper bound multiplier=",e_up_mult))

        output_file.write("%s%.9f \n" %("Refined density lower bound multiplier=",rho_low_mult_ref))
        output_file.write("%s%.9f \n" %("Refined density upper bound multiplier=",rho_up_mult_ref))

        output_file.write("%s%.9f%s \n" %("Density below which larger bounds are used=",rho_limit," [kg/m3]"))
        output_file.write("%s%.9f%s \n" %("Density below which finer refinment is applied=",rho_lim_finer," [kg/m3]"))

        output_file.close()

        return
