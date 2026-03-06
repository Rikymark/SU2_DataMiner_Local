<p style="margin-bottom:1cm;"> </p>
<p align="center">
        <img src="Documentation/images/SU2DataMiner_logo.png" width="200"/> 
</p>
<p style="margin-bottom:1cm;"> </p
>

# SU2 DataMiner
This repository describes the workflow for manifold generation for data-driven fluid modeling in SU2. The workflow allows the user to generate fluid data and convert these into tables and train multi-layer perceptrons in order to retrieve thermo-chemical quantities during simulations in SU2. The applications are currently limited to non-ideal computational fluid dynamics and flamelet-generated manifold simulations for arbitrary fluids and reactants respectively. 

## MAIN MODIFICATIONS FROM feature_LUT VERSION
### Tested with REFPROP
To add the possibility to compute transport quantities of organic fluids NIST REFPROP is called via CoolProp backend.

It is possible to install REFPROP in Linux starting from the Windows installation if the folder FOLTRAN with .FOR files is present in the
installation, which can be usually found in C:\Program Files (x86)\REFPROP

To install proceed in the following way (TESTED WITH WSL UBUNTU 24.04):
1. *Copy the installation in Linux*: copy REFPROP from Windows to the folder /opt/refprop
        ```
        sudo mkdir -p /opt/refprop
        sudo cp -r "/mnt/c/Program Files (x86)/REFPROP/"* /opt/refprop/
        sudo chmod -R a+rX /opt/refprop
        ```
2. *Check the existence of the necessary files*
        ```
        ls -ld /opt/refprop/FLUIDS /opt/refprop/MIXTURES /opt/refprop/FORTRAN
        ls -l /opt/refprop/FORTRAN/SETUP.FOR
        ls -l /opt/refprop/FORTRAN/DLLFILES/PASS_FTN.FOR
        ```
3. *Install the Fortan compiler*
        ```
        sudo apt update
        sudo apt install -y git cmake make gfortran
        ```
4. *Clone REFPROP-cmake with the necessary submodels from Github*
        ```
        cd ~
        rm -rf REFPROP-cmake
        git clone --recurse-submodules https://github.com/usnistgov/REFPROP-cmake.git
        ```
5. *Configure and compile REFPROP with REFPROP-cmake*
        ```
        cd ~/REFPROP-cmake
        rm -rf build
        mkdir build && cd build
        cmake .. -DREFPROP_FORTRAN_PATH=/opt/refprop/FORTRAN -DCMAKE_BUILD_TYPE=Release
        cmake --build . -j
        ```
6. *Copy the librefprop.so library in /opt/refprop*
        ```
        sudo cp "$(find . -name 'librefprop.so' -print -quit)" /opt/refprop/librefprop.so
        sudo chmod a+r /opt/refprop/librefprop.so
        ls -l /opt/refprop/librefprop.so
        ```
7. *Register the library*
        ```
        echo "/opt/refprop" | sudo tee /etc/ld.so.conf.d/refprop.conf >/dev/null
        sudo ldconfig
        ldconfig -p | grep -i refprop || true
        ```
8. *Test the correct working of REFPROP with a small Python script*
        ```
        python - <<'PY'
        import CoolProp.CoolProp as CP
        from CoolProp.CoolProp import PropsSI

        CP.set_config_string(CP.ALTERNATIVE_REFPROP_PATH, "/opt/refprop")
        CP.set_config_string(CP.ALTERNATIVE_REFPROP_LIBRARY_PATH, "/opt/refprop/librefprop.so")

        print("CoolProp:", CP.get_global_param_string("version"))
        print("rho water:", PropsSI("D","T",300,"P",101325,"REFPROP::Water"))
        PY
        ```
9. *Bonus: set environment variables* 
        To avoid adding the REFPROP path in every script one can indicate the necessary environment variables in .bashrc as
        ```
        export COOLPROP_ALTERNATIVE_REFPROP_PATH="/opt/refprop/"
        export COOLPROP_ALTERNATIVE_REFPROP_LIBRARY_PATH="/opt/refprop/librefprop.so"
        ```

### Added others accepted phases
The liquid, supercritical liquid, and two-phase states have been added to the accepted phases in the class DataGenerator_CoolProp in the file Data_Generation/DataGenerator_NICFD.py

### Properties computation in the two-phase region
The following properties are computed in liquid, vapor and two-phase regions through a density-internal energy 2D grid:
1. Temperature.
2. Pressure.
3. Speed of sound^2. In two-phase region is computed with the forward difference of the definition (dP/drho)@ s=const.
4. Entropy.
5. Vapor quality.
6. dP/drho @ e=const. Computed with forward difference in two-phase region by assigning a dP+e=const.
7. dP/de @ rho=const. Computed with forward difference in two-phase region by assigning a dP+rho=const.
8. dh/drho @ e=const. Computed with forward difference in two-phase region by assigning a dP+e=const.
9. dh/de @ rho=const. Computed with forward difference in two-phase region by assigning a dP+rho=const.
10. dh/dP @ rho=const. Computed with forward difference in two-phase region by assigning a dP+rho=const.
11. dh/drho @ P=const. Computed with forward difference in two-phase region by assigning a dh+P=const.
12. ds/drho @ e=const. Computed with forward difference in two-phase region by assigning a dP+e=const.
13. ds/de @ rho=const. Computed with forward difference in two-phase region by assigning a dP+rho=const.
14. ds/dP @ rho=const. Computed with forward difference in two-phase region by assigning a dP+rho=const.
15. ds/drho @ P=const. Computed with forward difference in two-phase region by assigning a drho+P=const.
16. Specific heat at constant pressure. Computed as Cp=alpha*Cp,vap+(1-alpha)*Cp,liq, where alpha is the vapor void fraction, in the two-phase region.
17. Specific heat at constant volume. Computed as Cv=alpha*Cv,vap+(1-alpha)*Cv,liq, where alpha is the vapor void fraction, in the two-phase region.
18. Enthalpy is added in the LuT as its definition h=e+P/rho.
19. dT/drho @ e=const. Computed with forward difference in two-phase region by assigning a dP+e=const.
20. dT/de @ rho=const. Computed with forward difference in two-phase region by assigning a dP+rho=const.

### Contour plots 
The contour plots (in P-s) of all the quantities computed by the code and saved in the LuT are drawn and saved.

### Save vtk files
The quantities saved in the lut are saved in a .vtk file. The file is saved so that in the x axis is reported the density while in the y axis the internal energy is reported

### Refinment adapted to the expected thermodynamic transformation
Through the value of rho-e imported from a .csv or .txt file it is possible to refine only the region around the expected expansion. To add a better refinment in the low density region a if has been added in the function tasked to apply this refinment, called __ApplyRefinement_exp and defined in LUTGenerators.py 

## Capabilities
The SU2 DataMiner workflow allows the user to generate fluid data and convert these into look-up tables (LUT) or multi-layer perceptrons (MLP) for usage in SU2 simulations. The types of simulations for which this workflow is suitable are flamelet-generated manifold (FGM) and non-ideal computational fluid dynamics (NICFD) simulations. This tool allows the user to start from scratch and end up with a table input file or a set of MLP input files which can immediately be used within SU2. 

## Requirements and Set-Up
The SU2 DataMiner tool is python-based and was generated with python 3.13. Currently only Linux distributions are supported.
To install the required python modules, navigate to the SU2 DataMiner source code directory and run the following command:
```
python -m pip install -r required_packages.txt
```
Alternatively, a suitable conda environment can be created using the [environment recipe](environment.yml) through the following command:
```
conda env create -f environment.yml
```

After cloning this repository, add the following lines to your ```~/.bashrc``` in order to update your pythonpath accordingly:

```
export PINNTRAINING_HOME=<PATH_TO_SOURCE>
export PYTHONPATH=$PYTHONPATH:$PINNTRAINING_HOME
export PATH=$PATH:$PINNTRAINING_HOME/bin
``` 

where ```<PATH_TO_SOURCE>``` is the path to where you cloned the repository.

Tutorials can be found under ```TestCases```, proper documentation will follow soon.

## HOW TO OBTAIN THE NECESSARY FUILE FOR THE ADAPTED REFINMENT
1. If the results of a CFD simulation are available extract only the density and internal energy along a meaningfull path, such as the axis of symmetry for a nozzle
2. If no CFD results are available the script Predict_Expansion.py may be used

## Predict_Expansion SCRIPT
This script allow to predict the expected (isoentropic) expansion given the inlet total conditions and the outlet pressure to print a .txt for the adapted refinment (if no CFD info are available). For convenience the created .txt has a similar structure of the .csv saved by Paraview

## Getting Started

Generating a fluid data manifold for FGM or NICFD applications consists of the following steps:

1. *Generate a configuration*: The manifold settings such as the type of fluid, storage directory and range are stored in a configuration class. Configurations can be defined through python (example scripts are found in the TestCases folder, named ```generate_config.py```) or interactively through terminal inputs. In order to generate a configuration interactively, run the command ```GenerateConfig.py``` in the terminal.

2. *Generate fluid data*: Raw fluid data can be generated once the configuration is defined. Similarly to the configuration set-up, fluid data can be generated through a python interface enabling more flexibility, or through the terminal. Run the command ```GenerateFlameletData.py -h``` to see the available options. Optionally, flamelet data can be visualized through the ```PlotFlamelets.py``` command. 

3. *Process fluid data*: Raw fluid data needs to be processed in order to be converted into a manifold usable in SU2. Especially a flamelet-based manifold requires additional steps to convert raw flamelet data into a usable manifold. These steps include optimizing the progress variable, homogenizing the flamelet data, and grouping the various flamelet data into groups of high correlation. TODO: write executable for this step.

4. *Generate manifold*: The processed fluid data is ready for conversion into a manifold for SU2 simulations. The available formats are the look-up table (LUT) and multi-layer perceptron (MLP).

