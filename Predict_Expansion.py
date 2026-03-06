import numpy as np
import matplotlib.pyplot as plt
import CoolProp as CP
from CoolProp.CoolProp import PropsSI
import os

def comp_expected_exp(Pin,h_in,Pout,Model,fluid,n_discr, SaveFolder, SaveFile):

    HEOS=CP.AbstractState(Model, fluid)
    
    Pvect=np.linspace(Pin*1e5,Pout*1e5,n_discr)
    
    HEOS.update(CP.HmassP_INPUTS, h_in, Pin*1e5)
    s_in=HEOS.smass()

    e_vect,rho_vect=np.zeros(n_discr),np.zeros(n_discr)

    for i in range(n_discr):
        HEOS.update(CP.PSmass_INPUTS, Pvect[i], s_in)
        e_vect[i]=HEOS.umass()
        rho_vect[i]=HEOS.rhomass()

    output_file = open(f"{SaveFolder}/{SaveFile}", "w")
    output_file.write("%s, %s \n" %("rho [kg/m3]", "e [J/kg]"))
    
    for i in range(n_discr):
        output_file.write("%.5f, %.5f \n" %(rho_vect[i], e_vect[i]))

    output_file.close()

    fig,ax=plt.subplots()
    ax.plot(rho_vect,e_vect)
    ax.set_xlabel("Density [kg/m3]")
    ax.set_ylabel("Internal Energy [J/kg]")
    ax.grid(ls=":", c="lightgray")

    fig.set_tight_layout(True)
    
    plt.show(block=True)
    plt.savefig(f"{SaveFolder}/PredictedExp.pdf", dpi=600)
    plt.savefig(f"{SaveFolder}/PredictedExp.svg", dpi=600)

    return

if __name__ == "__main__":

    ######## USER'S INPUTS ########
    fluid = "MM"
    Model="REFPROP"
    Pin=19.31 # bar
    Tin=220.6 # degC
    TwoPH_inlet=False # If true, h_in is employed instead of Tin to compute the total inlet conditions
    h_in=None
    Pout=1.37 # bar

    SaveFolder="TROVA_New_Nozzle_Des_Exp"
    SaveFile="TROVA_New_Nozzle_Des_Expected_Exp.txt"

    n_discr=1000 # Number of points for the discretization of the expansion process

    ######## MAIN CODE ########
    if os.path.isdir(SaveFolder) is False:
        os.mkdir(SaveFolder)

    if not TwoPH_inlet:
        h_in=PropsSI('H','P',Pin*1e5,'T',(Tin+273.15),f"{Model}::{fluid}")

    comp_expected_exp(Pin,h_in,Pout,Model,fluid,n_discr, SaveFolder, SaveFile)


    