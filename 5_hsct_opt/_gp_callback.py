__all__ = [
    "gp_callback_generator",
]

"""
Define the TACS constitutive and element objects in the structural analysis.
Also setup the GPBladeConstitutive object to use closed-form solutions not the GP models in this

Author: Sean Engelstad
"""

from tacs import elements, constitutive, TACS
import numpy as np
import ml_buckling as mlb

dtype = TACS.dtype


# ==============================================================================
# Element callback function
# ==============================================================================
def gp_callback_generator(tacs_component_names):
    """
    method to build the gp callback at runtime using the tacs component names
    """

    # build one Axial and Shear GP model to be used for all const objects (no duplication)
    axialGP = constitutive.AxialGP.from_csv(csv_file=mlb.axialGP_csv)
    shearGP = constitutive.ShearGP.from_csv(csv_file=mlb.shearGP_csv)

    # now build a dictionary of PanelGP objects which manage the GP for each tacs component/panel
    panelGP_dict = constitutive.PanelGPs.component_dict(tacs_component_names, axialGP=axialGP, shearGP=shearGP)

    def gp_callback(
        dvNum, compID, compDescript, elemDescripts, specialDVs, **kwargs
    ):

        # get the panelGPs object associated with this tacs component
        panelGPs = panelGP_dict[compDescript]

        # need to use orthotropic ply for this class, but
        # use isotropic properties though
        E = 114e9
        nu = 0.361
        G = E / 2.0 / (1+nu)
    
        ortho_prop = constitutive.MaterialProperties(
            rho=4.43e3,  # density kg/m^3
            E1=114e9, # Young's modulus in 11 direction (Pa)
            E2=114e9, # Young's modulus in 22 direction (Pa)
            G12=G, # in-plane 1-2 shear modulus (Pa)
            G13=G, # Transverse 1-3 shear modulus (Pa)
            G23=G, # Transverse 2-3 shear modulus (Pa)
            nu12=0.361,  # 1-2 poisson's ratio
            T1=880e6,  # Tensile strength in 1 direction (Pa)
            C1=880e6,  # Compressive strength in 1 direction (Pa)
            T2=880e6,  # Tensile strength in 2 direction (Pa)
            C2=880e6,  # Compressive strength in 2 direction (Pa)
            S12=880e6,  # Shear strength direction (Pa)
        )
        ortho_ply = constitutive.OrthotropicPly(1.0, ortho_prop)
    
        # case by case initial ply angles
        if "OML" in compDescript:
            refAxis = np.array([0.0, 1.0, 0.0])
        else:
            refAxis = np.array([0.0, 0.0, 1.0])
    
    
        # The ordering of the DVs used by the GPBladeStiffenedShell model is:
        # - panel length
        # - stiffener pitch
        # - panel thickness
        # - panel ply fractions (not used in this case)
        # - stiffener height
        # - stiffener thickness
        # - stiffener ply fractions (not used in this case)
        # - panel width

        # create the design variable scales array
        DVscales = [
            1.0, # panel length
            1.0, # stiffener pitch
            100.0, # panel thickness
            10.0, # stiffener height
            100.0, # stiffener thickness
            1.0, # panel width
        ]
        # TBD can add panel and stifener ply fractions to the DVs

        # don't put in any GP models (so using closed-form solutions rn)
        con = constitutive.GPBladeStiffenedShellConstitutive(
            panelPly=ortho_ply,
            stiffenerPly=ortho_ply,
            panelLength=0.5, # choose wrong initial value first to check if it corrects in FUNtoFEM
            stiffenerPitch=0.2,
            panelThick=1.5e-2,
            panelPlyAngles=np.array([0.0], dtype=dtype),
            panelPlyFracs=np.array([1.0], dtype=dtype),
            stiffenerHeight=0.075,
            stiffenerThick=1e-2,
            stiffenerPlyAngles=np.array([0.0], dtype=dtype),
            stiffenerPlyFracs=np.array([1.0], dtype=dtype),
            panelWidth=0.5, # choose wrong initial value first to check if it corrects in FUNtoFEM
            flangeFraction=0.8,
            panelLengthNum=dvNum,
            stiffenerPitchNum=dvNum+1,
            panelThickNum=dvNum+2,
            stiffenerHeightNum=dvNum+3,
            stiffenerThickNum=dvNum+4,
            panelWidthNum=dvNum+5,
            panelGPs=panelGPs,
        )
        # Set the KS weight really low so that all failure modes make a
        # significant contribution to the failure function derivatives
        con.setKSWeight(100.0)  #20.0

        con.setStiffenerPitchBounds(0.05, 0.5)
        con.setPanelThicknessBounds(0.002, 0.1)
        con.setStiffenerHeightBounds(0.002, 0.1)
        con.setStiffenerThicknessBounds(0.002, 0.1)

        # --- Create reference axis transform to define the stiffener direction ---
        transform = elements.ShellRefAxisTransform(refAxis)

        # --- Create the element object ---
        elemList = []
        for descript in elemDescripts:
            if descript == "CQUAD4":
                elem = elements.Quad4Shell(transform, con)
            elif descript == "CTRIA3":
                elem = elements.Tri3Shell(transform, con)

            elemList += [elem]

        return elemList, DVscales
    return gp_callback
