__all__ = ["CompositeMaterial"]

from .composite_material_utility import CompositeMaterialUtility

class CompositeMaterial:
    def __init__(
        self,
        E11,
        nu12,
        E22=None,
        G12=None,
        _G23=None,
        _G13=None,
        ply_angle=None,
        material_name=None,
    ):
        self.E11 = E11
        self.nu12 = nu12
        self._E22 = E22
        self._G12 = G12
        self._G23 = _G23
        self._G13 = _G13
        self.ply_angle = ply_angle
        self.material_name = material_name

    @property
    def nu21(self) -> float:
        """reversed 12 Poisson's ratio"""
        return self.nu12 * self.E22 / self.E11

    @property
    def E22(self) -> float:
        if self._E22 is None:
            return self.E11
        else:
            return self._E22

    @property
    def G12(self) -> float:
        if self._G12 is None:
            return self.E11 / 2.0 / (1 + self.nu12)
        else:
            return self._G12

    @classmethod
    def get_materials(cls):
        return [
            cls.solvay5320,
            cls.solvayMTM45,
            cls.torayBT250E,
            cls.hexcelIM7,
            cls.victrexAE,
        ]
    
    @classmethod
    def get_material_from_str(cls, mat_name:str):
        method_names = [_.__qualname__ for _ in cls.get_materials()]
        materials = cls.get_materials()
        _method = None
        for i,method_name in enumerate(method_names):
            if mat_name in method_name:
                _method = materials[i]
        assert _method is not None
        return _method

    # MATERIALS CLASS METHODS
    # -----------------------------------------------------------

    # NIAR composite materials

    @classmethod
    def solvay5320(cls, ply_angle=0.0):
        """
        NIAR dataset - Solvay 5320-1 material (thermoset)
        Fiber: T650 unitape, Resin: Cycom 5320-1
        Room Temperature Dry (RTD) mean properties shown below
        units in Pa, ND
        """
        comp_utility = CompositeMaterialUtility(
            E11=138.461e9, E22=9.177e9, nu12=0.326, G12=4.957e9
        )
        comp_utility.rotate_ply(ply_angle)

        return cls(
            material_name="solvay5320",
            ply_angle=ply_angle,
            E11=comp_utility.E11,
            E22=comp_utility.E22,
            nu12=comp_utility.nu12,
            G12=comp_utility.G12,
        )

    @classmethod
    def solvayMTM45(cls, ply_angle=0.0):
        """
        NIAR dataset - Solvay MTM45 material (thermoset)
        Style: 12K AS4 Unidirectional
        Room Temperature Dry (RTD) mean properties shown below
        units in Pa, ND
        """
        comp_utility = CompositeMaterialUtility(
            E11=129.5e9, E22=7.936e9, nu12=0.313, G12=4.764e9
        )
        comp_utility.rotate_ply(ply_angle)

        return cls(
            material_name="solvayMTM45",
            ply_angle=ply_angle,
            E11=comp_utility.E11,
            E22=comp_utility.E22,
            nu12=comp_utility.nu12,
            G12=comp_utility.G12,
        )

    @classmethod
    def torayBT250E(cls, ply_angle=0.0):
        """
        NIAR dataset - Toray (formerly Tencate) BT250E-6 S2 Unitape Gr 284 material (thermoset)
        Room Temperature Dry (RTD) mean properties shown below
        units in Pa, ND
        """
        comp_utility = CompositeMaterialUtility(
            E11=44.74e9, E22=11.36e9, nu12=0.278, G12=3.77e9
        )
        comp_utility.rotate_ply(ply_angle)

        return cls(
            material_name="torayBT250E",
            ply_angle=ply_angle,
            E11=comp_utility.E11,
            E22=comp_utility.E22,
            nu12=comp_utility.nu12,
            G12=comp_utility.G12,
        )

    @classmethod
    def victrexAE(cls, ply_angle=0.0):
        """
        NIAR dataset - Victrex AE 250 LMPAEK (thermoplastic)
        Room Temperature Dry (RTD) mean properties shown below
        units in Pa, ND
        """
        comp_utility = CompositeMaterialUtility(
            E11=131.69e9, E22=9.694e9, nu12=0.3192, G12=4.524e9
        )
        comp_utility.rotate_ply(ply_angle)

        return cls(
            material_name="victrexAE",
            ply_angle=ply_angle,
            E11=comp_utility.E11,
            E22=comp_utility.E22,
            nu12=comp_utility.nu12,
            G12=comp_utility.G12,
        )

    @classmethod
    def hexcelIM7(cls, ply_angle=0.0):
        """
        NIAR dataset - Hexcel 8552 IM7 Unidirectional Prepreg (thermoset)
        Room Temperature Dry (RTD) mean properties shown below
        units in Pa, ND
        """
        comp_utility = CompositeMaterialUtility(
            E11=158.51e9, nu12=0.316, E22=8.96e9, G12=4.688e9
        )
        comp_utility.rotate_ply(ply_angle)

        return cls(
            material_name="hexcelIM7",
            ply_angle=ply_angle,
            E11=comp_utility.E11,
            E22=comp_utility.E22,
            nu12=comp_utility.nu12,
            G12=comp_utility.G12,
        )