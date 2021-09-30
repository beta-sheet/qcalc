#!/usr/bin/env python
# coding: utf-8

import numpy as np
from qcalc.core.BondChargeDistributionMethod import BondChargeDistributionMethod

class SQE (BondChargeDistributionMethod):
    
    def __init__(self, connectivity, distanceMatrix, diameters, hardness, bondHardness, electronegativity, chargeTransferTopology, \
            kappa=1, lam=1, netCharge=0, maxOrder=1):
            
        super().__init__(connectivity, distanceMatrix, diameters, chargeTransferTopology, netCharge, maxOrder)
            
        self.checkDim(electronegativity, self.N)
        self.checkDim(hardness, self.N)
        
        self.bondHardness = bondHardness
        self.hardness = hardness
        self.electronegativity = electronegativity
        self.kappa = kappa
        self.lam = lam
        
        
    def compute (self):
        
        # atomic J matrix with diagonal scaled by lam^2
        self.coulomb = self.coulombIntegrals()
        scalingFactor1 = self.lam * self.lam
        self.JMatrix = scalingFactor1 * np.diag(self.hardness) + self.coulomb
        
        # transform to bond variables
        self.bVars = self.bondVars()
        self.checkDim(self.bondHardness, self.B)
        self.bondElneg = self.bondElectronegativity(self.electronegativity, self.bVars)
        self.bondJMatrix = self.calcBondJMatrix(self.JMatrix, self.bVars)
        
        # add bond hardness on the diagonal, scaled by kappa^2
        scalingFactor2 = self.kappa * self.kappa
        self.bondJMatrix = self.bondJMatrix + scalingFactor2 * 2 * np.diag(self.bondHardness)
        
        # solve system
        self.bondCharges = self.solve(self.bondElneg, self.bondJMatrix)
        self.charges = self.toAtomicCharges(self.bondCharges, self.bVars)
        return self.charges