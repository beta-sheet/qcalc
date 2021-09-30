#!/usr/bin/env python
# coding: utf-8

import numpy as np
from qcalc.core.BondChargeDistributionMethod import BondChargeDistributionMethod

class AACT (BondChargeDistributionMethod):
    
    def __init__(self, connectivity, distanceMatrix, diameters, bondHardness, electronegativity, chargeTransferTopology, \
            netCharge=0, maxOrder=1):
            
        super().__init__(connectivity, distanceMatrix, diameters, chargeTransferTopology, netCharge, maxOrder)
            
        self.checkDim(electronegativity, self.N)
        
        self.bondHardness = bondHardness
        self.electronegativity = electronegativity
        
    
    def compute (self):
        
        # here, the atomic J matrix has 0 on the diagonal
        self.JMatrix = self.coulombIntegrals()
        
        # transform to bond variables
        self.bVars = self.bondVars()
        self.checkDim(self.bondHardness, self.B)
        self.bondElneg = self.bondElectronegativity(self.electronegativity, self.bVars)
        self.bondJMatrix = self.calcBondJMatrix(self.JMatrix, self.bVars)
        
        # add bond hardness on the diagonal
        self.bondJMatrix = self.bondJMatrix + 2 * np.diag(self.bondHardness)
        
        # solve system
        self.bondCharges = self.solve(self.bondElneg, self.bondJMatrix)
        self.charges = self.toAtomicCharges(self.bondCharges, self.bVars)
        return self.charges