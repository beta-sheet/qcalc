#!/usr/bin/env python
# coding: utf-8

import numpy as np
from qcalc.core.BondChargeDistributionMethod import BondChargeDistributionMethod

class AACT (BondChargeDistributionMethod):

    # parameter specifications
    paramSpec = ["electronegativity"]
    bondParamSpec = ["hardness"]
    
    def __init__(self, connectivity, distanceMatrix, diameters, bondHardness, electronegativity, chargeTransferTopology, \
            netCharge=0, maxOrder=1, fpepsi=False):
            
        super().__init__(connectivity, distanceMatrix, diameters, chargeTransferTopology, netCharge, maxOrder, fpepsi)
            
        self.checkDim(electronegativity, self.N)
        
        self.bondHardness = bondHardness
        self.electronegativity = electronegativity

        # 2B overwritten in compute
        self.B = len(bondHardness)
        
    
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

    def setIndices(self, indices, bondIndices, ntypes):
        self.elnegIndices = indices
        self.bondHardnessIndices = bondIndices + ntypes

    # for optimization
    def setParams(self, paramsArr):
        self.electronegativity = paramsArr[self.elnegIndices]
        self.bondHardness = paramsArr[self.bondHardnessIndices]
