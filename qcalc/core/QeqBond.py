#!/usr/bin/env python
# coding: utf-8

import numpy as np
from qcalc.core.BondChargeDistributionMethod import BondChargeDistributionMethod

class QEqBond (BondChargeDistributionMethod):

    # parameter specifications
    paramSpec = ["electronegativity", "hardness"]
    bondParamSpec = []
    
    def __init__(self, connectivity, distanceMatrix, diameters, hardness, electronegativity, \
            chargeTransferTopology, netCharge=0, maxOrder=1, fpepsi=False):
            
        super().__init__(connectivity, distanceMatrix, diameters, chargeTransferTopology, netCharge, maxOrder, fpepsi)
            
        self.checkDim(hardness, self.N)
        self.checkDim(electronegativity, self.N)
        
        self.hardness = hardness
        self.electronegativity = electronegativity
            
            
    def compute(self):
            
        # atomic J Matrix
        self.coulomb = self.coulombIntegrals()
        self.JMatrix = np.diag(self.hardness) + self.coulomb
            
        # transform to bond variables
        self.bVars = self.bondVars()
        self.bondElneg = self.bondElectronegativity(self.electronegativity, self.bVars)
        self.bondJMatrix = self.calcBondJMatrix(self.JMatrix, self.bVars)
            
        # solve system
        self.bondCharges = self.solve(self.bondElneg, self.bondJMatrix)
        self.charges = self.toAtomicCharges(self.bondCharges, self.bVars)
        return self.charges

    def setIndices(self, indices, ntypes):
        self.elnegIndices = indices
        self.hardnessIndices = indices + ntypes

    # for optimization
    def setParams(self, paramsArr):
        self.electronegativity = paramsArr[self.elnegIndices]
        self.hardness = paramsArr[self.hardnessIndices]