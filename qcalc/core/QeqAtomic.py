#!/usr/bin/env python
# coding: utf-8

import numpy as np
from qcalc.core.ChargeDistributionMethod import ChargeDistributionMethod

class QEqAtomic (ChargeDistributionMethod):
        
    def __init__ (self, connectivity, distanceMatrix, diameters, hardness, electronegativity, netCharge=0, maxOrder=1):
        
        super().__init__(connectivity, distanceMatrix, diameters, netCharge, maxOrder)
        
        self.checkDim(hardness, self.N)
        self.checkDim(electronegativity, self.N)
        
        self.hardness = hardness
        self.electronegativity = electronegativity
        
   
    def solve (self, JMatrix):
        # prepare hardness matrix
        N = self.N
        X = JMatrix.copy()
        X = X - X[0]
        X[0] = 1
    
        # prepare vector with electronegativity differences
        Y = -self.electronegativity.copy()
        Y = Y - Y[0]
        Y[0] = self.netCharge
    
        # solve system of equations
        charges = np.linalg.solve(X, Y)
        return charges

        
    # N x N system of equations
    # returns charges
    def compute (self):
    
        ## same as for EEM
        # compute Coulomb integrals
        self.coulomb = self.coulombIntegrals()
        self.JMatrix = np.diag(self.hardness) + self.coulomb
        self.charges = self.solve(self.JMatrix)
    
        return self.charges