#!/usr/bin/env python
# coding: utf-8

import numpy as np
from qcalc.core.ChargeDistributionMethod import ChargeDistributionMethod

class EEM (ChargeDistributionMethod):

    # parameter specifications
    paramSpec = ["electronegativity", "hardness"]
    bondParamSpec = []
    
    def __init__ (self, connectivity, distanceMatrix, diameters, hardness, electronegativity, netCharge=0, maxOrder=2, fpepsi=False):
        
        super().__init__(connectivity, distanceMatrix, diameters, netCharge, maxOrder, fpepsi)
        
        self.checkDim(hardness, self.N)
        self.checkDim(electronegativity, self.N)
        
        self.hardness = hardness
        self.electronegativity = electronegativity
    
    
    def solve(self, JMatrix):
        # prepare augmented hardness matrix
        N = self.N
        X = np.zeros((N + 1, N + 1))
        X[:,:-1][:-1] = JMatrix
        X[-1] = 1
        X[:,-1] = -1
        X[-1,-1] = 0
    
        # prepare vector with electronegativities
        Y = np.zeros(N + 1)
        Y[:-1] = -self.electronegativity
        Y[-1] = self.netCharge
    
        # solve system of equations
        res = np.linalg.solve(X, Y)
        charges = res[:-1]
        electronegativityEq = res[-1]
    
        return charges, electronegativityEq
    
    
    # N+1 x N+1 system of equations
    # returns charges, electronegativityEq
    def compute (self):
        
        # atomic J Matrix and Coulomb integrals
        self.coulomb = self.coulombIntegrals()
        self.JMatrix = np.diag(self.hardness) + self.coulomb
        self.charges, self.electronegativityEq = self.solve(self.JMatrix)
        return self.charges

    def setIndices(self, indices, ntypes):
        self.elnegIndices = indices
        self.hardnessIndices = indices + ntypes

    # for optimization
    def setParams(self, paramsArr):
        self.electronegativity = paramsArr[self.elnegIndices]
        self.hardness = paramsArr[self.hardnessIndices]