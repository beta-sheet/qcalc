#!/usr/bin/env python
# coding: utf-8

import numpy as np
from qcalc.core.ChargeDistributionMethod import ChargeDistributionMethod

class BondChargeDistributionMethod (ChargeDistributionMethod):
    
    def __init__(self, connectivity, distanceMatrix, diameters, chargeTransferTopology, netCharge=0, maxOrder=1, fpepsi=False):
            
        super().__init__(connectivity, distanceMatrix, diameters, netCharge, maxOrder, fpepsi)
        
        self.checkDim(chargeTransferTopology, self.N)
        self.chargeTransferTopology = chargeTransferTopology
        
        
    # get bond variable definitions as pairs of indices
    def bondVars (self):
        bVars = np.argwhere(self.chargeTransferTopology)
        upperTriangle = np.where(bVars[:,0] < bVars[:,1])
        bVars = bVars[upperTriangle]
        self.B = len(bVars)
        return bVars

    
    # map bond charges to atoms
    def toAtomicCharges(self, bondCharges, bVars):
        charges = np.repeat(self.netCharge / self.N, self.N)
        for b, [i, j] in enumerate(bVars):
            charges[i] = charges[i] - bondCharges[b]
            charges[j] = charges[j] + bondCharges[b]
        return charges  

    
    # electronegativity vector in bond variables
    def bondElectronegativity (self, electronegativity, bVars): 
        bondElneg = np.zeros(self.B)  
        for b, [i, j] in enumerate(bVars):
            bondElneg[b] = electronegativity[j] - electronegativity[i]  
        return bondElneg   
 

    # transform J Matrix to bond space
    # hardness: 1D vector (N)
    # res: B x B (B: #entries in upper triangle of CTT matrix)
    def calcBondJMatrix (self, JMatrix, bVars):
    
        # initialize B x B hardness matrix
        B = self.B
        bondJMatrix = np.zeros((B, B))
    
        # construct hardness matrix
        for b1, [i, j] in enumerate(bVars):
            for b2, [k, l] in enumerate(bVars):
                bondJMatrix[b1, b2] = JMatrix[i,k] - JMatrix[i,l] - JMatrix[j,k] + JMatrix[j,l]
        
        return bondJMatrix
    
    
    # B x B system of equations
    # returns bond charges
    def solve (self, bondElneg, bondJMatrix):
    
        # system only has an unique solution for N-1 bond variables
        # otherwise we use SVD to get a particular solution instead
        if self.B <= self.N - 1:
            bondCharges = np.linalg.solve(bondJMatrix, -bondElneg)
        else: 
            # https://stackoverflow.com/questions/59292279/solving-linear-systems-of-equations-with-svd-decomposition
            U, s, Vh = np.linalg.svd(bondJMatrix)
            c = np.dot(U.T, -bondElneg)
            w = np.dot(np.diag(1/s), c)
            bondCharges = np.dot(Vh.conj().T, w)
        
        return bondCharges