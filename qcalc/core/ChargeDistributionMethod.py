#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy.special import erf

class ChargeDistributionMethod:
    
    def __init__(self, connectivity, distanceMatrix, diameters, netCharge=0, maxOrder=1, fpepsi=False):
        # some basic error checking
        N = connectivity.shape[0]
        self.checkDim(connectivity, N)
        self.checkDim(distanceMatrix, N)
        self.checkDim(diameters, N)
        
        self.connectivity = connectivity
        self.distanceMatrix = distanceMatrix
        self.diameters = diameters
        self.maxOrder = maxOrder
        self.fpepsi = fpepsi
        self.N = N
        self.netCharge = netCharge
        
    
    # generic material for dimensionality checks
    def checkDim (self, obj, N):
        if len(obj.shape) == 1:
            assert len(obj) == N, "Error: got vector of length " + str(len(obj)) + ", expected " + str(N)
        else: # vec is a matrix
            assert obj.shape == (N, N), "Error: got matrix of shape " + str(obj.shape) + ", expected (" + str(N) + "," + str(N) + ")"
        
        
    def coulombIntegrals (self):
        if self.fpepsi:
            FPEPSI = 1.44
        else:
            FPEPSI = 1.
        N = self.N
        # calculate Coulomb integrals
        coulomb = np.zeros((N, N))
        for i in range(0,N):
            for j in range(i+1,N):
                if self.connectivity[i,j] <= self.maxOrder:
                    coulomb[i,j] = FPEPSI / self.distanceMatrix[i,j] * erf( self.distanceMatrix[i,j] \
                        / np.sqrt( self.diameters[i]**2 + self.diameters[j]**2 ))
                    coulomb[j,i] = coulomb[i,j]
        return coulomb

