#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

class Parameter:

    def __init__(self, df):
        self.df = df.copy()
        self.N = df.shape[0]

        self.hardnessLabel = "hardness"
        self.elnegLabel = "electronegativity"
        self.bondHardnessLabel = "hardness"
        self.bondElnegLabel = 'electronegativity'
        self.diameterLabel = 'diameters'
        self.constraints = []

    def setParamSpec(self, paramSpec):
        self.paramSpec = paramSpec
        self.M = len(paramSpec)

    def addConstraint(self, atom, prop, id="atom"):
        propIndex = self.paramSpec.index(prop)
        atomIndex = self.df.index[self.df[id] == atom].tolist()[0]
        value = self.df.iloc[atomIndex][prop]
        self.constraints.append((self.N * propIndex + atomIndex, value))
        

    def extractRows(self, types, typeLabel):
        rows = np.array([self.df.index[self.df[typeLabel] == a][0] for a in types])
        return rows

    def extractProp(self, types, typeLabel, prop):
        rows = self.extractRows(types, typeLabel)
        col = self.df.iloc[rows,:][prop]
        return np.array(col)

    def toArray(self):
        arr = np.array([])
        for colName in self.paramSpec:
            col = self.df[colName].to_numpy()
            arr = np.concatenate((arr, col), axis=0)
        # add constraints
        constrIndices = [c[0] for c in self.constraints]
        constrValues = [c[1] for c in self.constraints]
        constrArr = np.delete(arr, constrIndices)
        mask = list(range(0,len(constrIndices)))
        updConstrIndices = [ci-m for ci,m in zip(constrIndices, mask)]
        return constrArr, (updConstrIndices, constrValues)

    def arrayIndices(self, types, typeLabel, prop):
        rows = self.extractRows(types, typeLabel)
        propIndex = self.paramSpec.index(prop)
        indices = propIndex * self.N + rows
        return indices

    def update(self, constrArr):
        # add constrained values
        constrIndices = np.array([c[0] for c in self.constraints], dtype="int")
        constrValues = np.array([c[1] for c in self.constraints])
        mask = np.arange(0,len(constrIndices))
        updConstrIndices = constrIndices - mask
        arr = np.insert(constrArr, updConstrIndices, constrValues)
        # update df
        start = 0
        for colName in self.paramSpec:
            col = arr[start:start+self.N]
            self.df[colName] = col
            start += self.N
