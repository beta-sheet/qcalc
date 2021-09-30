#!/usr/bin/env python
# coding: utf-8

import numpy as np
from qcalc.core.EEM import EEM
from qcalc.core.QeqAtomic import QEqAtomic
from qcalc.core.QeqBond import QEqBond
from qcalc.core.AACT import AACT
from qcalc.core.SQE import SQE
from qcalc.util.utils import getConnectivity
from qcalc.util.rdkitUtils import extractMol, getAtomLabels, getBondLabels
from qcalc.parameter import Parameter

def computeCharges (mol, paramTable, method, atomTypeFunc, **kwargs):
    params, bondParams = createParameters(paramTable, method, **kwargs)
    worker = createWorker(mol, params, method, atomTypeFunc, bondParams=bondParams, **kwargs)
    charges = worker.compute()
    return charges

# called only once
def createParameters(paramTable, method, **kwargs):

    # create atomic params
    params = Parameter(paramTable)

    # for bond methods: create bond parasm
    if method == "AACT" or method == "SQE":
        bondParamTable = kwargs.get("bondParamTable", None)
        if bondParamTable is None:
            raise Exception("bondParams is required for AACT and SQE")
        bondParams = Parameter(bondParamTable)
    else:
        bondParams = None

    # assign specs (rows in params table which will be converted to array for opt)  
    if method == "EEM":
        params.setParamSpec(EEM.paramSpec)
    elif method == "Qeq" or method == 'QeqAtomic':
        params.setParamSpec(QEqAtomic.paramSpec)
    elif method == "QeqBond":
        params.setParamSpec(QEqBond.paramSpec)
    elif method == 'AACT':
        params.setParamSpec(AACT.paramSpec)
        bondParams.setParamSpec(AACT.bondParamSpec)
    elif method == "SQE":
        params.setParamSpec(SQE.paramSpec)
        bondParams.setParamSpec(SQE.bondParamSpec)
    else:
        raise Exception("method " + method + "is undefined")

    # return object
    return params, bondParams

# called once for every molecule
def createWorker (mol, params, method, atomTypeFunc, **kwargs):

    # optional parameters
    netCharge = kwargs.get("netCharge", 0)
    maxOrder = kwargs.get("maxOrder", 1)
    fpepsi = kwargs.get("fpepsi", False)
    bondParams = kwargs.get("bondParams", None)

    # extract info from molecule
    molDict = extractMol(mol)
    atoms = molDict["atoms"]
    connectivity = molDict["connectivity"]
    distanceMatrix = molDict["distanceMatrix"]

    # function to assign bond types
    bondTypeFunc = kwargs.get("bondTypeFunc", None)

    # charge transfer topology: coupling over bonds
    chargeTransferFilter = lambda x: 1 if (x <= 1) else 0
    chargeTransferTopology = np.vectorize(chargeTransferFilter)(connectivity)

    # initialize atom types
    atomTypes = getAtomLabels(mol, atomTypeFunc)

    # extract atomic params
    electronegativity = params.extractProp(atomTypes, "atom", "electronegativity")
    hardness = params.extractProp(atomTypes, "atom", "hardness")
    diameters = params.extractProp(atomTypes, "atom", "diameter")

    # here we also need bond parameters
    if method == "AACT" or method == "SQE":
        if bondParams is None:
            raise Exception("bondParams not provided. This is most likely a bug in the top-level function.")
        if bondTypeFunc is None:
            raise Exception("bondTypeFunc is required for AACT and SQE")
        bondTypes = getBondLabels(mol, bondTypeFunc)
        bondElneg = bondParams.extractProp(bondTypes, "type", "electronegativity")
        bondHardness = bondParams.extractProp(bondTypes, "type", "hardness")
    

    # initialize method
    if method == "EEM":
        worker = EEM(connectivity, distanceMatrix, diameters, hardness, electronegativity, netCharge, maxOrder, fpepsi)
    elif method == "Qeq" or method == "QeqAtomic":
        worker = QEqAtomic(connectivity, distanceMatrix, diameters, hardness, electronegativity, netCharge, maxOrder, fpepsi)
    elif method == "QeqBond":
        worker = QEqBond(connectivity, distanceMatrix, diameters, hardness, electronegativity, chargeTransferTopology, netCharge, maxOrder, fpepsi)
    elif method == "AACT":
        worker = AACT(connectivity, distanceMatrix, diameters, bondHardness, electronegativity, chargeTransferTopology, netCharge, maxOrder, fpepsi)
    elif method == "SQE":
        # here we also need kappa and lam
        kappa = kwargs.get("kappa", 1)
        lam = kwargs.get("lam", 1)
        worker = SQE(connectivity, distanceMatrix, diameters, hardness, bondHardness, electronegativity, chargeTransferTopology, kappa, lam, netCharge, maxOrder, fpepsi)
    else:
        raise Exception("method " + method + "is undefined")

    # needed for custom weighting schemes
    worker.atomTypes = atomTypes

    # keep track of indices for optimization
    indices = params.extractRows(atomTypes, "atom")

    if method == "AACT" or method == "SQE":
        bondIndices = bondParams.extractRows(bondTypes, "type")
        worker.setIndices(indices, bondIndices, params.N)
    else:
        worker.setIndices(indices, params.N)

    return worker

if __name__ == "__main__":
    print("in main")