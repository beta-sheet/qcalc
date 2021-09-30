#!/usr/bin/env python
# coding: utf-8

import numpy as np
from qcalc.core.EEM import EEM
from qcalc.core.QeqAtomic import QEqAtomic
from qcalc.core.QeqBond import QEqBond
from qcalc.core.AACT import AACT
from qcalc.core.SQE import SQE
from qcalc.util.utils import getConnectivity
from qcalc.util.rdkitUtils import extractMol, extractParams, extractBondParams

def computeCharges (mol, params, method, atomTypeFunc, **kwargs):

    # optional parameters
    netCharge = kwargs.get("netCharge", 0)
    maxOrder = kwargs.get("maxOrder", 1)
    bondParams = kwargs.get("bondParams", None)

    # non-default column names in parameter table
    atomLabel = kwargs.get("atomLabel", "atom")
    hardnessLabel = kwargs.get("hardnessLabel", "hardness")
    elnegLabel = kwargs.get("electronegativityLabel", "electronegativity")
    diameterLabel = kwargs.get("diameterLabel", "diameter")

    # non-default column names in bond parameter table
    typeLabel = kwargs.get("typeLabel", "type")
    bondHardnessLabel = kwargs.get("bondHardnessLabel", "hardness")
    bondElnegLabel = kwargs.get("bondElectronegativityLabel", "electronegativity")

    # extract info from molecule
    molDict = extractMol(mol)
    atoms = molDict["atoms"]
    connectivity = molDict["connectivity"]
    distanceMatrix = molDict["distanceMatrix"]

    # function to assign bond types
    bondTypeFunc = kwargs.get("bondTypeFunc", None)

    # charge transfer topology: coupling over bonds
    chargeTransferFilter = lambda x: 1 if (x <= maxOrder) else 0
    chargeTransferTopology = np.vectorize(chargeTransferFilter)(connectivity)

    # extract atomic params
    electronegativity, hardness, diameters = extractParams(mol, params, atomTypeFunc, atomLabel, elnegLabel, hardnessLabel, diameterLabel)

    # here we also need bond parameters
    if method == "AACT" or method == "SQE":
        if bondParams is None:
            raise Exception("bondParams is required for AACT and SQE")
        bondElneg, bondHardness = extractBondParams(mol, bondParams, bondTypeFunc, typeLabel, bondElnegLabel, bondHardnessLabel)
        if bondTypeFunc is None:
            raise Exception("bondTypeFunc is required for AACT and SQE")

    # initialize method
    if method == "EEM":
        worker = EEM(connectivity, distanceMatrix, diameters, hardness, electronegativity, netCharge, maxOrder)
    elif method == "Qeq" or method == "QeqAtomic":
        worker = QEqAtomic(connectivity, distanceMatrix, diameters, hardness, electronegativity, netCharge, maxOrder)
    elif method == "QeqBond":
        worker = QEqBond(connectivity, distanceMatrix, diameters, hardness, electronegativity, chargeTransferTopology, netCharge, maxOrder)
    elif method == "AACT":
        worker = AACT(connectivity, distanceMatrix, diameters, bondHardness, electronegativity, chargeTransferTopology, netCharge, maxOrder)
    elif method == "SQE":
        # here we also need kappa and lam
        kappa = kwargs.get("kappa", 1)
        lam = kwargs.get("lam", 1)
        worker = SQE(connectivity, distanceMatrix, diameters, hardness, bondHardness, electronegativity, chargeTransferTopology, kappa, lam, netCharge, maxOrder)
    else:
        raise Exception("method " + method + "is undefined")

    # compute the charges
    charges = worker.compute()
    return worker