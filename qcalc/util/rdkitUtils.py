from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdchem
import numpy as np
from scipy.spatial import distance_matrix
import pandas as pd
from qcalc.util.utils import getConnectivity

# add charge on H to charge of atom it is bonded to
def _addChargeH(atom, hAtom):
    try:
        chargeH = hAtom.GetDoubleProp("molFileAlias")
    except KeyError:
        chargeH = 0
    chargeSelf = atom.GetDoubleProp("charge")
    atom.SetDoubleProp("charge", chargeSelf + chargeH)
    hAtom.SetDoubleProp("charge", 0)


def extractMol(mol):
    
    m = Chem.RemoveHs(mol)
    
    # make list of atoms without Hs and store the new indices
    atoms = [a.GetSymbol() for a in m.GetAtoms()]
            
    # make list of (non-H) bonds
    bonds = []
    for b in m.GetBonds():
        b1 = b.GetBeginAtom()
        b2 = b.GetEndAtom()
        bonds.append([b1.GetIdx(), b2.GetIdx()])
            
    # create connectivity & distance matrix
    connectivity = getConnectivity(atoms, bonds)
    
    # calculate distance matrix
    conf = m.GetConformers()[0]
    positions = 0.1*conf.GetPositions()  # A -> nm
    distanceMatrix = distance_matrix(positions, positions)
        
    # store everything in a dict
    molData = dict()
    molData["atoms"] = atoms
    molData["bonds"] = bonds
    molData["connectivity"] = connectivity
    molData["distanceMatrix"] = distanceMatrix
            
    return molData   
    
def getAtomLabels(mol, atomTypeFunc):
    return np.array([atomTypeFunc(atom, mol) for atom in mol.GetAtoms()])

def getBondLabels(mol, bondTypeFunc):
    return np.array([bondTypeFunc(bond, mol) for bond in mol.GetBonds()])


def extractIndices(mol, params, atomTypeFunc, typeLabel, elnegLabel, hardnessLabel, diameterLabel):
    atomTypes = np.array([atomTypeFunc(atom, mol) for atom in mol.GetAtoms()])
    try:
        indices = np.array([params.index[params[typeLabel] == a][0] for a in atomTypes])
    except IndexError:
        print("Invalid atom type - check your atomTypeFunc!")
    return indices

# extract electronegativities, hardnesses and diameters
def extractParams (mol, params, atomTypeFunc, atomLabel, elnegLabel, hardnessLabel, diameterLabel):
    atomTypes = np.array([atomTypeFunc(atom, mol) for atom in mol.GetAtoms()])
    electronegativity = np.array([params.loc[params[atomLabel] == a][elnegLabel].to_numpy()[0] for a in atomTypes])
    hardness = np.array([params.loc[params[atomLabel] == a][hardnessLabel].to_numpy()[0] for a in atomTypes])
    diameters = np.array([params.loc[params[atomLabel] == a][diameterLabel].to_numpy()[0] for a in atomTypes])
    return electronegativity, hardness, diameters

# extract bond electronegativities and hardnesses
def extractBondParams (mol, bondParams, bondTypeFunc, typeLabel, bondElnegLabel, bondHardnessLabel):
    bondTypes = np.array([bondTypeFunc(bond, mol) for bond in mol.GetBonds()])
    bondElneg = np.array([bondParams.loc[bondParams[typeLabel] == b][bondElnegLabel].to_numpy()[0] for b in bondTypes])
    bondHardness = np.array([bondParams.loc[bondParams[typeLabel] == b][bondHardnessLabel].to_numpy()[0] for b in bondTypes])
    return bondElneg, bondHardness

def extractCharges(mol):
    
    # initialize all charges to 0
    for a in mol.GetAtoms():
        a.SetDoubleProp("charge", 0)
    
    # add charges of H atoms onto carbons which they are bound to
    for b in mol.GetBonds():
        b1 = b.GetBeginAtom()
        b2 = b.GetEndAtom()
        if b1.GetSymbol() == "H":
            _addChargeH(atom=b2, hAtom=b1)
        elif b2.GetSymbol() == "H":
            _addChargeH(atom=b1, hAtom=b2)
    
    # make plotting easier
    charges = []
    
    # add charges on non-H atoms
    for a in mol.GetAtoms():
        if a.GetSymbol() != "H":
            chargeSelf = a.GetDoubleProp("molFileAlias")
            chargeH = a.GetDoubleProp("charge")
            a.SetDoubleProp("charge", chargeSelf + chargeH)
            charges.append(chargeSelf + chargeH)
            
    return np.array(charges)