from qcalc.workers import createWorker, createParameters
from qcalc.parameter import Parameter
from scipy.optimize import dual_annealing, basinhopping, shgo
import numpy as np
import pandas as pd


def calculateWeights(atomTypes, params):
    counts = np.array([atomTypes.count(a) for a in params.df["atom"]])
    # avoid division by zero
    counts = np.where(counts == 0, np.inf, counts)
    wcol = 1./np.sqrt(counts)
    # debugging
    params.df["weights"] = wcol
    weights = {k:v for k,v in list(zip(params.df["atom"], wcol))}
    return weights

# weights is a dict {atomType: weight}
# targetCharges is array of arrays
def costFunction(arr, workers, weights, targetCharges, constr):
    
    # add constrained values
    paramsArr = np.insert(arr, constr[0], constr[1])

    # update parameters
    for worker in workers:
        worker.setParams(paramsArr)

    totalCost = 0

    # compute charges
    for worker, target in list(zip(workers, targetCharges)):
        charges = worker.compute()
        w = np.array([weights[a] for a in worker.atomTypes])
        totalCost += np.sum(w*(charges - target)**2)

    return totalCost

def createParamsArr(params, bondParams):
    paramsArr, constr = params.toArray()
    if bondParams is not None:
        bondParamsArr, bondConstr = bondParams.toArray()
        # update constraint indices
        bondIndices = [bi + len(paramsArr) for bi in bondConstr[0]]
        constrIndices = constr[0] + bondIndices
        constrValues = constr[1] + bondConstr[1] 
        constr = (constrIndices, constrValues)
        # concatenate params
        paramsArr = np.concatenate((paramsArr, bondParamsArr), axis=0)
    return paramsArr, constr

def updateParams(params, bondParams, paramsArr):
    lenParams = params.M * params.N - len(params.constraints)
    params.update(paramsArr[:lenParams])
    if (bondParams is not None) and len(paramsArr[lenParams:]) > 0:
        bondParams.update(paramsArr[lenParams:])

def createWorkers(mols, params, method, atomTypeFunc, bondParams, **kwargs):
    workers = []
    for mol in mols:
        worker = createWorker(mol, params, method, atomTypeFunc, bondParams=bondParams, **kwargs)
        workers.append(worker)
    return workers 


# mols - list
# targetCharges - list of lists
def optimizeParameters(mols, paramTable, method, atomTypeFunc, targetCharges, **kwargs):

    # prepare parameters
    params, bondParams = createParameters(paramTable, method, **kwargs)
    paramsArr, constr = createParamsArr(params, bondParams)

    # prepare workers
    workers = createWorkers(mols, params, method, atomTypeFunc, bondParams, **kwargs)

    # calculate weights based on counts of atom types in the training set
    atomTypes = [a for w in workers for a in w.atomTypes]
    weights = calculateWeights(atomTypes, params)

    # parameter ranges - [0, 10*initial params]
    ranges = [(1E-3,10*p) for p in paramsArr]

    # run optimization
    #maxiter = kwargs.get("maxiter", 1000)
    #opt = dual_annealing(costFunction, ranges, args=(workers, weights, targetCharges, constr), maxiter=maxiter)

    #return opt

    # evaluate cost function - to be done in optimize
    #cost = costFunction(paramsArr, workers, weights, targetCharges)
    return workers, paramsArr, weights

if __name__ == "__main__":
    print("Calling optimize")
