import numpy as np

def getConnectivity(atoms, bonds):
    # create connectivity matrix
    connectivity = np.zeros((len(atoms), len(atoms)))

    # add bonds
    for b in bonds:
        connectivity[b[0],b[1]] = 1
        connectivity[b[1],b[0]] = 1

    # higher-order connectivity 
    while (len(np.where(connectivity == 0)[0]) > len(atoms)):
        tmpConnectivity = connectivity.copy()
        #loop over matrix
        for i in range(0,len(atoms)):
            neighborsI = [idx for idx,k in enumerate(connectivity[i]) if (k == 1)]
            #print("---" + str(neighborsI) + "---")
            for ni in neighborsI:
                neighborsJ = [idx for idx,k in enumerate(connectivity[ni]) if (k)]
                #print(neighborsJ)
                for nj in neighborsJ:
                    if connectivity[i,nj] == 0 and i != nj:
                        order = connectivity[i,ni] + connectivity[ni,nj]
                        tmpConnectivity[i,nj] = order
                        tmpConnectivity[nj,i] = order

        connectivity = tmpConnectivity.copy()  
        
    return connectivity