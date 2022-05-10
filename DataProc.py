import numpy as np

def ReadFile(FileName):
    data = np.genfromtxt(FileName)
    #----Read Data----
    data = np.delete(data, 1, 1)        #Remove 'atom'
    data = np.delete(data, 2, 1)        #Remove 'atom'
    data = np.delete(data, 2, 1)        #Remove 'shell'
    data = np.delete(data, 2, 1)        #Remove 'is'
    data = np.delete(data, 2, 1)        #Remove 'irs'
    data = np.delete(data, 6, 1)        #Remove 'mRy'
    data = np.delete(data, 7, 1)        #Remove 'pj'
    data = np.delete(data, 7, 1)        #Remove 'pj'
    data = np.delete(data, 7, 1)        #Remove 'pj'
    data = np.delete(data, 7, 1)        #Remove 'rel.cell Coord'
    data = np.delete(data, 7, 1)        #Remove 'rel.cell Coord'
    data = np.delete(data, 7, 1)        #Remove 'rel.cell Coord'
    #data Columns are: Slab_i, Slab_j, X_j, Y_j, Z_j, R, J(meV)


    #----In-Layer Array----
    InLayer = np.zeros((np.shape(data)[0],5))
    for i in np.arange(np.shape(data)[0]):
        if data[i,0] == data[i,1]:
            InLayer[i,0] = data[i,0]
            InLayer[i,1] = data[i,2]
            InLayer[i,2] = data[i,3]
            InLayer[i,3] = data[i,5]
            InLayer[i,4] = data[i,6]
    InLayer = InLayer[~np.all(InLayer == 0, axis=1)]
    #InLayer Columns are: Slab_i, X_j, Y_j, R, J(meV)

    InLayer = InLayer[np.abs(InLayer[:,4]).argsort(),:]
    InLayer = InLayer[InLayer[:,3].argsort(kind='mergesort'),:]
    InLayer = InLayer[InLayer[:,0].argsort(kind='mergesort'),:]
    #InLayer Rows sorted by abs(J), Radius, Slab_i


    #----Inter-Layer Array----
    InterLayer = np.zeros((np.shape(data)[0],7))
    for i in np.arange(np.shape(data)[0]):
        if data[i,0] != data[i,1]:
            InterLayer[i,:] = data[i,:]
    InterLayer = InterLayer[~np.all(InterLayer == 0, axis=1)]
    #InterLayer Columns are: Slab_i, Slab_j, X_j, Y_j, Z_j, R, J(meV)

    InterLayer = InterLayer[np.abs(InterLayer[:,6]).argsort(),:]
    InterLayer = InterLayer[InterLayer[:,5].argsort(kind='mergesort'),:]
    InterLayer = InterLayer[InterLayer[:,1].argsort(kind='mergesort'),:]
    InterLayer = InterLayer[InterLayer[:,0].argsort(kind='mergesort'),:]
    #InterLayer Rows sorted by abs(J), Radius, Slab_j, Slab_i
    return InLayer, InterLayer



def ReadBulkFile(FileName):
    data = np.genfromtxt(FileName)
    #----Read Data----
    data = np.delete(data, 1, 1)        #Remove 'atom'
    data = np.delete(data, 2, 1)        #Remove 'atom'
    data = np.delete(data, 2, 1)        #Remove 'shell'
    data = np.delete(data, 2, 1)        #Remove 'is'
    data = np.delete(data, 2, 1)        #Remove 'irs'
    data = np.delete(data, 6, 1)        #Remove 'mRy'
    data = np.delete(data, 7, 1)        #Remove 'pj'
    data = np.delete(data, 7, 1)        #Remove 'pj'
    data = np.delete(data, 7, 1)        #Remove 'pj'
    data = np.delete(data, 7, 1)        #Remove 'rel.cell Coord'
    data = np.delete(data, 7, 1)        #Remove 'rel.cell Coord'
    data = np.delete(data, 7, 1)        #Remove 'rel.cell Coord'
    #data Columns are: Slab_i, Slab_j, X_j, Y_j, Z_j, R, J(meV)


    #----In-Layer Array----
    InLayer = np.zeros((np.shape(data)[0],5))
    for i in np.arange(np.shape(data)[0]):
        if data[i,4] == 0:
            InLayer[i,0] = data[i,0]
            InLayer[i,1] = data[i,2]
            InLayer[i,2] = data[i,3]
            InLayer[i,3] = data[i,5]
            InLayer[i,4] = data[i,6]
    InLayer = InLayer[~np.all(InLayer == 0, axis=1)]
    #InLayer Columns are: Slab_i, X_j, Y_j, R, J(meV)

    InLayer = InLayer[np.abs(InLayer[:,4]).argsort(),:]
    InLayer = InLayer[InLayer[:,3].argsort(kind='mergesort'),:]
    InLayer = InLayer[InLayer[:,0].argsort(kind='mergesort'),:]
    #InLayer Rows sorted by abs(J), Radius, Slab_i


    #----Inter-Layer Array----
    InterLayer = np.zeros((np.shape(data)[0],7))
    for i in np.arange(np.shape(data)[0]):
        if data[i,4] != 0:
            InterLayer[i,:] = data[i,:]
    InterLayer = InterLayer[~np.all(InterLayer == 0, axis=1)]
    #InterLayer Columns are: Slab_i, Slab_j, X_j, Y_j, Z_j, R, J(meV)

    InterLayer = InterLayer[np.abs(InterLayer[:,6]).argsort(),:]
    InterLayer = InterLayer[InterLayer[:,5].argsort(kind='mergesort'),:]
    InterLayer = InterLayer[InterLayer[:,4].argsort(kind='mergesort'),:]
    InterLayer = InterLayer[InterLayer[:,0].argsort(kind='mergesort'),:]
    #InterLayer Rows sorted by abs(J), Radius, Z_j, Slab_i
    return InLayer, InterLayer

