import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import time
from math import sqrt

from DataProc import ReadFile, ReadBulkFile

#mempool = cp.get_default_memory_pool()
#with cp.cuda.Device(0):
#    mempool.set_limit(size = 1 * (1024**3) )  # 1 GiB

################################################################################################
#                                      SYSTEM SETUP                                            #
################################################################################################
#----Output Settings----
OutputType = 'File'         #'Window' or 'File'
SaveFile = 'PGF'            #'PGF' or 'PNG'
SaveName = 'BCC-Fe-011-300L.pgf'     # SaveFile Name with correct extension
PlotTitle = 'BCC Fe - (011) - 300Layers'

#----General Settings----
Type = 'Moment'       #'Spin' or 'Moment'
S = 3/2             #Spin / Moment
g = 2               #G-factor
H = 300              #Number of Layers

#----Exchange Parameters----
surf_data = "tmp_jrs.dat"           #Jrs and Lattice file
bulk_data = "tmp_jrs.dat"     #Jrs and Lattice file

Surf_SlabNr = np.array([21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6])    #Slab_i of S, S-1, ...
Bulk_SlabNr = 6                                     #Slab_i of Bulk

#----K-Path----
KPath = np.array([[0.6693   , 0, 1.3386],
                  [-0.94653 , 0, -0.47327  ],
                  [0        , 0, 0  ]])                        #K-Points
KPath[:,2] = 0.43 * KPath[:,2]
KPath[:,0] = 0.87 * KPath[:,0]
KPath = np.pi*KPath*(1/2.710000)
KPath = np.fliplr(KPath)
PointNames = np.array(['M', '$\Gamma$', 'X'])   #K-Point Names

Nq = 50         #Samples between Points

################################################################################################
################################################################################################
#----Output Setup----
if OutputType == 'File' and SaveFile == 'PGF':
    import matplotlib as mpl
    mpl.use('pgf')
    

#----Dispersion Path----
N_KPoints = np.size(KPath[0,:])
Path = np.zeros((2,(N_KPoints -1) * Nq))
for i in np.arange(N_KPoints-1):
    kx = np.linspace(KPath[0,i],KPath[0,i+1],Nq)
    ky = np.linspace(KPath[1,i],KPath[1,i+1],Nq)
    Path[0,(i*Nq):((i+1)*Nq)] = kx
    Path[1,(i*Nq):((i+1)*Nq)] = ky


#----Read Data----
InLayer, InterLayer = ReadFile(surf_data)
InLayerB, InterLayerB = ReadFile(bulk_data)


#----Setup Surf_SlabNr Array----
N_SurfData = np.size(Surf_SlabNr)
Reps = H - N_SurfData
if Reps > 0:
    Surf_SlabNr = np.pad(Surf_SlabNr,
                         (0, Reps), 'constant',
                         constant_values=(0, Bulk_SlabNr))


#----J0(l,m)----
def J0(l,m):
    if l < N_SurfData and m < N_SurfData:
        if l == m:
            Slab = InLayer[np.where(InLayer[:,0] == Surf_SlabNr[l])]
            return np.sum(Slab[:,4])
        else:
            Slab = InterLayer[np.where(InterLayer[:,0] == Surf_SlabNr[l])]
            Slab = Slab[np.where(Slab[:,1] == Surf_SlabNr[m])]
            return np.sum(Slab[:,6])

    else:
        if l == m:
            Slab = InLayerB[np.where(InLayerB[:,0] == Bulk_SlabNr)]
            return np.sum(Slab[:,4])
        else:
            Sign = m-l
            Slab = InterLayerB[np.where(InterLayerB[:,0] == Bulk_SlabNr)]
            if np.abs(Sign) <= 5:
                Slab = Slab[np.where(Slab[:,1] == Bulk_SlabNr+Sign)]
                return np.sum(Slab[:,6])
            else:
                return 0


#----Jq(l,m,qx,qy)----
def Jq(l,m,qx,qy):
    Jq = 0.
    if l < N_SurfData and m < N_SurfData:
        if l == m:
            Slab = InLayer[np.where(InLayer[:,0] == Surf_SlabNr[l])]
            N_Sites = np.size(Slab[:,0])
            for i in np.arange(N_Sites):
                Rx = Slab[i,1]
                Ry = Slab[i,2]
                Exp = (Rx * qx) + (Ry * qy)
                Jq += Slab[i,4] * np.exp(1j * Exp)
            return np.real(Jq)
        else:
            Slab = InterLayer[np.where(InterLayer[:,0] == Surf_SlabNr[l])]
            Slab = Slab[np.where(Slab[:,1] == Surf_SlabNr[m])]
            N_Sites = np.size(Slab[:,0])
            for i in np.arange(N_Sites):
                Rx = Slab[i,2]
                Ry = Slab[i,3]
                Exp = (Rx * qx) + (Ry * qy)
                Jq = Jq + Slab[i,6] * np.exp(1j * Exp)
            return np.real(Jq)
    
    else:
        if l == m:
            Slab = InLayerB[np.where(InLayerB[:,0] == Bulk_SlabNr)]
            N_Sites = np.size(Slab[:,0])
            for i in np.arange(N_Sites):
                Rx = Slab[i,1]
                Ry = Slab[i,2]
                Exp = (Rx * qx) + (Ry * qy)
                Jq = Jq + Slab[i,4] * np.exp(1j * Exp)
            return np.real(Jq)
        else:
            Sign = m-l
            Slab = InterLayerB[np.where(InterLayerB[:,0] == Bulk_SlabNr)]
            if np.abs(Sign) <= 5:
                Slab = Slab[np.where(Slab[:,1] == Bulk_SlabNr+Sign)]
                N_Sites = np.size(Slab[:,0])
                for i in np.arange(N_Sites):
                    Rx = Slab[i,2]
                    Ry = Slab[i,3]
                    Exp = (Rx * qx) + (Ry * qy)
                    Jq = Jq + Slab[i,6] * np.exp(1j * Exp)
                return np.real(Jq)
            else:
                return 0
    
        
#----Torque Matrix NEW----
def Tqmat(qx,qy):    
    Tq = cp.zeros((H,H))
    for i in np.arange(H):
        Inter0 = 0.
        for k in np.arange(H):
            Inter0 = Inter0 + J0(i,k)
        J_q = Jq(i,i,qx,qy)
        Tq[i][i] = Inter0 - J_q
    for i in np.arange(H):
        for k in np.arange(H):
            if i!=k:
                J_q = Jq(i,k,qx,qy)
                Tq[i][k] = - J_q
    if Type == 'Spin':
        return g*Tq*S
    elif Type == 'Moment':
        return g*Tq/S


#----Eigenvalues(parallel)---
def Spec():
    Num = (N_KPoints -1) * Nq
    Tq_vec = cp.empty((Num,H,H))
    for i in np.arange(Num):
        Tq_vec[i,:,:] = Tqmat(Path[0,i],Path[1,i])
    Spec = cp.linalg.eigvalsh(Tq_vec, UPLO='L')
    return cp.asnumpy(cp.transpose(Spec))
    
    
print('Solving for Eigenvalues...')
t0 = time.time()
Spectr = Spec()
t1 = time.time()
print('Time: ',t1-t0)


#----Display Layers----
def LayerSpec(List,Spec,style,Names):
    List = np.asarray(List)
    plot_x = np.arange((Spec.shape)[1])
    for i in List:
        plt.plot(plot_x,Spec[i,:],'{}'.format(style))  
    for i in Nq*np.arange(np.size(KPath[0,:])):
        plt.axvline(x=i-1,color='black',linewidth=0.5)
    plt.xticks([])
    plt.xticks((Nq)*np.arange(np.size(KPath[0,:]))-1,Names)
    Names = np.append(Names,Names[0])
    plt.xticks( list(plt.xticks()[0]) +list([0]),Names)
    plt.title(PlotTitle)
    plt.ylabel('Energy [meV]')
    plt.xlim([0,np.amax(plot_x)])
    plt.ylim(bottom=np.amin(Spec[np.amin(List),:]))

    
LayerSpec(np.arange(0,H),Spectr,'m-',PointNames)
#LayerSpec(np.array([0]),Spectr,'-',PointNames)


plt.tight_layout()
#----Output----
if OutputType == 'Window':
    plt.show()
elif OutputType == 'File':
   plt.savefig(SaveName, transparent=True) 


