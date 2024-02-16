import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import time

animate = True
bw_cmap = colors.ListedColormap(['black', 'white'])

Te = 3.0
L = 6                       
N_spins = L**2                   
J = 1          
Spins_s = 2*np.random.randint(0,2,N_spins) - 1

NN_s = np.zeros([N_spins,4],int)
for i1 in range(N_spins):
    NN_s[i1,0] = i1+1
    if i1%L==(L-1):
        NN_s[i1,0] = i1+1-L
    NN_s[i1,1] = i1+L
    if i1 >= (N_spins-L):
        NN_s[i1,1] = i1+L-N_spins
    NN_s[i1,2] = i1-1
    if i1%L == 0:
        NN_s[i1,2] = i1+L-1
    NN_s[i1,3] = i1-L
    if i1 <= (L-1):
        NN_s[i1,3] = N_spins-L+i1

def Energy(J1,N1,NN_1,vec_1):
    E_aux = 0.0
    for wx1 in range(N1):
        E_aux += -J1*vec_1[wx1]*(vec_1[NN_1[wx1,0]]+vec_1[NN_1[wx1,1]])
    return E_aux

def MC_step(J1,T1,N1,NN_1,vec_1):
    for wx1 in range(N1):
        site = np.random.randint(0,N1)
        delta_E = 2*J1*vec_1[site]*(vec_1[NN_1[wx1,0]]+vec_1[NN_1[wx1,1]]+vec_1[NN_1[wx1,2]]+vec_1[NN_1[wx1,3]])
        if (delta_E <= 0) or (np.random.random() < np.exp(-delta_E/T1)):
            vec_1[site] = -vec_1[site]
    return vec_1


n_meas = int(80)
E_s = np.zeros(n_meas,float)
M_s = np.zeros(n_meas,float)
for i1 in range(n_meas):
    MC_step(J,Te,N_spins,NN_s,Spins_s)
    E_s[i1] =  Energy(J,N_spins,NN_s,Spins_s)
    M_s[i1] = abs(np.sum(Spins_s))/L**2
    if animate:
        plt.clf()
        plt.imshow(Spins_s.reshape((L,L)), cmap=bw_cmap, norm=colors.BoundaryNorm([-1,0,1], bw_cmap.N), interpolation='nearest' )
        plt.xticks([])
        plt.yticks([])
        plt.title('%d x %d Ising model, T = %.3f' %(L,L,Te))
        plt.pause(0.05)
plt.show()
plt.plot(M_s,marker='*',linestyle='None')
plt.xlabel("MC_step")
plt.ylabel("$|M|$")
plt.show()