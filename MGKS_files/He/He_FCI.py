#! /home/timgould/anaconda3/bin/python3

import psi4
from Broadway.OOEDFT import ExcitationHelper
from Broadway.Helpers import *
from Broadway.PlanHandler import *
from psi4Engine.Engine import psi4Engine, psi4Density
from LibExactThermal import *

import numpy as np

import sys

psi4.set_output_file("__He.out")


np.set_printoptions(precision=4, suppress=True)

psi4.set_memory('4 gb')

Options = ReadCFG("He.cfg")

RBox = Options['RBox']
kbTAll = Options['kbT']
DFA = Options['DFA']
Basis = Options['Basis']


psi4.set_options({
    'basis': Basis,
    'reference': 'rhf',
    'opdm': True,
    'dft_spherical_points': 110,
    'dft_radial_points': 100,
    'dft_bs_radius_alpha': 1.2,
})

psi4.geometry('He\nsymmetry c1')


E, wfn = psi4.energy(DFA, return_wfn=True)

Engine = psi4Engine(wfn)
XHelp = ExcitationHelper(Engine)

VBox = -1/RBox**2 * ( Engine.Quad_ao[0] + Engine.Quad_ao[3] + Engine.Quad_ao[5] )
Engine.UpdateV(VBox)
XHelp = ExcitationHelper(Engine)


#################################################
# Evaluate the FCI reference
#################################################
NRoots = {(1,2):92, (2,1):60, (2,3):40, (3,2):130, (3,4):60, (4,1):5, (4,3):4}
FCIH = FCIHelper(Basis, VBox, NRoots, Kind='He', 
                 Force=True, # This should be True
                 Delta3='Auto', # Correction for third order (qz/fci - tz/cisd)
                 )
FCIH.FixSinglets()
FCIH.Report(Raw = False)

E1 = FCIH.Data[1][2]['E_Box'][0]
E2 = FCIH.Data[2][1]['E_Box'][0]
E3 = FCIH.Data[3][2]['E_Box'][0]
E4 = FCIH.Data[4][1]['E_Box'][0]

IP = E1 - E2
EA = E2 - E3

E_ST = FCIH.Data[2][3]['E_Box'][0]-FCIH.Data[2][1]['E_Box'][0]
E_SS = FCIH.Data[2][1]['E_Box'][1]-FCIH.Data[2][1]['E_Box'][0]


print("IP = %5.2f EA = %5.2f E_ST = %5.2f E_SS = %5.2f"\
      %(IP*eV, EA*eV, E_ST*eV, E_SS*eV))

print('='*72)
mu = -(IP+EA)/2
Stride = 10
for N in FCIH.Data:
    E0 = [0,E1,E2,E3,E4][N]
    for D in FCIH.Data[N]:
        E_N_D = FCIH.Data[N][D]['E_Box'] - mu*(N-2)
        X = (E_N_D[E_N_D<(E2+50/eV)] - E2)*eV

        if len(X)==0: continue
        
        if len(X)<=Stride:
            print("%d & %d &  "%(N,D) + " & ".join(["%5.2f"%(x) for x in X]) + '  \\\\')
        else:
            Y = X[:Stride]
            print("%d & %d &  "%(N,D) + " & ".join(["%5.2f"%(x) for x in Y]) + '  \\\\')
            for k in range(Stride,len(X),Stride):
                Y = X[k:(k+Stride)]
                print("  &   &  " + " & ".join(["%5.2f"%(x) for x in Y]) + '  \\\\')


#######################################################################
# Finally compute some densities
#######################################################################


# Get a grid
psi4.core.clean()
psi4.geometry('0 1\nHe\nsymmetry c1')
psi4.set_options({'reference':'rhf',})
E, wfn = psi4.energy('pbe', return_wfn=True)

p4D = psi4Density(wfn)
_, xyz, w = p4D.Density(Engine.Da, return_all=True)

# Find the spherical abscissae and weights
lr_xyz = np.round(np.log(np.sum(xyz**2,axis=1))/1e-6)*1e-6
lr_unique = np.sort(np.unique(lr_xyz))
r = np.exp(lr_unique/2)
wr = 0.*r

k0 = len(lr_unique)//2
kk = lr_xyz==lr_unique[k0]
wphi = w[kk]/np.sum(w[kk])
phi = xyz[kk,:]/np.exp(lr_xyz[kk]/2.)[:,None]

for k, lr in enumerate(lr_unique):
    kk = lr_xyz==lr
    wr[k] = np.sum(w[kk])

def n_grid(D):
    n_xyz = p4D.Density(D, eta_n=1e-20)
    n_r = 0.*lr_unique
    for k, lr in enumerate(lr_unique):
        kk = lr_xyz==lr
        wphi = w[kk]/np.sum(w[kk])
        n_r[k] = np.dot(n_xyz[kk], wphi)

    return n_r

n_Exact = {}
for kbT in Options['kbT']:
    W_All, E_T, D_T = FCIH.Solve(kbT)
    n_Exact[kbT] = n_grid(D_T)

np.savez('Data/He_Density_%s.npz'%(Basis), n_Exact = n_Exact, r = r, wr = wr,
         xyz = xyz, w = w, wphi = wphi, lr_xyz = lr_xyz, lr_unique = lr_unique)


