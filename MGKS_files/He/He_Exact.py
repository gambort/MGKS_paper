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

StopAfterTest = False
if len(sys.argv)>1:
    if sys.argv[1].lower()[:4]=='test':
        StopAfterTest = True

np.set_printoptions(precision=4, suppress=True)

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
    'dft_radial_points': 50,
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
NRoots = {(1,2):92, (2,1):60, (2,3):40, (3,2):130, (3,4):60, (4,1):3, (4,3):3}
FCIH = FCIHelper(Basis, VBox, NRoots, Kind='He', Force=False, 
                Delta3='Auto', # Correction for third order (qz/fci - tz/cisd)
)
FCIH.Report(Raw = False)


IP = FCIH.Data[1][2]['E_Box'][0]-FCIH.Data[2][1]['E_Box'][0]
EA = FCIH.Data[2][1]['E_Box'][0]-FCIH.Data[3][2]['E_Box'][0]

E_ST = FCIH.Data[2][3]['E_Box'][0]-FCIH.Data[2][1]['E_Box'][0]
E_SS = FCIH.Data[2][1]['E_Box'][1]-FCIH.Data[2][1]['E_Box'][0]

print('='*72)
print("Error analysis for He - small numbers (<50%) in right three columns indicate reliable results")
FCIH.EstimateErrors(0, 10)
print('='*72)

print("IP = %5.2f EA = %5.2f E_ST = %5.2f E_SS = %5.2f"\
      %(IP*eV, EA*eV, E_ST*eV, E_SS*eV))

if StopAfterTest: quit()

Ens = { N:{D: FCIH.Data[N][D]['E_Box'] for D in FCIH.Data[N] } for N in FCIH.Data }

RefData = {
    'E_ST': E_ST, 'E_SS': E_SS, 'IP': IP, 'EA': EA,
    'Ens': Ens,
}

#################################################
# Evaluate the KS reference
#################################################
KSTH = KSThermalHelper(Engine, XHelp)
ShowCut = 5e-6

# Work out when to include densities
NDensInclude = 3
def IncludeDensity(k):
    return ((k%NDensInclude)==0) or (k==(len(kbTAll)-1))

# Initialise the density storage
p4D = psi4Density(wfn)
_, xyz, w = p4D.Density(Engine.Da, return_all=True)
rhoData = {
    'xyz': xyz, 'w': w, 'Densities': {},
}

# Creat the suffix
Suff = ''
if not(DFA.lower()=='svwn'): Suff += '_%s'%(DFA.lower().replace('-','_'))

CacheFile = "Data/He_Exact_Thermal_%s%s.npz"%(Basis, Suff)

Data = {}
Data_EGKS = {}
ExtraData = {}
ExtraData_EGKS = {}
for k_T, kbT in enumerate(kbTAll):
    print('='*72)
    print('kbT = %10.5f Ha %10.2f eV'%(kbT, kbT*eV))

    W_Dict, E_T, D_T = FCIH.Solve(kbT)
    for N in W_Dict: 
        for D in W_Dict[N]:
            W = W_Dict[N][D]
            print("W(%d,%d) = %10.6f total %10.6f last"%(N, D, np.sum(W), W[-1]))

    if IncludeDensity(k_T):
        EnProps, MiscProps, DensProps = KSTH.Solve(kbT, FCIHelper = FCIH,
                                                   Return_Densities=True, p4D=p4D)
        rhoData['Densities'][kbT] = DensProps
    else:
        EnProps, MiscProps = KSTH.Solve(kbT, FCIHelper = FCIH,)
    EnProps_EGKS, MiscProps_EGKS = KSTH.Solve(kbT, FCIHelper = FCIH, EGKS=True,)

    eps_KS  = MiscProps['epsilon']
    eps_GKS = MiscProps_EGKS['epsilon']

    print("Gap KS  = %6.3f Gap GKS = %6.3f"%(eps_KS[1]-eps_KS[0], eps_GKS[1]-eps_GKS[0]),
          flush=True)

    Data[kbT] = EnProps
    Data_EGKS[kbT] = EnProps_EGKS
    ExtraData[kbT] = MiscProps
    ExtraData_EGKS[kbT] = MiscProps_EGKS
    
    np.savez(CacheFile, 
             Data = Data, Data_EGKS = Data_EGKS,
             ExtraData = ExtraData, ExtraData_EGKS = ExtraData_EGKS,
             RefData = RefData,
             rhoData = rhoData,)
