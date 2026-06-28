#! /home/timgould/anaconda3/bin/python3

import psi4
from Broadway.OOEDFT import ExcitationHelper
from Broadway.Helpers import *
from Broadway.PlanHandler import *
from psi4Engine.Engine import psi4Engine, psi4Density, GX24, TextDFA

import numpy as np

import sys

psi4.set_output_file("__H2.out")

from LibExactThermal import *

np.set_printoptions(precision=4, suppress=True)


Options = ReadCFG("He.cfg")

RBox = Options['RBox']
kbTAll = Options['kbT']
DFA = Options['DFA']
Basis = Options['Basis']
psi4.set_options({
    'basis': Basis,
    'reference': 'rhf',
    'df_basis_mp2': 'hepotbasis',
    'opdm': True,
    'dft_spherical_points': 110,
    'dft_radial_points': 50,
    'dft_bs_radius_alpha': 1.2,
})

psi4.geometry('He\nsymmetry c1')


if DFA.upper()=='GX24':
    E, wfn = psi4.energy('scf', dft_functional = GX24(), return_wfn=True)
elif DFA.upper()=='OT':
    E, wfn = psi4.energy('scf', dft_functional = TextDFA('wpbe_37.5_0.664'),
                         return_wfn=True)
else:
    E, wfn = psi4.energy(DFA, return_wfn=True)

Engine = psi4Engine(wfn)
XHelp = ExcitationHelper(Engine)

VBox = -1/RBox**2 * ( Engine.Quad_ao[0] + Engine.Quad_ao[3] + Engine.Quad_ao[5] )
Engine.UpdateV(VBox)
XHelp = ExcitationHelper(Engine)

#################################################
# Evaluate the EGKS and KS reference
#################################################
HFTH = HFThermalHelper(Engine, XHelp)
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

CacheFile = "Data/He_HF_Thermal_%s%s.npz"%(Basis, Suff)

try:
    X = np.load(CacheFile, allow_pickle=True)
    Data = X['Data'][()]
except:
    Data = {}

for k_T, kbT in enumerate(kbTAll):
    print('='*72)
    print('kbT = %10.5f Ha %10.2f eV'%(kbT, kbT*eV))

    if kbT in Data: continue

    HFTH.SolveHF(kbT)
    HFTH.SolveEXX(kbT)

    if np.abs(HFTH.HFProps['FE'] - HFTH.KSProps['FE'])>0.1:
        print("Skipping due to large EXX/HF difference")
        print(HFTH.Last_Q)
        continue

    print('-'*36)

    EnProps = {}
    print('EGKS HF')
    EnProps['HF' ] = HFTH.ComputeEns(HFTH.HFProps, Report=True)
    print('EKS EXX')
    EnProps['EXX'] = HFTH.ComputeEns(HFTH.KSProps, Report=True)
    
    Data[kbT] = EnProps

    np.savez(CacheFile, Data = Data )
