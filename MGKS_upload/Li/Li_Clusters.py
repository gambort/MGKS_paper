#! /home/timgould/anaconda3/bin/python3

import psi4
from psi4Engine.Engine import psi4Engine

from LibEGKS import *

import numpy as np
import scipy.linalg as la



if __name__ == "__main__":
    print("""
Run as:
Li_Clusters.py [N] [Basis]

where [N] defaults to 8 and [Basis] defaults to def2-smsvp
""")
    
    import sys
    
    NLi = 8
    Basis = 'def2-smsvp'

    if len(sys.argv)>1:
        NLi = int(sys.argv[1])
    if len(sys.argv)>2:
        Basis = sys.argv[2].lower()


    def RSDFA(alpha, omega):
        return {'name': 'PBE_%.2f_%.2f'%(alpha,omega),
                'x_hf': {'alpha': alpha, 'beta': 1-alpha, 'omega': omega},
                'x_functionals': {'GGA_X_HJS_PBE': {'alpha': 1-alpha, 'omega': omega}},
                'c_functionals': {'GGA_C_PBE': {'alpha': 1.0}}
                }
    

    if NLi==0:
        DFAList = ('PBE', 'PBE0', 'HSE06', )
    else:
        alpha = 0.375
        if Basis=='def2-smsvp':
            omega = { 2: 0.216, 8: 0.131, 14: 0.089 }[NLi]
        else:
            omega = { 2: 0.180, 8: 0.101, 14: 0.063 }[NLi]

        OptDFA = RSDFA(alpha, omega)
        DFAList = ('PBE', 'PBE0', 'HSE06', 'OT-RSH')


    kbTAll = np.logspace(-4,-1,22)

    for DFA in DFAList:
        print('='*72)
        print(DFA)
        LiH = LiHelper(NLi, DFA=DFA, 
                       DFADict = OptDFA if DFA=='OT-RSH' else None,
                       Basis=Basis, NCache=5)
        for kbT in kbTAll:
            print('='*72)
            print('kbT = %8.6f Ha = %6.3f eV'%(kbT, kbT*eV))
            Converged = LiH.SolveEGKS(kbT, Mix_SCF = None, Report=3)
            if False and Converged:
                LiH.SolveInvert(kbT)

        psi4.core.clean()
