__version__ = 1.1
__author__ = "Tim Gould"

from Broadway.EDFT import CoreExcitationHelper
from Broadway.LDAFits import *

import numpy as np
import scipy.linalg as la

eV = 27.211

zero_round = 1e-5

class pExcitationHelper(CoreExcitationHelper):
    # Generic solver routine - currently can only call SolverpEDFT
    def Solver(self, Plan, kFrom=None, kTo=None,
               Dipole = False, # Return the dipole as well
               **kwargs):
        return self.SolverpEDFT(Plan, **kwargs)
        
    # Solve the pEDFT equations iteratively
    def SolverpEDFT(self, Plan,
                    Mix = 0.7, MaxIter = 50,
                    EThresh = 1e-6, epsThresh = 4e-5,
                    **kwargs):
        
        Plan_gs = self.SolveGS(PlanOnly=True)
        E_GS, FC_GS = self.GetEnergy(Plan_gs)
        H_GS = (self.CE.T).dot(FC_GS)

        self.epsilonE, UE = la.eigh(H_GS)
        self.CE = self.CE.dot(UE)

        if len(Plan['kTo'])>1:
            print("pEDFT does not work for multi-excitations - returning None")
            return None
        
        kTo = Plan['kTo'][0]
        dkTo = kTo - self.kl
        
        EOld = 0.
        epsOld = 0.
        for step in range(MaxIter):
            E, FockList, FockMap = self.GetEnergyFocks(Plan)
            if not(kTo in FockList):
                break
            FC = FockList[kTo]

            Hvir = FC[self.kl:,self.kl:] # Virtual space only
            epsvir, Uvir = la.eigh(Hvir)

            Avir = np.real(la.logm(Uvir))
            Avir = (Avir - Avir.T)/2

            Uvir = la.expm(Mix*Avir)

            eps = (Uvir[:,dkTo]).dot(Hvir).dot(Uvir[:,dkTo])

            if self.Report>=3:
                print("%3d %11.5f %10.2e %11.5f %10.2e"\
                      %(step, eps, eps-epsOld, E, E-EOld))
                

            Converged = np.abs(E-EOld)<EThresh*Mix and np.abs(eps-epsOld)<epsThresh*Mix

            if Converged:
                break
           
            self.CE[:,self.kl:] = self.CE[:,self.kl:].dot(Uvir)
            self.epsilonE[self.kl:] = np.einsum('pk,qk,pq->k',
                                                Uvir, Uvir, Hvir)
            
            EOld = E
            epsOld = eps

        if not(Converged):
            print("Warning! Calculations did not converge so returning None")
            print("Errors: eps -> %10.2e, E -> %10.2e [Ha/Mix]"\
                  %( (eps-epsOld)/Mix, (E-EOld)/Mix )
            )
            return None

        return E


class ExcitationHelper(pExcitationHelper):
    1


if __name__ == "__main__":
    1
