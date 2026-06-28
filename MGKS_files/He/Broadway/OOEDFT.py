__version__ = 1.6
__author__ = "Tim Gould"

from Broadway.EDFT import CoreExcitationHelper
from Broadway.LDAFits import *

import numpy as np
import scipy.linalg as la
import numpy.random as ra

eV = 27.211

zero_round = 1e-5

# Generate U or 
def OO_CondenseFList(FMap, FList, Mode='U', f=None):
    N = 0
    for k in FMap:
        N = max(np.max(FMap[k])+1, N)
    U = np.zeros((N,N))
    for k1 in FList:
        for k2 in FList:
            if Mode=='U' and k2>=k1: continue
            elif Mode=='F' and k2>k1: continue

            for a in FMap[k1]:
                if Mode=='U':
                    Delta = f[k2] * FList[k2][a,FMap[k2]] - f[k1] * FList[k1][a,FMap[k2]]
                    U[a,FMap[k2]] = -Delta
                    U[FMap[k2],a] =  Delta
                else:
                    Delta = 0.5*(FList[k1][a,FMap[k2]] + FList[k2][a,FMap[k2]])

                    U[a,FMap[k2]] = Delta
                    U[FMap[k2],a] = Delta
    return U

# Estimate the eigenvalues
def OO_GetEpsilon(FList, FMap):
    eps = np.zeros((FList[-1].shape[0],))
    for k in FMap:
        kk = FMap[k]
        eps[kk] = np.diag(FList[k])[kk]
    return eps


# Estimate of the Hessian
def OO_EstimateIH(f, epsilon, eta=0.1, Abs = True):
    H0 = -2.*(f[:,None] - f[None,:])*(epsilon[:,None] - epsilon[None,:])
    #return np.sign(H0)/np.maximum(np.abs(H0), eta)
    if Abs:
        return np.abs(H0)/(H0**2 + eta**2)
    else:
        return H0/(H0**2 + eta**2)


class OOExcitationHelper(CoreExcitationHelper):
    # Set the internal properties
    def SetProps(self, 
                 MaxIter = 200, ShowIter = 20,
                 DECut = 1e-7, DepsCut = 1e-6,
                 Fail = True,
                 **kwargs):
        self.Props['MaxIter'] = MaxIter
        self.Props['ShowIter'] = ShowIter
        self.Props['DECut']   = DECut
        self.Props['DepsCut'] = DepsCut
        self.Props['Fail']    = Fail
        
    # Generic solver routine - currently can only call SolverpEDFT
    def Solver(self, Plan, kFrom=None, kTo=None,
               Dipole = False, # Return the dipole as well
               **kwargs):
        return self.SolverOOEDFT(Plan, **kwargs)
        
    def SolverOOEDFT(self, Plan, **kwargs):
        return self.SolverOOEDFT_LS(Plan, **kwargs)
    
    # Solve the pEDFT equations iteratively using a
    # line search after NLineStep iterations
    def SolverOOEDFT_LS(self, Plan,
                        Dipole = False, # Return the dipole as well

                        Reset = True,

                        delta = 1.0, # Default delta guess
                        NLineWarm = 5, # Use line search at first
                        NLineStep = 50, # Start quadratic refinement at this step
                        Prefactor = True, # Use an approximate 2nd deriv prefactor

                        FreezeTo = None, # Freeze up to this orbital

                        IH_eta = 0.1, # Factor for approximate 2nd derivative
                        IH_abs = True, # Use absolute values

                        GJitter = 0.0, # Introduce a stochastic factor A -> A*(1+Gjitter*rand)
                        
                        MixUOld = 0.3, # Mix this much of the old U (negative is random mixing)
                        DE_Break = 1e-7, # Break when energy varies less than this
                        UM_Break = 5e-5, # Break when ||U|| is less than this
                        MaxIter = 500, # Maximum number of steps to take
                        **kwargs):
        if Reset:
            self.CE = 1.*self.C0
            self.epsilonE = 1.*self.epsilon0

        # Always rebuild the recipes before beginning
        Plan.GenerateFockRecipes(NOrb=self.CE.shape[1])

        E_Trial, FC_Trial = self.GetEnergy(Plan)
        E, FList, FMap = self.GetEnergyFocks(Plan)

        # Get the occupation factors in full size
        NOrb = FC_Trial.shape[1]
        f = np.zeros((NOrb,))
        f_ = Plan['1RDM'].np
        if len(f_)>len(f):
            f = f_[:len(f)]
        else:
            f[:len(f_)] = f_

        if Prefactor: # Approximate second deriative
            IH0 = OO_EstimateIH(f, OO_GetEpsilon(FList, FMap), eta = IH_eta, Abs = IH_abs)
        else:
            IH0 = 1.

        # Initialize last steps
        P = np.eye(NOrb) # Stores the total unitary transformation of CE
        EOld = E # Old energy
        AMatOld = None # Old AMat
        POld = None # Old P
        UOver = 1. # For orthogonality to last step
        delta_Old = delta # Last step size

        if self.Report>0:
            print("%4s %12s %7s %6s %6s %6s [ %23s ]"\
                  %('step', 'En [Ha]', 'DE [eV]', 'delta', '||U||', 'U_ang',
                    'Line search E [eV]'),
                  flush=True)


        # Mask out terms that shouldn't interact because of
        # symmetry
        SUnique = set(list(self.Sym_k)) # Unique symmetries
        if len(SUnique)>1:
            Mask = np.zeros((NOrb,NOrb))
            for S in SUnique:
                k = np.argwhere(self.Sym_k==S).reshape((-1,))
                Mask[(k[:,None], k)] = 1.
        else:
            Mask = np.ones((NOrb,NOrb))

        # Mask out frozen terms
        if not(FreezeTo is None):
            Mask[:FreezeTo, :] = 0.
            Mask[:, :FreezeTo] = 0.

        self.Converged = False
        for step in range(MaxIter):
            E, FList, FMap = self.GetEnergyFocks(Plan)

            AMat = IH0 * OO_CondenseFList(FMap, FList, f=f, Mode='U')
            AMat *= Mask # Zero out terms that should not couple (if any)

            if Prefactor: # Approximate second derivative
                IH0 = OO_EstimateIH(f, OO_GetEpsilon(FList, FMap), eta = IH_eta, Abs = IH_abs)

            if GJitter>0.:
                L = ra.randn(AMat.shape[0], AMat.shape[1])
                AMat += GJitter*AMat*(L-L.T)/2

            # Mix in the last step for faster convergence
            if not(AMatOld is None):
                AMat = (1-MixUOld)*AMat + MixUOld*AMatOld

                # Magnitude of U is Tr[AMat AMat]
                UM = np.sqrt(np.sum(AMat**2))

                # Get the dot product of old and new
                if (UM*UMOld)>1e-10:
                    UOver = np.vdot(AMat, AMatOld)/UM/UMOld
                else:
                    UOver = 1.0
            else:
                UOver = 1.0
                UM = np.sqrt(np.sum(AMat**2))

            if (step>=NLineStep) or (step<NLineWarm):
                # Make a qudaratic fit to determine best change
                deltas = [0, 1.5*delta, 3.0*delta]
                
                C1 = self.CE.dot(la.expm( deltas[1]*AMat))
                C2 = self.CE.dot(la.expm( deltas[2]*AMat))


                E1, _ = self.GetEnergy(Plan, C=C1)
                E2, _ = self.GetEnergy(Plan, C=C2)

                pE = np.polyfit(deltas, [E, E1, E2], 2)

                if pE[0]>0.:
                    delta_Opt = pE[1]/(-2.*pE[0])
                    delta_Opt = np.sign(delta_Opt) * min(deltas[2], np.abs(delta_Opt))
                else:
                    delta_Opt = deltas[2]*np.sign(E-E2)
            elif step==NLineWarm:
                E1 = E_Trial
                E2 = E_Trial
                delta_Opt = delta
                delta_Old = delta
            else:
                E1 = E_Trial
                E2 = E_Trial
                delta_Opt = delta_Old

            delta_Old = (delta_Opt + delta_Old)/2

            if self.Report>0:
                print("%4d %12.5f %7.3f %6.3f %6.4f %6.3f [ %7.2f %7.2f %7.2f ]"\
                      %(step, E, eV*(E-E_Trial), delta_Opt, UM, UOver,
                        (E-E_Trial)*eV, (E1-E_Trial)*eV, (E2-E_Trial)*eV ),
                      flush=True)


            # Unitary transform from AMat
            UMat = la.expm( delta_Opt*AMat)
            P = P.dot(UMat)
            self.CE = self.CE.dot(UMat)


            # Test for convergence after at least one step
            if (step>0) and ((np.abs(EOld - E)<DE_Break)
                             or (UM < UM_Break*(1-np.abs(MixUOld)))):
                self.Converged = True
                break

            EOld = E
            AMatOld = AMat
            UMOld = UM


        # Store properties of this run
        self.LastPlan = Plan
        self.Lastf = f

        # Get epsilonE and CE by rediagonalising down blocks
        self.epsilonE = self.epsilonE.reshape((-1,)) # Force 1D
        for k in FList:
            # Get the relevant block from the Fock matrix list
            FBlock = FList[k][FMap[k],:][:,FMap[k]]

            # Diagonalise the block
            epsBlock, uBlock = la.eigh(FBlock)
            # Update CE and epsilonE
            self.epsilonE[FMap[k]] = epsBlock
            self.CE[:,FMap[k]] = self.CE[:,FMap[k]].dot(uBlock)

        if self.Report>=3: # Full debug mode
            k1 = max(self.kh-2,0)
            k2 = min(self.kl+3,NOrb)

            print("Final U - with k = [ %s ]"%( " ".join(["%3d"%(x) for x in range(k1,k2)])))
            P = la.expm(AMat)
            print(P[k1:k2,k1:k2])
            print("Final f   = [ %s ]"%\
                  ( " ".join(["%7.3f"%(x) for x in f[k1:k2]])))
            print("Final eps = [ %s ]"%\
                  ( " ".join(["%7.3f"%(x) for x in self.epsilonE[k1:k2]])))
            print(" ".join(["%s = %8.3f"%(k, self.LastEns[k])
                        for k in ('ETV', 'Hx', 'xcDFA', 'Extra')]))

        # The last F is formed on epsilonE and CE
        SC = self.Engine.Get_SC(self.CE)
        
        Occ = Plan['1RDM']
        self.LastD = Occ.D(self.CE)
        
        if self.Converged:
            return E
        elif self.Props['Fail']:
            if self.Report>=0: print("Error! Failed to converge - returning None")
            return None
        else:
            if self.Report>0: print("Warning! Failed to converge - returning last calc")
            return E


class ExcitationHelper(OOExcitationHelper):
    1


if __name__ == "__main__":

    1
