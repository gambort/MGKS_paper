__version__ = 1.5
__author__ = "Tim Gould"

from .OOEDFT import *
from .PlanHandler import *

import numpy as np
import scipy.linalg as la
import numpy.random as ra

eV = 27.211

zero_round = 1e-5

def ReNorm(C, SC_Op):
    """ Renormalize all eigenvectors in C_{pk} using S_{pq}"""
    return C / np.sqrt(np.einsum('pk,pk->k',C,SC_Op(C)))

def ReOrtho(C, SC_Op):
    """ Reorthogonalise the matrix on S - this is an iterative
    'purification' that works when C is nearly orthogonal

    Note, I stumbled upon this algorithm by fucking around with
    linear algebra and can't prove why it works well.
    It seems to be a Newton variant!
    """
    ErrOld = 1e10
    C0 = ReNorm(C, SC_Op)
    for step in range(10):
        C = ReNorm(C, SC_Op) # First renormalise
        Q = SC_Op(C) # Define Q = SC so that C^TQ=I is our goal

        # Define the error matrix
        E = (C.T).dot(Q) - np.eye(C.shape[1])

        # And its scalar form
        Err = np.mean(E**2)
        
        if Err<1e-12: break

        # Fail 'gracefully' if algorithm fails
        if Err>ErrOld: 
            print("%d Err = %.2e, ErrOld = %.2e"%(step, Err, ErrOld))
            print("Error is increasing - failed and returning ReNorm(C)")
            return C0

        # Get the correction term
        D = -0.5 * Q.dot(la.solve((Q.T).dot(Q), E))

        # Update C and move on
        C = C + D

        # Store the last error for checking failures
        ErrOld = Err

    return C

# Orbital optimized but with limited excited orbitals
class OOLowExcitationHelper(OOExcitationHelper):
    """
    The OOLowExcitationHelper class is for lower memory OO solutions'

    It solves the OO problem in a reduced set of orbitals (by default
    up to 4 virtual orbitals on the double occupied GS) and then
    updates the basis by using the residuals.

    It is very slow at the moment as the residuals are introduced
    through an unoptimized gradient descent with line search.
    """
    def __init__(self, Engine, 
                 NExcite = 8, # Specifies the number of virtual orbitals to use
                 xi = None,
                 Report = 0,
                 **kwargs):
        # Level of reporting
        self.Report = Report

        self.SetEngine(Engine, NExcite)

        if self.Engine.Has_RS:
            # If passed RS assume rs-EDFA
            if Report>=0: print("Assuming EDFA and setting xi=0.32")
            self.Setxi(-0.32)
        else:
            self.Setxi(0.0)

        # This is the general properties
        self.Props = {}
        
        self.SetProps(**kwargs)
        self.SetMix(**kwargs)
        
        self.SetFrom(**kwargs)
        self.SetTo(**kwargs)

        self.SetExtraOptions(**kwargs)
            
    # Initialise the engine and derived quantities
    def SetEngine(self, Engine, NExcite = 8):
        self.Engine = Engine

        self.NAtom = Engine.NAtom

        self.f = Engine.f
        self.f_Occ = Engine.f_Occ
        self.Na = Engine.Na
        self.Nb = Engine.Nb
        self.Nel = int(np.round(np.sum(self.f)))
        self.kh = Engine.kh
        self.kl = Engine.kl

        self.NOrb = Engine.kh+1+NExcite

        self.C0 = Engine.C[:,:self.NOrb]*1.
        self.epsilon0 = Engine.epsilon[:self.NOrb]
        self.f = self.f[:self.NOrb]
        self.f_Occ = self.f_Occ[:self.NOrb]
        self.Sym_k = Engine.Sym_k[:self.NOrb]

        # Use the initial orbitals
        self.CE = self.C0*1.
        self.epsilonE = self.epsilon0*1.

        # By default it hasn't converged
        self.Converged = False

    # Set the internal properties
    def SetProps(self, 
                 MaxIter = 5, RepeatIter = 80,
                 DECut = 1e-7, DepsCut = 1e-6,
                 DELoop = 1e-5,
                 **kwargs):
        self.Props['MaxIter'] = MaxIter
        self.Props['ShowIter'] = 10
        self.Props['RepeatIter'] = RepeatIter
        self.Props['DECut']   = DECut
        self.Props['DepsCut'] = DepsCut

        self.Props['DELoop']   = DELoop

        self.Props['Fail'] = False # Return E when OO doesn't converge

    # Generic solver routine
    def Solver(self, Plan, **kwargs):
        return self.Solver_ls(Plan, **kwargs)

    # Solver routine based on updates via residuals
    def Solver_ls(self, Plan, kFrom=None, kTo=None,
               Reset = True, # Default Reset

               GradientKind = 'Simple', # Which gradient descent            

               wThresh = 1e-6, # Keep eigenvalues of this much

               fStar = 0.5, # Effective occ for double occ
               epsThresh = 0.1, # Define a 'small' eigenvalue
               BMax = 0.5, # Maximum value of B allowed

               Drag = 0.1, # Mix in this much old step
               xStep = -1, # Use fixed step (set <=0 for line search)
               xMax = 1.0, # Maximum mix
               **kwargs):
        
        EOld = 1e3
        Converged = False

        DCOld = None
        for loop in range(self.Props['RepeatIter']):
            ###################################################################
            # Optimize using the current orbitals
            E = self.SolverOOEDFT_LS(Plan, 
                                     # Always run for shortish times
                                     MaxIter = self.Props['MaxIter'],
                                     # Reset the orbitals on first time only
                                     Reset = (Reset and (loop==0)), 
                                     # Do a line search on first time only
                                     NLineWarm = (self.Props['MaxIter'] if loop==0 else 0),
                                     # Pass any other arguments
                                     **kwargs)

            ###################################################################
            # Then update the orbitals
            #
            # Uses a line-search gradient descent in a basis of residuals

            # Get Fi C for all i and C
            E, FList, FMap = self.GetEnergyFocks(Plan, Raw=True)

            # Some matrices to save typing later
            NOrbs = self.CE.shape[1]

            C = 1.*self.CE
            SC = self.Engine.Get_SC(C)

            # Get the residuals in R
            R = 0.*self.CE
            for k in FList:
                kp = FMap[k]
                FC_k = FList[k][:,kp]
                R[:,kp] = self.Engine.Get_InvSC(FC_k) # This needs refining (NOT TRUE LOWMEM)

            # Ensure the residual are orthogonal to C
            R -= np.einsum("pk,pj,qj->qk", R, SC, C)
            # Calculate the overlap of residuals
            O = (R.T).dot(self.Engine.Get_SC(R))

            if GradientKind.upper()[:2]=="OR": # NOT RECOMMENDED
                # Use states that are orthogonal to each other and the current orbitals
                # Find a set of internally orthogonal residuals p
                #
                # May be better if (for some reason) you have a high fraction of all
                # possible orbitals
                w, v = la.eigh(O)
                if w.max()<wThresh:
                    # None        
                    DC = None       
                    Converged = True
                else:
                    # Some
                    ii = np.abs(w)>wThresh
                    P = np.einsum('kx,x->kx', v[:,ii], 1/np.sqrt(w[ii]))
                    p = R.dot(P)

                    # Normalized energy is:
                    # \sum_i f_i [ \epsilon_i + \sum_{x}B_{xi}(L_{xi} - \epsilon_iB_{xi}) ]
                    # where L = OP which is minmized when
                    # B = 1/diag(epsilon) L
                    # But, we replace 1/diag(epsilon) by a non-problematic transition for
                    # small values of epsilon

                    epsInv = self.epsilonE/(epsThresh**2 + self.epsilonE**2)
                    L = O.dot(P)
                    B = np.einsum('i,ix->ix', epsInv, L)
                    DC = p.dot(B.T) # Change in C
            else: # RECOMMENDED! Default is the simple approach
                # Simple diagonal expansion along the residuals
                if np.max(np.diag(O))<2*wThresh: # Residuals too small
                    DC = None
                    Converged = True
                else:
                    # Prefactor has two part:
                    # B_f is the occupation dependence 0<=f(2+f*-f)/(1+f*/2)^2<=1
                    # B_eps is a damped `exact' form
                    #     0 <= BMax * tanh(1/BMax * 1/sqrt(eps^2 + O)) <= BMax
                    # Updates are diaognal on R

                    fE = Plan['1RDM'].PadTo(NOrbs).np # Occupation factors

                    B_f = fE*(2+fStar-fE)/(1+fStar/2)**2
                    B_eps = BMax * np.tanh(1/BMax * 1/np.sqrt(self.epsilonE**2 + np.diag(O)))

                    B = - B_f * B_eps
                    DC = R*B # Update C

            if not(DC is None):
                # Introduce some drag
                if not(DCOld is None): DC += Drag*(DCOld - DC)
                #DC -= C.dot((SC.T).dot(DC))

                if xStep<=0.:
                    # Line search
                    Ep, _ = self.GetEnergy(Plan, C=ReOrtho(C+DC, self.Engine.Get_SC))
                    Em, _ = self.GetEnergy(Plan, C=ReOrtho(C-DC, self.Engine.Get_SC))

                    pE = np.polyfit([-1,0,1],[Em,E,Ep],2)
                    if pE[0]>0.: # Has a minima so find it
                        x = -pE[1]/(2*pE[0])
                        x = np.sign(x)*min(np.abs(x), xMax)
                    else: # Use the edge with the lowest energy
                        if Ep>=Em: x = 1.
                        else: x = -1.
                    EN = np.polyval(pE, x)
                else:
                    x = xStep
                    EN = E

                self.CE = ReOrtho(C + x*DC, self.Engine.Get_SC)


                # DCOld is made orthogonal to new DC
                DCOld = DC*1.
            
                if (self.Report==0 and (loop%10)==0) \
                    or (self.Report>0):
                    print("%3d E = %10.5f, ENew = %10.5f @ %5.2f with %d"%(loop, E, EN, x, DC.shape[1]))
            else:
                if self.Report>=0:
                    print("%3d E = %10.5f, No residuals"%(loop, E))

            if np.abs(E-EOld)<self.Props['DELoop'] or Converged: break
            EOld = E


        return E

class ExcitationHelper(OOLowExcitationHelper):
    1


if __name__ == "__main__":

    1
