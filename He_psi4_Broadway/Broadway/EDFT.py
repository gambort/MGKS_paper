__version__ = 1.6
__author__ = "Tim Gould"

import numpy as np
import scipy.linalg as la

from .LDAFits import *
from .PlanHandler import *

eV = 27.211
kcal = 627.5
kCal = 627.5
kJ = kcal*4.184

zero_round = 1e-5

class CoreExcitationHelper:
    """The main object for doing actual EDFT calculations.
    """
    def __init__(self, Engine,
                 xi = None,
                 Report = 0,
                 **kwargs):
        """Initialise the default excitation helper

        Args:
            Engine (Engine class): A valid Engine class with a minimal list of routines (see PDF for details)
            xi (float, optional): Explicitly set density-driven correlation parameter xi. Defaults to None=Automatic.
            Report (int, optional): Reporting level - higher = more info and -1 = silent. Defaults to 0.
        """
        # Level of reporting
        self.Report = Report

        self.SetEngine(Engine)

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

    # This deals with class-specific extras
    def SetExtraOptions(self, **kwargs):
        1

    # Initialise the engine and derived quantities
    def SetEngine(self, Engine):
        """Read key information from Engine and set up access to key routines

        Args:
            Engine (Engine class): A valid Engine class with a minimal list of routines (see PDF for details)
        """
        self.Engine = Engine

        self.NAtom = Engine.NAtom

        self.f = Engine.f*1.
        self.f_Occ = Engine.f_Occ*1.
        self.Na = Engine.Na
        self.Nb = Engine.Nb
        self.Nel = int(np.round(np.sum(self.f)))
        self.kh = Engine.kh
        self.kl = Engine.kl

        self.C0 = Engine.C*1.
        self.epsilon0 = Engine.epsilon*1.
        self.Sym_k = Engine.Sym_k

        # Use the initial orbitals
        self.epsilonE = Engine.epsilon*1.
        self.CE = Engine.C*1.

        # By default it hasn't converged
        self.Converged = False

    # Set default xi
    def Setxi(self, xi):
        """Set the scaling factor xi for density-driven correlations

        Args:
            xi (float): A positive value (takes |xi| anyway) and None goes to the default xi=0.32
        """
        # This is 
        if xi is None:
            self.xi = 0.32
        else:
            self.xi = np.abs(xi)

        
    # Set the internal properties
    def SetProps(self, MaxIter = 200, ShowIter = 20,
                 DECut = 1e-7, DepsCut = 1e-6,
                 Fail = True,
                 **kwargs):
        """Set up key properties that almost all solvers will require

        Args:
            MaxIter (int, optional): Maximum number of iterations. Defaults to 200.
            ShowIter (int, optional): Show at this number of iterations. Defaults to 20.
            DECut (_type_, optional): Energy cutoff for convergence. Defaults to 1e-7.
            DepsCut (_type_, optional): Eigenvalue cutoff for convergence. Defaults to 1e-6.
            Fail (bool, optional): Fail badly when things don't converge. Defaults to True.
        """
        self.Props['MaxIter'] = MaxIter
        self.Props['ShowIter'] = ShowIter
        self.Props['DECut']   = DECut
        self.Props['DepsCut'] = DepsCut
        self.Props['Fail']    = Fail

    # Update the internal properties
    def UpdateProps(self, **kwargs):
        """Update internal props using **kwargs as a dictionary
        """
        for v in self.Props:
            if v in kwargs: self.Props[v]  = kwargs[v]

    # Set the mixing properties
    def SetMix(self, MixC = 0.5, Mix = None, Mix2 = None,
               **kwargs):
        """_summary_

        Args:
            MixC (float, optional): Basic mixing parameter. Defaults to 0.5.
            Mix (_type_, optional): Core mixing parameter. Defaults to None=Auto.
            Mix2 (_type_, optional): Secondary mixing parameter. Defaults to Mix.
        """
        self.Props['MixC'] = MixC
        self.Props['Mix']  = Mix
        if Mix2 is None:
            self.Props['Mix2'] = Mix
        else:
            self.Props['Mix2'] = Mix2

    # Convert kFrom or kTo
    def Process_k(self, k='HOMO'):
        """Convert a free form description of an orbital into a valid index

        Args:
            k (str or int, optional): Explicit value of name. Defaults to 'HOMO'.

        Returns:
            int: Orbital index
        """
        if k is None: # Handle None
            return None 
        elif isinstance(k, str): # Handle strings
            Txt = k.upper().replace('HOMO', 'H').replace('LUMO', 'L')
            ID = Txt[0]
            if len(Txt)>1:
                Delta = int(Txt[1:])
            else: Delta = 0

            return {'H':self.kh, 'L':self.kl}[ID]+Delta
        elif hasattr(k, '__iter__'): # Handle iterables
            kout = []
            for kk in k:
                kout += [self.Process_k(kk),]
            return kout
        else: # If it's an integer leave it alone
            return k

    # Set the 'from' orbital by number (<=kh) or symmetry
    def SetFrom(self, k=None, Sym=None, **kwargs):
        """Set the 'from' orbital by number (<=kh) or symmetry

        Args:
            k (str or int, optional): See Process_k
            Sym (int, optional): Highest occupied in this symmetry - overrides k if not None. Defaults to None.

        Returns:
            int: Orbital index
        """
        k = self.Process_k(k)
        
        if not(Sym is None):
            # By symmetry
            self.kFrom = self.Engine.kh
            for q in range(self.Engine.kh):
                k = self.Engine.kh - q
                if self.Sym_k[k] == Sym:
                    self.kFrom = k
                    break
        elif (k is None) or (k>self.Engine.kh):
            self.kFrom = self.Engine.kh
        else:
            self.kFrom = k

        if self.Report>=3:
            print("k_From = %3d with sym %2d"\
                  %(self.kFrom, self.Sym_k[self.kFrom]))

        return self.kFrom

    # Set the 'to' orbital by number (>kh) or symmetry
    def SetTo(self, k=None, Sym=None, **kwargs):
        """Set the 'to' orbital by number (<=kh) or symmetry

        Args:
            k (str or int, optional): See Process_k
            Sym (int, optional): Lowest unoccupied in this symmetry - overrides k if not None. Defaults to None.

        Returns:
            int: Orbital index
        """
        k = self.Process_k(k)
        
        if not(Sym is None):
            # By symmetry
            self.kTo = self.Engine.kl
            for k in range(self.Engine.kl, self.Engine.nbf):
                if self.Sym_k[k] == Sym:
                    self.kTo = k
                    break
        elif (k is None) or (k<self.Engine.kl):
            self.kTo = self.Engine.kl
        else:
            self.kTo = k

        if self.Report>=3:
            print("k_To   = %3d with sym %2d"\
                  %(self.kTo  , self.Sym_k[self.kTo  ]))

        return self.kTo

    # Reset the current orbitals to their initial values
    def ResetOrbitals(self):
        """Reset the orbitals to their original value (by default inherited from Engine)
        """
        self.CE = self.C0*1.
        self.epsilonE = self.epsilon0*1.

    # Reset the current orbitals to their initial values
    # may eventually yield different answers to ResetOrbitals
    def ResetNaturalOrbitals(self):
        """Reset the orbitals to their original value (by default inherited from Engine)
        May eventually yield different answers to ResetOrbitals
        """
        self.CE = self.C0*1.
        self.epsilonE = self.epsilon0*1.

    # Make the current orbitals the 'initial' ones
    def FreezeOrbitals(self):
        """Set the original value (for Resets) to the current value
        """
        self.C0 = self.CE*1.
        self.epsilon0 = self.epsilonE*1.

    # Set the orbitals to specific values
    def SetOrbitals(self, epsilon, C):
        """Set the value for the internal orbitals
        """
        if not(epsilon.shape==self.epsilonE.shape) \
            or not(C.shape==self.CE.shape):
            print("Error - must make sure dimensions are right")
            quit()

        self.CE = C*1.
        self.epsilonE = epsilon*1.

    def _PrePlan(self, xi, kFrom, kTo):
        if xi is None: xi = self.xi
        if kFrom is None: kFrom = self.kFrom
        if kTo   is None: kTo   = self.kTo
        kFrom = self.Process_k(kFrom)
        kTo   = self.Process_k(kTo  )
        return xi, kFrom, kTo
    
    def SolvePol(self, Pol=0, 
                 Na=None, Nb=None,
                 xi=None,
                 kFrom = None, kTo = None,
                 Auto = True,
                 Show=False,
                 PlanOnly=False,
                 **kwargs):
        """Solve for a spin-polarized ground state

        Args:
            Pol (int, optional): Excess electron number - ignored if Na, Nb specified. Defaults to 0.
            Na (int, optional): Number of alpha electrons. Defaults to None.
            Nb (int, optional): Number of beta electrons. Defaults to None.
            PlanOnly (bool, optional): Return a Plan instead of solving for energy. Defaults to False.

        Returns:
            float: Energy after optimization
            Plan class: If PlanOnly=True
        """
        xi, kFrom, kTo = self._PrePlan(xi, kFrom, kTo)

        if not(Na is None) and not(Nb is None):
            Na, Nb = max(Na, Nb), min(Na, Nb)
        else:
            NEl = self.Nel # Number of electrons
            if (NEl+Pol-1)%2==0: # Pol is consistent with NEl
                Na = int((NEl+(Pol-1))//2)
                Nb = int((NEl-(Pol-1))//2)
            else: # Do for the cation
                Na = int((NEl-1+(Pol-1))//2)
                Nb = int((NEl-1-(Pol-1))//2)

        Plan = PlanHandler(xi=np.abs(xi))\
            .Polarized(Na, Nb, Auto=Auto)
        if Show: print(Plan)

        if PlanOnly: return Plan
        else: return self.Solver(Plan, **kwargs)

    def SolveGS(self, xi=None,
                kFrom = None, kTo = None,
                Show=False,
                PlanOnly=False,
                **kwargs):
        """Solve for an unpolarized ground state

        Args:
            Pol (int, optional): Excess electron number - ignored if Na, Nb specified. Defaults to 0.
            Na (int, optional): Number of alpha electrons. Defaults to None.
            Nb (int, optional): Number of beta electrons. Defaults to None.
            PlanOnly (bool, optional): Return a Plan instead of solving for energy. Defaults to False.

        Returns:
            float: Energy after optimization
            Plan class: If PlanOnly=True
        """
        xi, kFrom, kTo = self._PrePlan(xi, kFrom, kTo)

        if self.Nel%2 == 1: # Handle odd electron number
            return self.SolvePol(Na=self.Na, Nb=self.Nb,
                                 kFrom=kFrom, kTo=kTo, Show=Show, PlanOnly=PlanOnly,
                                 **kwargs)

        Plan = PlanHandler(xi=np.abs(xi))\
            .Singlet(self.Nel)
        if Show: print(Plan)

        if PlanOnly: return Plan       
        else: return self.Solver(Plan, **kwargs)

    def SolveTS(self, xi=None,
                kFrom = None, kTo = None,
                Auto = True,
                Show=False,
                PlanOnly=False,
                **kwargs):
        """ Solve the TS problem
        Setting PlanOnly returns the plan
        """
        xi, kFrom, kTo = self._PrePlan(xi, kFrom, kTo)

        if self.Nel%2 == 1: # Handle odd electron number
            print("Excitation is ill-defined for odd electron number")
            return None

        Plan = PlanHandler(xi=np.abs(xi))\
            .Triplet(self.Nel, From=[kFrom], To=[kTo], Auto=Auto)
        if Show: print(Plan)

        if PlanOnly: return Plan       
        else: return self.Solver(Plan, **kwargs)
    
    def SolveSS(self, **kwargs): return self.SolveSX(**kwargs)
    def SolveSX(self, xi=None,
                kFrom = None, kTo = None,
                PromotedFrom = False,
                PromotedTo = False,
                Show=False,
                PlanOnly=False,
                **kwargs):
        """ Solve the TS problem
        Setting PromotedFrom or PromotedTo to True creates an excitation
        without extra de-excitations
            (use it for lowest states of a given symmetry)
        Setting PlanOnly returns the plan
        """
        xi, kFrom, kTo = self._PrePlan(xi, kFrom, kTo)

        if self.Nel%2 == 1: # Handle odd electron number
            print("Excitation is ill-defined for odd electron number")
            return None

        if not(PromotedFrom) and not(PromotedTo):
            Plan = PlanHandler(xi=np.abs(xi))\
                .Singlet(self.Nel, From=[kFrom], To=[kTo])
        else:
            if PromotedFrom: From = self.kh
            else: From = kFrom
            if PromotedTo: To = self.kl
            else: To = kTo

            Plan = PlanHandler(xi=np.abs(xi))\
                .Singlet(self.Nel, From=[From], To=[To])
            Plan.Swap(From,kFrom)
            Plan.Swap(To,kTo)
        if Show: print(Plan)

        if PlanOnly: return Plan       
        else: return self.Solver(Plan, **kwargs)
    
    def SolveDX(self, xi=None,
                kFrom = None, kTo = None,
                kFrom2 = None, kTo2 = None,
                Show=False,
                PlanOnly=False,
                **kwargs):
        """ Solve the DX problem
        Setting PlanOnly returns the plan
        Defaults are as follows:
        if _both_ kFrom2 and kTo2 are set to None
          no degeneracies:  kFrom^2 -> kTo^2
          degenerate kTo:   kFrom^2 -> kTo,kTo+1
          degenerate kFrom: kFrom-1,kFrom -> kTo^2
          both degenerate:  kFrom-1,kFrom -> kTo,kTo+1
        else
        1) set unspecified to be same
        2) case:            kFrom, kFrom2 -> kTo, kTo2
        """
        xi, kFrom, kTo = self._PrePlan(xi, kFrom, kTo)

        if self.Nel%2 == 1: # Handle odd electron number
            print("Excitation is ill-defined for odd electron number")
            return None

        # If kFrom2 and kTo2 are None use the degeneracies to determine excitation
        if (kFrom2 is None) and (kTo2 is None):
            # Handle degeneracies
            if np.abs(self.epsilon0[kFrom-1]-self.epsilon0[kFrom])<zero_round: From = [kFrom-1, kFrom]
            else: From = [kFrom, kFrom]
            if np.abs(self.epsilon0[kTo+1]-self.epsilon0[kTo])<zero_round: To = [kTo, kTo+1]
            else: To = [kTo, kTo]
        else:
            if kFrom2 is None: kFrom2 = kFrom
            else: kFrom2 = self.Process_k(kFrom2)
            if kTo2 is None: kTo2 = kTo
            else: kTo2 = self.Process_k(kTo2)
            From = [kFrom, kFrom2]
            To   = [kTo  , kTo2  ]

        Plan = PlanHandler(xi=np.abs(xi))\
            .Singlet(self.Nel, From=From, To=To)
        if Show: print(Plan)

        if PlanOnly: return Plan       
        else: return self.Solver(Plan, **kwargs)
        
    def GetEnergy(self, Plan, C=None,
                  StoreParts = False):
        """
        Compute the energy and return it and the _default_
        Fock matrix acting on C

        C defaults to self.CE

        returns E, F*C

        if StoreParts is set to True it stores all ingredients
        (energies and Fock C) for reuse by other routines, to
        avoid excessive calculations.
        """
        if C is None: C = self.CE

        self.fE = Plan['1RDM'].f()

        # Generate the Fock recipes
        if Plan.Recipes is None:
            Plan.GenerateFockRecipes(NBas=C.shape[0])

        FockOccs    = Plan.Recipes['FockOccs']
        FockWeights = Plan.Recipes['FockWeights']
        FockExtras  = Plan.Recipes['FockExtra']
        
        k0 = Plan.Recipes['Virtual'][-1] # Use the last of these by default
        
        HxOccs = Plan.Recipes['FockOccs']['Hx']
        HxPlan = FockWeights[k0]['Hx']

        xcDFAOccs = Plan.Recipes['FockOccs']['xcDFA']
        xcDFAPlan = FockWeights[k0]['xcDFA']

        ExtrasEnergyPlan = Plan.ExtraList()
        ExtrasFockPlan = FockExtras[k0]

        Occ = Plan['1RDM']
        ff = Occ.PadTo(C.shape[1]).np

        FCTV = self.Engine.Get_TC(C) + self.Engine.Get_VC(C)
        ETV = np.einsum('pk,pk,k->', C, FCTV, ff)

        self.LastEns = {'ETV':ETV, }
        if StoreParts: self.LastEns['FCTV'] = FCTV

        # Initialise the energy and (up) Fock matrix using
        # the trivial kinetic (T) and external potential (V)        
        E = ETV
        FC = FCTV*1.
        if (self.Report>=20):
            print("W = %.3f, ETV = %10.5f"%(1., ETV))

        # Add the Hartree-exchange terms from the weighted
        # sum of existing DFAs (with appropriate scaling of x)
        self.LastEns['Hx'] = 0.
        self.LastEns['Hx Parts'] = []
        for Occ, (WE, WFa, WFb) in zip(HxOccs, HxPlan):
            Ca, Cb = Occ.Cab(C) # Get alpha and beta orbitals
            if Occ.IsDbl():
                EHx, FCHxa = self.Engine.GetHx_FC(C, Ca=Ca)
                FCHxb = FCHxa
                FC += (WFa+WFb)*FCHxa
            else:
                EHx, FCHxa, FCHxb = self.Engine.GetHx_FC(C, Ca=Ca, Cb=Cb, BothSpin=True)
                FC += WFa*FCHxa + WFb*FCHxb

            if self.Report>=10:
                print("%.2f %.2f %.2f %.1f %.1f %10.4f"\
                    %(WE, WFa, WFb, 
                        Ca.shape[1], Cb.shape[1],
                        EHx))

            E += WE*EHx
            self.LastEns['Hx'] += WE*EHx
            if StoreParts:
                self.LastEns['Hx Parts'] += [(EHx, FCHxa*1., FCHxb*1.)]
            
        # Add the xc DFA terms from the weighted
        # sum of existing DFAs (with appropriate scaling of x)
        self.LastEns['xcDFA'] = 0.
        self.LastEns['xcDFA Parts'] = []
        for Occ, (WE, WFa, WFb) in zip(xcDFAOccs, xcDFAPlan):
            Ca, Cb = Occ.Cab(C)
            if Occ.IsDbl():
                Exc, FCxca = self.Engine.GetDFA_FC(C, Ca=Ca)
                FCxcb = FCxca
                FC += (WFa+WFb)*FCxca
            else:
                Exc, FCxca, FCxcb = self.Engine.GetDFA_FC(C, Ca=Ca, Cb=Cb, BothSpin=True)
                FC += WFa*FCxca + WFb*FCxcb

            E += WE*Exc
            self.LastEns['xcDFA'] += WE*Exc
            if StoreParts:
                self.LastEns['xcDFA Parts'] += [(Exc, FCxca*1., FCxcb*1.)]

        # Add the extra terms (of J and K form) that come from
        # EST' as well as anything missed in the Hartree term
        #
        # Note, the energy and Fock can be treated inconsistently
        # here to accommodate approximations
        self.LastEns['Extra'] = 0.
        self.LastEns['Extra Parts'] = []

        if not(ExtrasEnergyPlan is None) and (len(ExtrasEnergyPlan)>0):
            EList, FCList = {}, {}

            # Build the cache
            for _, _, j, k, Kind in ExtrasEnergyPlan + ExtrasFockPlan:
                Pair = (min(j,k),max(j,k))
                EIndx = (Pair,Kind)
                FIndx = (j,Kind)

                if not(FIndx in FCList) or not(EIndx in EList):
                    EEx_, FCEx_ = self.Engine.GetExtra_FC(C, Extra=[(1.,1., j, k, Kind)])
                    EList[EIndx] = EEx_
                    FCList[FIndx] = FCEx_

            # Calculate the energy
            EEx = 0.
            for WE, _, j, k, Kind in ExtrasEnergyPlan:
                Pair = (min(j,k),max(j,k))
                EEx += WE*EList[(Pair,Kind)]

            FCEx = 0.
            for _, WF, j, k, Kind in ExtrasFockPlan:
                FCEx += WF*FCList[(j,Kind)]

            E += EEx
            FC += FCEx

            self.LastEns['Extra'] = EEx
            if StoreParts:
                self.LastEns['Extra Parts'] = {'EList':EList, 'FCList':FCList }

        if self.Report>=10:
            print("One-electron : %10.4f"%(self.LastEns['ETV']))
            print("Two-electron : %10.4f"%(self.LastEns['Hx']))
            print("DFT energy   : %10.4f"%(self.LastEns['xcDFA']))
            print("Extra energy : %10.4f"%(self.LastEns['Extra']))

        # Finally, add the nuclear-nuclear term to the energy
        E += self.Engine.Enn

        self.LastPlan = Plan

        # Return E and F(up)
        return E, FC

    # Get the energy and the orbital Fock matrices
    # Note, only computes different Fock matrices
    def GetEnergyFocks(self, Plan, C=None,
                       Raw=False):
        """
        Compute the energy and full list of Focks

        by default or Raw==False:
            returns E, list of[(C.T)*F*C], list of Maps

        if Raw==True:
            returns E, list of [F*C], list of Maps

        C defaults to self.CE
        """
        if C is None: C = self.CE

        self.fE = Plan['1RDM'].f()

        E, _ = self.GetEnergy(Plan, C=C, StoreParts=True)

        FockOccs    = Plan.Recipes['FockOccs']
        FockWeights = Plan.Recipes['FockWeights']
        FockExtras  = Plan.Recipes['FockExtra']

        FockList = {}
        for k in FockWeights:
            FC = self.LastEns['FCTV']*1.

            HxPlan = FockWeights[k]['Hx']
            for KFock, (_, WFa, WFb) in enumerate(HxPlan):
                if self.Report>=10:
                    print("%3d : Hx : Occ(%d) with %.3f %.3f"%(k, KFock, WFa, WFb))
                _, FCa, FCb = self.LastEns['Hx Parts'][KFock]
                FC += WFa*FCa + WFb*FCb

            xcDFAPlan = FockWeights[k]['xcDFA']
            for KFock, (_, WFa, WFb) in enumerate(xcDFAPlan):
                if self.Report>=10:
                    print("%3d : xc : Occ(%d) with %.3f %.3f"%(k, KFock, WFa, WFb))
                _, FCa, FCb = self.LastEns['xcDFA Parts'][KFock]
                FC += WFa*FCa + WFb*FCb

            ExtrasFockPlan = FockExtras[k]
            for _, WF, j, k, Kind in ExtrasFockPlan:
                if self.Report>=10:
                    print("%3d : Ex : %.3f %d %d %s"%(k, WF, j, k, Kind))
                FIndx = (j,Kind)
                if not(FIndx in self.LastEns['Extra Parts']['FCList']):
                    _, FCEx = self.Engine.GetExtra_FC(C, Extra=[(1.,1., j, k, Kind)])
                    self.LastEns['Extra Parts']['FCList'][FIndx] = FCEx
                else:
                    FCEx = self.LastEns['Extra Parts']['FCList'][FIndx]

                FC += WF*FCEx

            if Raw: FockList[k] = FC
            else: FockList[k] = (C.T).dot(FC)

        FockMap = Plan.Recipes['Map']

        # Set the appropriate Fock matrix for the virtual states
        FVirtual = 0.
        WVirtual = 1./ len(Plan.Recipes['Virtual'])
        for k in Plan.Recipes['Virtual']:
            FVirtual += FockList[k] * WVirtual
        FockList[-1] = FVirtual
                
        return E, FockList, FockMap

        
    def GetEnergyCorrection(self, Plan=None, C=None):
        """ Get the DFA "singlet" energy correction (not recommended) """
        if Plan is None: Plan = self.LastPlan
        if C is None: C = self.CE

        # No correction for non-singlets
        if not(Plan['Singlet']): return 0.

        WList = []
        DList = []
        for W, Occ in Plan['xcDFA']:
            if Occ.IsDbl(): continue # Skip double

            kSwap = np.argwhere(Occ.f()==1.).reshape((-1,))
            k_to_swap = kSwap[0]

            OccS = Occ.Copy()
            OccS.SwapSpin(k_to_swap)

            Da, Db = Occ.Dab(C)
            DSa, DSb = OccS.Dab(C)
            WList += [-2*W,2*W]
            DList += [(Da,Db), (DSa,DSb)]

        ExcList = self.Engine.GetEDFA_ID('cPBE', DList)
        DeltaExc = np.dot(WList, ExcList)

        return DeltaExc


    def Solver(self, Plan, kFrom=None, kTo=None,
               **kwargs):
        """Generic solver routine - overwritten in inherited classes"""
        return self.SolverFrozen(Plan, kFrom=kFrom, kTo=kTo,
                                 **kwargs)

    def SolverFrozen(self, Plan = None,
                     **kwargs):
        """ 
        Frozen solver routine using current orbitals
        This evaluates EDFT@DFT if called before orbital updates
        or EDFT@Last run if called later
        """
        if Plan is None: Plan = self.LastPlan

        epsE = self.epsilonE
        CE = self.CE
        self.fE = Plan['1RDM'].f()

        E, F = self.GetEnergy(Plan, CE)
        Occ = Plan['1RDM']

        self.LastPlan = Plan
        self.Lastf = Occ.f()*1.
        self.LastD = Occ.D(CE)
        
        return E
    
    def Solver1Order(self, Plan = None, Scale=1.,
                     eta_eps = 1e-5, **kwargs):
        """ 
        First order correction to current energy
        WARNING! This is not tested.
        This evaluates EDFT@DFT if called before orbital updates
        or EDFT@Last run if called later
        """

        self.ResetOrbitals()
        E0, FockList, FockMap = self.GetEnergyFocks(Plan)

        NOcc = Plan['1RDM'].NOcc
        f = Plan['1RDM'].np[:NOcc]
        eps = self.epsilonE

        Num = 0.*FockList[-1]
        for k in FockMap:
            F_k = FockList[k]
            Indx = FockMap[k]
            Indx = Indx[Indx<len(f)] # Make sure we only keep up to len(f)
            if len(Indx)==0: continue

            Num[:,Indx] += f[Indx] * F_k[:,Indx]**2

        DE = 0.
        for i in range(len(eps)):
            for j in range(len(eps)):
                if np.abs(eps[i]-eps[j])<eta_eps: continue
                DE += Num[j,i]/(eps[i]-eps[j])

        
        return E0 + DE*Scale

        
    def GetFullDipole(self):
        """ Get the dipole of the current density """
        return self.GetDipole(f=self.Lastf)
    
    def GetTransDipole(self, i=None, a=None):
        """ Get the current transition dipole """
        return self.GetDipole(i=i, a=a)
        
    def GetDipole(self, i=None, a=None, f=None, C=None):
        """ 
        Calculate a dipole.
        If f is not specified:
          i defaults to From
          a defaults to To
          then returns \sqrt{2}<\phi_i|r|\phi_a>
        if f is specified:
          returns \int n(r)r dr where n(r)=\sum_i f_i|\phi_i(r)|^2

        C defaults to self.CE
        """
        if C is None: C = self.CE

        if not(f is None): # Specify a 1RDM
            CP = C[:,:len(f)]
            DCP = self.Engine.Get_1BC(CP, 'Dipole')
            return np.einsum('xpk,pk,k->x', DCP, CP, f)

        if i is None: i = self.kFrom
        if a is None: a = self.kTo

        Ci = C[:,i]
        DCa = self.Engine.Get_1BC(C[:,a], 'Dipole')
        return np.sqrt(2) * np.einsum('xp,p->x', DCa, Ci)
    
    def GetRDM(self, f = None, C = None,
               pyscf=False):
        """
        Get the current alpha and beta 1RDMs
          Da and Db in the orbital basis
                  
        Not recommended in the LowMem solver    
        
        Setting pyscf to True gets it in pyscf style    
        """
        if f is None: f = self.Lastf
        if C is None: C = self.CE

        Occ = OccHelper(f)

        Da, Db = Occ.Dab(C)

        if pyscf:
            D = np.zeros((2, Da.shape[0], Da.shape[0]))
            D[0,:,:] = Da
            D[1,:,:] = Db
            return D

        return Da, Db

class CoreEDFTHelper(CoreExcitationHelper):
    """Alias for CoreExcitationHelper
    """

if __name__ == "__main__":
    
   1
