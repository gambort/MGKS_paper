__version__ = 1.6
__author__ = "Tim Gould"

import psi4
import numpy as np
import scipy.linalg as la

import numpy.random as np_ra

from psi4Engine.ERIHelpers import *
from psi4Engine.OtherHelpers import *
from psi4Engine.SymHelpers import *

#from psi4Engine.LibPairDens import *

eV = 27.211
zero_round = 1e-6

np.set_printoptions(precision=4, suppress=True, floatmode="fixed")


#################################################################################################
# This is the main psi4 engine
#################################################################################################

class psi4Engine:
    def __init__(self, wfn, wfn_Ref = None,
                 alpha = None, beta = None, omega = None,
                 ComputeERIA = False,
                 ComputeERI = False,
                 InternalSymHelper = False,
                 Report = 0):
        """_summary_

        Args:
            wfn (psi4_wavefunction): A valid psi4 wavefunction object (HF or DFT)
            wfn_Ref (psi4_wavefunction, optional): Read key elements from this instead. Defaults to None=ignore.
            alpha (float, optional): alpha parameter for K. Defaults to internal.
            beta (float, optional): alpha parameter for K_w (I think). Defaults to internal.
            omega (float, optional): range-separation parameter. Defaults to internal.
            ComputeERIA (bool, optional): Force an (very large) ERIA to be computed. Defaults to False.
            ComputeERI (bool, optional): Force an (even larger) ERI to be computed. Defaults to False.
            InternalSymHelper (bool, optional): Force a really crap symmetry helper that shouldn't be used. Defaults to False.
            Report (int, optional): Reporting level - higher = more info and -1 = silent. Defaults to 0.
        """
        self.__kind__ = "__psi4Engine__"

        self.Report = Report
        self.wfn = wfn
        if wfn_Ref is None:
            wfn_Ref = wfn

        
        self.DirectFock = True # True means the Fock space is small enough for direct solutions

        self.JKHelp = None
        self.ERIA = None

        if InternalSymHelper:
            print("Warning this does not work - quitting")
            self.SymHelp = SymHelperInternal(wfn)
            quit()
        else:
            self.SymHelp = SymHelper(wfn)

        self.Has_RS = None # Required by all engines

        self.Na = wfn_Ref.nalpha() # Required by all engines
        self.Nb = wfn_Ref.nbeta() # Required by all engines

        # Set up the occupations
        self.f = np.ones(self.Na) # Required by all engines
        self.f[:self.Nb] += 1.

        self.Da = self.SymHelp.Dense(wfn_Ref.Da().to_array()) # Required by all engines
        self.Db = self.SymHelp.Dense(wfn_Ref.Db().to_array()) # Required by all engines
        self.D = self.Da + self.Db # Required by all engines
        self.epsilon = self.SymHelp.epsilon() # Required by all engines
        self.C = self.SymHelp.C() # Required by all engines
        self.F = self.SymHelp.Dense(wfn_Ref.Fa().to_array())

        # Number of basis functions
        self.NBas = wfn_Ref.nmo()  # Required by all engines
        # Total number of occupied orbitals
        self.NOcc = (wfn_Ref.nalpha() + wfn_Ref.nbeta() + 1)//2  # Required by all engines
        # Index of HOMO
        self.kh = self.NOcc-1  # Required by all engines
        # Index of LUMO
        self.kl = self.kh+1  # Required by all engines

        # Symmetry group by basis function
        self.Sym_k = self.SymHelp.s_sorted # Required by all engines

        # Default occupation factors
        self.f_Occ = 2.*np.ones((self.NOcc,))

        if self.SymHelp.NSym>1 and self.Report>=0:
            self.SymHelp.SymReport(self.kh)


        self.nbf = self.wfn.nmo() # Number of basis functions
        self.ComputeERIA = ComputeERIA
        self.ComputeERI  = ComputeERI

        self._SetDFA(wfn)
        self.OverwriteHybridParams(alpha=alpha, beta=beta, omega=omega)

        self.Update_from_wfn()

    def Update_from_wfn(self, wfn=None):
        """Set everything from a psi4 wfn object + optional scales

        Args:
            wfn (psi4_wavefunction): A valid psi4 wavefunction object (HF or DFT)
        """
        if wfn is None:  wfn = self.wfn
        else: self.wfn = wfn

        basis = wfn.basisset()
        self.basis = basis
        self.nbf = wfn.nmo()   # Required by all engines
        self.NAtom = self.basis.molecule().natom()   # Required by all engines

        if not(basis.has_puream()):
            print("Must use a spherical basis set, not cartesian")
            print("Recommend rerunning with def2 or cc-type basis set")
            quit()


        self.Enn = self.basis.molecule().nuclear_repulsion_energy() # Required by all engines

        self.mints = psi4.core.MintsHelper(self.basis)

        self.S_ao = self.mints.ao_overlap().to_array(dense=True) # Recommended if compact basis set engine
        self.T_ao = self.mints.ao_kinetic().to_array(dense=True) # Recommended if compact basis set engine
        self.V_ao = self.mints.ao_potential().to_array(dense=True) # Recommended if compact basis set engine
        try:
            self.VECP_ao = self.mints.ao_ecp().to_array(dense=True)
            if np.max(np.abs(self.VECP_ao))>0.:
                self.V_ao += self.VECP_ao
                print("Has ECP - now added to potential")
        except:
            1
            
        self.H_ao = self.T_ao + self.V_ao

        self.V_ao_Init = 1.*self.V_ao

        self.Dip_ao = np.array([x.to_array(dense=True)
                                for x in self.mints.ao_dipole()])
        self.Quad_ao = np.array([x.to_array(dense=True)
                                for x in self.mints.ao_quadrupole()])


        self._SetDFA(wfn)

        # Compute ERIA if asked
        if self.ComputeERIA:
            self.ERIA, self.ERIA_w \
                = GetDensityFit(wfn, self.basis, self.mints, self.omega)
        else: self.ERIA, self.ERIA_w = None, None

        # Compute ERI if asked (generally enormous - need lots of RAM):
        if self.ComputeERI:
            self.ERI_ao = self.mints.ao_eri().to_array()



        self.JKHelp = JKHelper(wfn, self.omega)

        # Atom by basis function
        self.Atom_k = np.array(
            [ self.basis.function_to_center(i) for i in range(self.nbf) ]
        )


    def __del__(self):
        """Delete the object and explicitly tidy up the big arrays
        """
        #print("Closing psi4Engine")
        if not(self.JKHelp is None): self.JKHelp.__del__()
        if not(self.ERIA is None): del self.ERIA


    def _SetDFA(self, wfn=None, ScaleDFA_x=1., ScaleDFA_c=1.):
        """Set the DFA from a psi4 wfn object + optional scales

        Args:
            wfn (psi4_wavefunction): A valid psi4 wavefunction object (HF or DFT)
            ScaleDFA_x (float, optional): Scale the DFA for exchange. Defaults to 1..
            ScaleDFA_c (float, optional): Scale the DFA for correlation. Defaults to 1..
        """
        if wfn is None: wfn = self.wfn

        # These are used for DFA calculations
        try:
            self.VPot = wfn.V_potential() # Note, this is a VBase class
            self.DFA = self.VPot.functional()
        except:
            self.VPot = None
            self.DFA = None

        # Ensure we have UKS for the excitations
        if not(self.DFA is None):
            # Convert DFA from RKS to UKS
            self.DFAU = sf_RKS_to_UKS(self.DFA, ScaleDFA_x=ScaleDFA_x, ScaleDFA_c=ScaleDFA_c)
            # Make a new VPot for the UKS DFA
            self.VPot = psi4.core.VBase.build(self.VPot.basis(), self.DFAU, "UV")
            self.VPot.initialize()

        # Work out the range-separation and density functional stuff
        self.xDFA, self.xDFA_w = 0., 0.
        self.omega = None
        self.alpha = 0.
        self.beta = 0.
        if not(self.DFA is None):
            # Implement DFAs
            if self.DFA.is_x_hybrid():
                # Hybrid functional
                self.alpha = self.DFA.x_alpha()
                self.xDFA = 1. - self.alpha
                if self.DFA.is_x_lrc():
                    # Range-separated hybrid
                    self.omega = self.DFA.x_omega()
                    self.beta = self.DFA.x_beta()

                    if self.Report>=0:
                        print("# RS hybrid alpha = %.2f, beta = %.2f, omega = %.2f"\
                              %(self.alpha, self.beta, self.omega),
                              flush=True)

                    self.xDFA = 1. - self.alpha # 1. - self.alpha
                    self.xDFA_w = - self.beta # - self.beta
            else:
                # Conventional functional
                self.xDFA = 1.

            # psi4 matrices for DFA evaluation
            self.DMa = psi4.core.Matrix(self.nbf, self.nbf)
            self.DMb = psi4.core.Matrix(self.nbf, self.nbf)
            self.VMa = psi4.core.Matrix(self.nbf, self.nbf)
            self.VMb = psi4.core.Matrix(self.nbf, self.nbf)
        else:
            self.alpha = 1. # Pure HF theory
            self.xDFA = 0. # Pure HF theory

        self.Has_RS = not(self.omega is None)  # Required by all engines

    def OverwriteHybridParams(self, alpha=None, beta=None, omega=None):
        """Overwrite the hybrid parameters only

        Args:
            alpha (float, optional): alpha parameter for K. Defaults to internal.
            beta (float, optional): alpha parameter for K_w (I think). Defaults to internal.
            omega (float, optional): range-separation parameter. Defaults to internal.
        """
        if not(alpha is None): self.alpha = alpha
        if not(beta is None): self.beta = beta
        if not(omega is None): self.omega = omega

        self.xDFA = 1. - self.alpha # 1. - self.alpha
        self.xDFA_w = - self.beta # - self.beta

        self.Has_RS = not(self.omega is None)

    def OverwriteHybrid(self, alpha=None, beta=None, omega=None):
        """Overwrite the hybrid in full

        Args:
            alpha (float, optional): alpha parameter for K. Defaults to internal.
            beta (float, optional): alpha parameter for K_w (I think). Defaults to internal.
            omega (float, optional): range-separation parameter. Defaults to internal.
        """
        if not(alpha is None): self.alpha = alpha
        if not(beta is None): self.beta = beta
        if not(omega is None): self.omega = omega

        self.xDFA = 1. - self.alpha # 1. - self.alpha
        self.xDFA_w = - self.beta # - self.beta

        self.Has_RS = not(self.omega is None)

        DFA_Dict =  {'name': 'GX24_%.4f_%.5f'%(self.alpha, self.omega),
            'x_hf': {'alpha': self.alpha, 'beta': self.beta, 'omega': self.omega},
            'x_functionals': {'GGA_X_HJS_PBE': {'alpha': 1.-self.alpha, 'omega': self.omega}},
            'c_functionals': {'GGA_C_PBE': {'alpha': 1.0}}
            }

        npoints = psi4.core.get_option("SCF", "DFT_BLOCK_MAX_POINTS")
        self.DFAU, _ = sf_from_dict(DFA_Dict,npoints,1,False)

            # Make a new VPot for the UKS DFA
        self.VPot = psi4.core.VBase.build(self.VPot.basis(), self.DFAU, "UV")
        self.VPot.initialize()

    def OverridePotential(self, wfn):
        """Use a new wfn object to overwrite the kinetic energy and potential matrices.

        Args:
            wfn (psi4_wavefunction): A valid psi4 wavefunction object (HF or DFT)
        """
        
        self.H_ao = wfn.H().to_array(dense=True)
        self.V_ao = self.H_ao - self.T_ao

    def UpdateJK(self):
        """Force an update to the J and K and K_w matrices
        """
        self.JKHelp.NewJK(self.basis, self.omega)

    
    def UpdateV(self, Delta_ao=0.):
        """Add a matrix to the raw potential. Resets the potential if no arguments.

        Args:
            Delta_ao (ndarray, optional): NBas x NBas array of potential coefficients. Defaults to 0..
        """
        self.V_ao = self.V_ao_Init*1. + Delta_ao
        self.H_ao = self.T_ao + self.V_ao


    def GetHalfERI(self, C1=None, C2=None, D=None,
                   K_w=False):
        """Advanced function: Compute sqrt([pq|rs]) x D_pq

        Args:
            C1 (ndarray, optional): NBas x M1 array of coeffieicents. Defaults to None.
            C2 (ndarray, optional): NBas x M2 array of coeffieicents. Defaults to None.
            D (ndarray, optional): NBas x NBas 1RDM. Must be speficied if C1 not provided.
            K_w (bool, optional): Use the range-separated ERI. Defaults to False.

        Returns:
            _type_: _description_
        """
        # Use RS or full
        if K_w: ERIA = self.ERIA_w
        else: ERIA = self.ERIA

        # Must be specified
        if ERIA is None:
            print("Must set ComputeERIA when Engine is initialized (with RS hybrid if reqd)")
            quit()

        if not(D is None):
            return np.tensordot(ERIA, D, axes=((1,2),(0,1)))

        if not(C1 is None):
            X = np.tensordot(ERIA, C1, axes=((2,),(0,)))
            if not(C2 is None):
                return np.tensordot(X, C2, axes=((1,),(0,)))
            else:
                return X
        else:
            return ERIA

    def GetFJ(self, CI, Pre=1.):
        """Get J matrix - \sum_{mpq} C_{mp} C_{mq} [pq|rs]

        Args:
            CI (ndarray): NBas x M array of coefficienta
            Pre (float, optional): Scale factor. Defaults to 1..

        Returns:
            ndarray: NBas x NBas Fock matrix
        """
        if np.abs(Pre)<zero_round: return 0.
        else: return Pre*self.JKHelp.FJ(CI)

    def GetFK(self, CI, Pre=1.):
        """Get K matrix - \sum_{mpq} C_{mp} C_{mq} [pr|sq]

        Args:
            CI (ndarray): NBas x M array of coefficienta
            Pre (float, optional): Scale factor. Defaults to 1..

        Returns:
            ndarray: NBas x NBas Fock matrix
        """
        if np.abs(Pre)<zero_round: return 0.
        else: return Pre*self.JKHelp.FK(CI)

    def GetFJ_w(self, CI, Pre=1.):
        """Get range-separated J matrix - \sum_{mpq} C_{mp} C_{mq} [pq|rs]_w

        Args:
            CI (ndarray): NBas x M array of coefficienta
            Pre (float, optional): Scale factor. Defaults to 1..

        Returns:
            ndarray: NBas x NBas Fock matrix
        """
        if np.abs(Pre)<zero_round: return 0.
        print("****  Calling RS FJ -- weird! ****")
        quit()

    def GetFK_w(self, CI, Pre=1.):
        """Get range-separated K matrix - \sum_{mpq} C_{mp} C_{mq} [pr|sq]_w

        Args:
            CI (ndarray): NBas x M array of coefficienta
            Pre (float, optional): Scale factor. Defaults to 1..

        Returns:
            ndarray: NBas x NBas Fock matrix
        """
        if np.abs(Pre)<zero_round: return 0.
        else: return Pre*self.JKHelp.FK_w(CI)

    def GetEMaster_Occ(self, f, C=None, Mode='J'):
        """Compute J, K or wK energy using C and f

        Args:
            f (ndarray): M cccupation factors
            C (ndarray, optional): NBas x M array of coefficients. Defaults to internal value.
            Mode (str, optional): 'J', 'K' or 'wK'. Defaults to 'J'.

        Returns:
            float: Energy
        """
        if C is None: C = self.CE

        C = C[:,:len(f)]
        CR = C * f[None,:]

        F = self.JKHelp.FMaster(C, CR, Mode)
        if isinstance(F, np.ndarray):
            return 0.5*np.einsum('pk,pq,qk', C, F, CR)
        else:
            return 0.

    def EJ_Occ(self, f, C=None):
        """Compute J energy using C and f

        Args:
            f (ndarray): M cccupation factors
            C (ndarray, optional): NBas x M array of coefficients. Defaults to internal value.

        Returns:
            float: Energy
        """
        return self.GetEMaster_Occ(f, C, 'J')

    def EK_Occ(self, f, C=None):
        """Compute K energy using C and f

        Args:
            f (ndarray): M cccupation factors
            C (ndarray, optional): NBas x M array of coefficients. Defaults to internal value.

        Returns:
            float: Energy
        """
        return self.GetEMaster_Occ(f, C, 'K')

    def EK_w_Occ(self, f, C=None):
        """Compute range-separated K energy using C and f

        Args:
            f (ndarray): M cccupation factors
            C (ndarray, optional): NBas x M array of coefficients. Defaults to internal value.

        Returns:
            float: Energy
        """
        return self.GetEMaster_Occ(f, C, 'wK')

    # Compute the DFA terms and return energy and V matrix
    def GetDFA(self, Ca=None, Cb=None, 
               Da=None, Db=None,
               Pre=1.,
               BothSpin=False, **kwargs):
        """Calculate the (partial) exchange-correlation energy and Fock matrix(ces)

        Args:
            Ca (ndarray, optional): NBas x M Occupied alpha orbitals. Defaults to internal occupied.
            Cb (ndarray, optional): NBas x M Occupied beta orbitals. Defaults to Ca.
            Da (ndarray, optional): NBas x NBas alpha 1RDM. Overrides Ca if both specified
            Db (ndarray, optional): NBas x NBas beta 1RDM. . Defaults to Da.
            Pre (float, optional): Scalar prefactor. Defaults to 1..
            BothSpin (bool, optional): Return alpha and beta Fock matrices. Defaults to False.

        Returns:
            float, ndarray: Energy and NBas x NBas alpha Fock matric
            float, ndarray, ndarray: If BothSpin is True
        """
        if Ca is None: Ca = self.C[:,:self.NOcc]

        # Terminate if DFA is unspecified
        if self.DFA is None:
            if BothSpin: return 0., 0., 0.
            else: return 0., 0.

        # Terminate if pre-factor is too small
        if np.abs(Pre)<zero_round:
            if BothSpin: return 0., 0., 0.
            else: return 0., 0.

        # Compute the 1RDM
        if Da is None:
            Da = np.dot(Ca, Ca.T)
            # Use Da by default
            if Cb is None: 
                Db = Da
            else:
                Db = np.dot(Cb, Cb.T)
        else:
            if Db is None: Db = Da


        self.DMa.np[:,:] = Da
        self.DMb.np[:,:] = Db
        self.VPot.set_D([self.DMa,self.DMb])
        self.VPot.compute_V([self.VMa,self.VMb])
        ExcDFA = Pre * self.VPot.quadrature_values()["FUNCTIONAL"]
        VxcDFA = Pre * self.VMa.to_array(dense=True)

        if BothSpin:
            return ExcDFA, VxcDFA, Pre * self.VMb.to_array(dense=True)

        return ExcDFA, VxcDFA

    def GetHF(self, **kwargs):
        return self.GetHx(**kwargs)
    def GetEXX(self, **kwargs):
        return self.GetHx(**kwargs)
    def GetHx(self, Ca=None, Cb=None,
              alphaH = 1., alpha = None, beta = None,
              BothSpin=False, **kwargs):
        """Calculate the Hartree-(partial) exchange energy and Fock matrix(ces)

        Args:
            Ca (ndarray, optional): NBas x M Occupied alpha orbitals. Defaults to internal occupied.
            Cb (ndarray, optional): NBas x M Occupied beta orbitals. Defaults to Ca.
            alphaH (float, optional): Scale factor on Hartree. Defaults to 1..
            alpha (float, optional): Scale factor on full exchange. Defaults to internal default.
            beta (float, optional): Scale factor on range-separated exchange. Defaults to internal default.
            BothSpin (bool, optional): Return alpha and beta Fock matrices. Defaults to False.

        Returns:
            float, ndarray: Energy and NBas x NBas alpha Fock matric
            float, ndarray, ndarray: If BothSpin is True
        """
        if alpha is None: alpha = self.alpha
        if beta is None: beta = self.beta

        if Ca is None: Ca = self.C[:,:self.NOcc]

        if Cb is None:
            VHx  = self.GetFJ(Ca, Pre=2.*alphaH)
            VHx += self.GetFK(Ca, Pre=-alpha)
            VHx += self.GetFK_w(Ca, Pre=-beta)

            D = np.dot(Ca, Ca.T)*2.

            if np.abs(alphaH)>zero_round:
                EHx = 0.5 * np.tensordot(D, VHx)
            else:
                EHx = 0.
            if BothSpin: return EHx, VHx, VHx
        else:
            FJ   = self.GetFJ(Ca, Pre=alphaH) + self.GetFJ(Cb, Pre=alphaH)
            FKa  = self.GetFK(Ca, Pre=-alpha)
            FKa += self.GetFK_w(Ca, Pre=-beta)
            FKb  = self.GetFK(Cb, Pre=-alpha)
            FKb += self.GetFK_w(Cb, Pre=-beta)

            VHx = FJ + FKa

            Da = np.dot(Ca, Ca.T)
            Db = np.dot(Cb, Cb.T)

            if np.abs(alphaH)>zero_round:
                EHx = 0.5 * np.tensordot(Da+Db, FJ)
            else:
                EHx = 0.

            if np.abs(alpha)>zero_round or np.abs(beta)>zero_round:
                EHx += 0.5 * ( np.tensordot(Da, FKa) + np.tensordot(Db, FKb) )

            if BothSpin: return EHx, VHx, FJ + FKb

        return EHx, VHx

    def GetExtra(self, C=None, Extra=None, **kwargs):
        """Generate energy and Fock matrix for Extras list

        Args:
            C (ndarray, optional): NBas x M array of coefficients. Defaults to internal.
            Extra (list, optional): Format is [(PreE,PreF,k1,k2,Kind), ...]. Defaults to None.

        Returns:
            float, ndarray: Energy and Fock matrix
        """

        # Extra = [(PreE,PreF,k1,k2,Kind), ...]
        # PreE and PreF are prefactors for E and F
        # k1 and k2 are orbitals
        # Kind is J, K or K_w

        if C is None: C = self.C
        if Extra is None: return 0., 0.

        EExtra, FExtra = 0., 0.
        for (PreE, PreF, k1, k2, Kind) in Extra:
            C1 = C[:,k1]
            C2 = C[:,k2]
            F11 = self.JKHelp.FMaster(C1, C1, Kind)

            EExtra += PreE * (C2).dot(F11).dot(C2)
            FExtra += PreF * F11

        return EExtra, FExtra

    def GetEnergy(self, C=None, f = None,
                  Extra = None, **kwargs):
        """Calculate the total energy Hx+xc+Extras

        Args:
            C (ndarray, optional): NBas x M array of coefficients. Defaults to internal.
            f (_type_, optional): M array of occupationf factors. Defaults to internal.
            Extra (list): See GetExtra. Defaults to None.

        Returns:
            _type_: _description_
        """
        if C is None: C = self.C
        if f is None: f = self.f_Occ

        if np.mean(np.abs(f - np.round(f)))>zero_round:
            print("Warning! Your occupation factors are non-integer - this is not the right routine")

        fa = np.minimum(f, 1.)
        fb = f - fa

        # Handle the 1-RDM
        nf = len(f)
        D = np.einsum('pk,qk,k->pq', C[:,:nf], C[:,:nf], f)

        FTV = self.T_ao + self.V_ao
        ETV = np.vdot(FTV, D)

        # Handle the Hartree and exchange and correlation
        if np.sum(np.abs(fa-fb))<zero_round: # UKS
            Ca = self.C[:,:nf][:,np.abs(fa-1.)<zero_round]

            EHx, VHx = self.GetHx(Ca=Ca)
            ExcDFA, VxcDFA = self.GetDFA(Da = D/2.)
        else: # RKS
            Ca = self.C[:,:nf][:,np.abs(fa-1.)<zero_round]
            Cb = self.C[:,:nf][:,np.abs(fb-1.)<zero_round]

            EHx, VHx = self.GetHx(Ca=Ca, Cb=Cb)

            Da = np.dot(Ca, Ca.T)
            Db = np.dot(Cb, Cb.T)

            ExcDFA, VxcDFA = self.GetDFA(Da = Da, Db = Db)

        E = ETV + EHx + ExcDFA + self.Enn
        F = FTV + VHx + VxcDFA

        if not(Extra is None):
            EExtra, FExtra = self.GetExtra(C, Extra)

            E += EExtra
            F += FExtra

        if self.Report>=3:
            print("ETV = %10.5f, EHx = %10.5f, ExcDFA = %10.5f"\
                  %(ETV, EHx, ExcDFA) )

        return E, F
    
    def GetX_FC(self, X, CR, BothSpin=False, **kwargs):
        """Generic handler to generate E, F x CR or E, Fa x CR, Fb x CR.
        Should not be used.
        """
        if BothSpin:
            E, Fa, Fb = X(BothSpin=True, **kwargs)
            if np.abs(E)>0.:
                return E, Fa.dot(CR), Fb.dot(CR)
            else: return 0, 0, 0
        else:
            E, F = X(BothSpin=False, **kwargs)
            if np.abs(E)>0.:
                return E, F.dot(CR)
            else: return 0, 0

    # Solvers
    def Solve(self, F, **kwargs): # Required by all engines
        """Alias for SolveFock"""
        return self.SolveFock(F, **kwargs)
    def SolveFock(self, F, **kwargs): # Required by all engines
        """Solve the Fock equations, exploiting symmetry if possible.

        Args:
            F (_type_): NBas x NBas Fock matrix

        Returns:
            ndarray, ndarray: NBas array of epsilon and NBas x NBas array of C
        """
        return self.SymHelp.SolveFock(F, **kwargs)

    ###################################################################################
    # Front facing operations required by new solver
    #
    # Return various matrices times CR e.g. np.dot(S, CR) for Get_SC
    ###################################################################################

    # One-body simple operationrs
    def Get_SC(self, CR, **kwargs):  # Required by all engines
        """Calculate S x CR where S is overlap

        Args:
            CR (ndarray): NBas x M array of coefficients

        Returns:
            ndarray: NBas x M array of S x coefficients
        """
        return self.S_ao.dot(CR)
    
    def Get_InvSC(self, CR, **kwargs): # Required by all engines
        """Calculate S-1 x CR where S-1 is inverse overlap

        Args:
            CR (ndarray): NBas x M array of coefficients

        Returns:
            ndarray: NBas x M array of S-1 x coefficients
        """
        return la.solve(self.S_ao, CR)
    
    def Get_TC(self, CR, **kwargs): # Required by all engines
        """Calculate T x CR where T is kinetic energy

        Args:
            CR (ndarray): NBas x M array of coefficients

        Returns:
            ndarray: NBas x M array of S x coefficients
        """
        return self.T_ao.dot(CR)
    
    def Get_VC(self, CR, **kwargs): # Required by all engines
        """Calculate V x CR where V is potential energy

        Args:
            CR (ndarray): NBas x M array of coefficients

        Returns:
            ndarray: NBas x M array of S x coefficients
        """
        return self.V_ao.dot(CR)
    
    def Get_1BC(self, CR, OpName, **kwargs): # Required by all engines (but can always return None)
        """Calculate X x CR where X is a one body perator

        Args:
            CR (ndarray): NBas x M array of coefficients
            OpName (string): Kind (only DIP works).

        Returns:
            ndarray: NBas x M array of X x coefficients
        """
        if OpName.upper()[:3]=="DIP":
            return self.Dip_ao.dot(CR)
        return None
    
    # One-and two-body operationrs
    def GetDFA_FC(self, CR, **kwargs): # Required by all engines
        """Like GetDFA except returns Fa (or Fa, Fb) x CR instead of Fa (or Fa, Fb)

        Args:
            CR (ndarray): NBas x M array of coefficients
        """
        return self.GetX_FC(self.GetDFA, CR, **kwargs)
    
    def GetHx_FC(self, CR, **kwargs): # Required by all engines
        """Like GetHx except returns Fa (or Fa, Fb) x CR instead of Fa (or Fa, Fb)

        Args:
            CR (ndarray): NBas x M array of coefficients
        """
        return self.GetX_FC(self.GetHx, CR, **kwargs)
    
    def GetExtra_FC(self, CR, **kwargs): # Required by all engines
        """Like GetExtra except returns F x CR instead of F

        Args:
            CR (ndarray): NBas x M array of coefficients
        """
        E, F = self.GetExtra(CR, **kwargs)
        return E, F.dot(CR)
    
    ###################################################################################
    # Routines for custom DFAs
    ###################################################################################

    def GetDFTGrid(self, xyz=False, GGA=False, MGGA=False, delta=0.):
        Grid = [None]*self.VPot.nblocks()

        basis = self.wfn.basisset()

        deriv = 0
        if GGA: deriv = 1

        for b in range(self.VPot.nblocks()):
            block = self.VPot.get_block(b)
            NP = block.npoints()

            Grid[b] = {'NP':NP, 'w': block.w().to_array() }

            if xyz:
                x = block.x().to_array()
                y = block.y().to_array()
                z = block.z().to_array()

                Grid[b]['xyz'] = np.vstack([x,y,z]).T

            blockopoints = psi4.core.BlockOPoints\
                ( block.x(), block.y(), block.z(), block.w(),
                  psi4.core.BasisExtents(basis,delta) )

            lpos = np.array(blockopoints.functions_local_to_global())

            funcs = psi4.core.BasisFunctions(basis, NP, basis.nbf())
            funcs.set_deriv(deriv)
            funcs.compute_functions(blockopoints)
            lphi = funcs.basis_values()["PHI"].to_array(dense=True)

            Grid[b]['lpos'] = lpos
            Grid[b]['lphi'] = lphi[:, lpos]

            if GGA:
                Grid[b]['lphi_x'] = funcs.basis_values()["PHI_X"].to_array(dense=True)[:, lpos]
                Grid[b]['lphi_y'] = funcs.basis_values()["PHI_Y"].to_array(dense=True)[:, lpos]
                Grid[b]['lphi_z'] = funcs.basis_values()["PHI_Z"].to_array(dense=True)[:, lpos]

        return Grid
        
    def GetEDFA_ID(self, DFAID, DList):
        """
        Get EDFA for a DFA defined by DFAID and apply it to:
          DList = [(Da1, Db1), (Da2, Db2), ...]
        to return:
          [Exc1, Exc2, ...]

        Notes re DFAID
        - cPBE and LYP should be handled smoothly
        - other DFAs can be dealt with using an internal description (e.g. as dictionary)
        """


        if isinstance(DFAID, str):
            DFAID = DFAID.upper()
            DFA_Dict = { 'name':'Temporary DFA'}
            if DFAID in ('CPBE', 'LYP'):
                DFA_Dict['x_functionals'] = {}
                DFA_Dict['c_functionals'] = {"GGA_C_%s"%(DFAID[-3:]): {}}
            elif DFAID == "PBE":
                DFA_Dict['x_functionals'] = {"GGA_X_PBE": {}}
                DFA_Dict['c_functionals'] = {"GGA_C_PBE": {}}
            elif DFAID == 'SELF':
                DFA_Dict = None
        else:
            DFA_Dict = DFAID

        if not(DFA_Dict is None):
            npoints = psi4.core.get_option("SCF", "DFT_BLOCK_MAX_POINTS")
            DFAU, _ = sf_from_dict(DFA_Dict,npoints,1,False)
            VPotT = psi4.core.VBase.build(self.VPot.basis(), DFAU, "UV")
            VPotT.initialize()
        else:
            VPotT = self.VPot

        Exc = []
        for (Da, Db) in DList:
            self.DMa.np[:,:] = Da
            self.DMb.np[:,:] = Db
            VPotT.set_D([self.DMa,self.DMb])
            VPotT.compute_V([self.VMa,self.VMb])
            Exc += [VPotT.quadrature_values()["FUNCTIONAL"]]

        return Exc


        1


    def GetGGAProps(self, Da, Db):
        xyz, w, (rhoa,rhob), _ = GetDensities(None, D1List=(Da,Db), wfn=self.wfn,
                                              return_w = True, return_xyz = True)
        rho = rhoa + rhob
        rho_m = np.maximum(rho, 1e-18)
        zeta = np.abs(rhoa-rhob)/rho_m

        rs, s, _ = GetGGAProps(self.wfn, Da+Db)

        return {'xyz':xyz, 'w':w, 'rho':rho, 'rs':rs, 's':s, 'zeta':zeta}



    def GetELDAProps(self, f=None, C=None):
        if self.DFA is None: return None
        if f is None: f = self.f_Occ
        if C is None: C = self.C

        C = C[:,:len(f)]

        D = np.einsum('pk,qk,k->pq', C, C, f)
        Da = np.einsum('pk,qk,k->pq', C, C, np.minimum(f,1.))
        fD = np.einsum('pk,qk,k->pq', C, C, f**2)

        xyz, w, (rho,rhoa,frho), _ = GetDensities(None, D1List=(D,Da,fD), wfn=self.wfn,
                                                  return_w = True, return_xyz = True)
        rho_m = np.maximum(rho, 1e-18)
        fbar = frho/np.maximum(rho, rho_m)
        zeta = np.abs(2*rhoa-rho)/rho_m
        rs = 0.62035 * rho_m**(-1/3)

        return {'xyz':xyz, 'w':w, 'rho':rho, 'rs':rs, 'zeta':zeta, 'fbar':fbar}

    def Gradient_ao(self, atom=0, Kind='P'):
        if Kind.upper()[0] in ('V', 'P'):
            X = self.mints.ao_oei_deriv1("POTENTIAL", atom)
        elif Kind.upper()[0] in ('T', 'K'):
            X = self.mints.ao_oei_deriv1("KINETIC", atom)
        elif Kind.upper()[0] in ('S', 'O'):
            X = self.mints.ao_oei_deriv1("OVERLAP", atom)
        else:
            return None

        return [x.to_array(dense=True) for x in X]

    def Gradient_NN(self):
        return self.basis.molecule().nuclear_repulsion_energy_deriv1().to_array(dense=True)

    def GetEGGAProps(self, f=None, C=None):
        if self.DFA is None: return None
        if f is None: f = self.f_Occ
        if C is None: C = self.C

        C = C[:,:len(f)]

        D = np.einsum('pk,qk,k->pq', C, C, f)
        Da = np.einsum('pk,qk,k->pq', C, C, np.minimum(f,1.))
        fD = np.einsum('pk,qk,k->pq', C, C, f**2)

        xyz, w, (rho,rhoa,frho), _ = GetDensities(None, D1List=(D,Da,fD), wfn=self.wfn,
                                                  return_w = True, return_xyz = True)
        rho_m = np.maximum(rho, 1e-18)
        fbar = frho/rho_m
        zeta = np.abs(2*rhoa-rho)/rho_m
        rs, s, _ = GetGGAProps(self.wfn, D)

        return {'xyz':xyz, 'w':w, 'rho':rho, 'rs':rs, 's':s, 'zeta':zeta, 'fbar':fbar}
    

class psi4Density:
    """
    Special routines for dealing with psi4 densities
    
    Not required by standard Broadway but useful for testing
    new ensemble functionals.
    """
    def __init__(self, wfn):
        self.wfn = wfn
        self.VPot = wfn.V_potential()
        self.Grid = [None]*self.VPot.nblocks()
        self.NGrid = 0

        basis = self.wfn.basisset()

        deriv = 1

        for b in range(self.VPot.nblocks()):
            block = self.VPot.get_block(b)
            NP = block.npoints()
            self.NGrid += NP

            self.Grid[b] = {'NP':NP, 'w': block.w().to_array() }

            x = block.x().to_array()
            y = block.y().to_array()
            z = block.z().to_array()

            self.Grid[b]['xyz'] = np.vstack([x,y,z]).T

            blockopoints = psi4.core.BlockOPoints\
                ( block.x(), block.y(), block.z(), block.w(),
                  psi4.core.BasisExtents(basis, 0.) )

            lpos = np.array(blockopoints.functions_local_to_global())

            funcs = psi4.core.BasisFunctions(basis, NP, basis.nbf())
            funcs.set_deriv(deriv)
            funcs.compute_functions(blockopoints)

            self.Grid[b]['lpos'] = lpos

            self.Grid[b]['lphi'] \
                = np.array(funcs.basis_values()["PHI"])[:NP, :lpos.shape[0]]
      
            self.Grid[b]['lphi_x'] \
                = np.array(funcs.basis_values()["PHI_X"])[:NP, :lpos.shape[0]]
            self.Grid[b]['lphi_y'] \
                = np.array(funcs.basis_values()["PHI_Y"])[:NP, :lpos.shape[0]]
            self.Grid[b]['lphi_z'] \
                = np.array(funcs.basis_values()["PHI_Z"])[:NP, :lpos.shape[0]]
            
    def D_to_grid(self, D, b, GGA=True, eta_n = 1e-12):
        GP = self.Grid[b]
        lD = D[(GP['lpos'][:,None], GP['lpos'])]
        lDphi = GP['lphi'].dot(lD)

        rho = np.maximum(np.einsum('xp,xp->x', lDphi, GP['lphi']), eta_n)
        if not(GGA): return rho

        s2 = 0.
        for k, dir in enumerate('xyz'):
            s2 += (2.*np.einsum('xp,xp->x', lDphi, GP['lphi_'+dir])/rho)**2

        s2 *= (0.02612/rho**(2/3))

        return rho, s2

            
    def BPBEX(self, D, Db=None):
        if not(Db is None): 
            D = Da + Db

        else:
            1

    def Phi(self, C):
        phi = np.zeros((self.NGrid,))
        
        # Loop over the blocks
        I0 = 0
        for b, GridProps in enumerate(self.Grid):
            # Obtain block information
            IP = I0 + GridProps['NP']

            GP = self.Grid[b]
            lC = C[GP['lpos']]
            phi[I0:IP] += GP['lphi'].dot(lC)

            I0 = IP
        return phi

    def Density(self, D=None, C=None, f=None,
                return_all=False, eta_n=1e-20):
        if not(f is None) and not(C is None):
            D = np.einsum('pk,qk,k->pq', C[:,:len(f)], C[:,:len(f)], f)

        if D is None: quit()

        rho = np.zeros((self.NGrid,))
        if return_all:
            xyz = np.zeros((self.NGrid,3))
            w = np.zeros((self.NGrid,))

        # Loop over the blocks
        I0 = 0
        for b, GridProps in enumerate(self.Grid):
            # Obtain block information
            IP = I0 + GridProps['NP']

            rho[I0:IP] = self.D_to_grid(D, b, GGA=False, eta_n=eta_n)
            if return_all:
                w[I0:IP] = GridProps['w']
                xyz[I0:IP,:] = GridProps['xyz']
            I0 = IP

        if return_all:
            return rho, xyz, w
        else:
            return rho
        
    def GGADensity(self, D=None, C=None, f=None,
                   return_all=False):
        if not(f is None) and not(C is None):
            D = np.einsum('pk,qk,k->pq', C[:,:len(f)], C[:,:len(f)], f)

        if D is None: quit()

        rho = np.zeros((self.NGrid,))
        s2  = np.zeros((self.NGrid,))
        if return_all:
            xyz = np.zeros((self.NGrid,3))
            w = np.zeros((self.NGrid,))

        # Loop over the blocks
        I0 = 0
        for b, GridProps in enumerate(self.Grid):
            # Obtain block information
            IP = I0 + GridProps['NP']

            rho[I0:IP], s2[I0:IP] = self.D_to_grid(D, b, GGA=True)
            if return_all:
                w[I0:IP] = GridProps['w']
                xyz[I0:IP,:] = GridProps['xyz']
            I0 = IP

        if return_all:
            return rho, s2, xyz, w
        else:
            return rho, s2

if __name__ == "__main__":

    psi4.set_output_file("__Engine.out")

    psi4.set_options({
        'basis' : 'def2-tzvp',
        'reference': 'rhf',
    })

    MolStr = """
C  0.000  0.000  0.000
H  0.000 -0.000  1.111
H  1.087 -0.000 -0.229
"""
    fgs = [2.,2.,2.,2.]
    fts = [2.,2.,2.,1.,1.]

    for DFA in ('pbe',):
        print(DFA)

        if DFA=='pbe': DFADict = TextDFA('PBE0_0.0_0.0')

        print(DFADict)


        psi4.geometry("0 1\n"+MolStr)
        psi4.set_options({
            'reference': 'rhf',
        })
        E0, wfn = psi4.energy("scf", dft_functional=DFADict, return_wfn=True)

        Engine = psi4Engine(wfn, Report=0)
        psi4D = psi4Density(wfn)

        Egs, _ = Engine.GetEnergy(f=fgs)

        D = Engine.D
        rho, s2, xyz, w = psi4D.GGADensity(D, return_all=True)

        Exc, _ = Engine.GetDFA(Da=D/2)

        print("%10.4f %10.4f %10.4f"%(E0, Egs, Exc))
        N = np.dot(w, rho)
        ExLDA = -3/4*(3/np.pi)**(1/3) * np.dot(w, rho**(4/3))
        mu, kappa = 0.21951, 0.804
        Fx = 1. + mu*s2/(1 + mu*s2/kappa)
        ExGGA = -3/4*(3/np.pi)**(1/3) * np.dot(w, rho**(4/3)*Fx)
        print("%10.4f %10.4f %10.4f"%(N, ExLDA, ExGGA))





    quit()
    for DFA in ('scf', 'pbe', 'pbe0', 'wb97x'):
        print(DFA)

        psi4.geometry("0 1\n"+MolStr)
        psi4.set_options({
            'reference': 'rhf',
        })
        E0, wfn = psi4.energy(DFA, return_wfn=True)

        Engine = psi4Engine(wfn, Report=0)
        Egs, _ = Engine.GetEnergy(f=fgs)

        Ets_old = 1e10
        for Single in (False, True):
            for step in range(20):
                Ets, Fts = Engine.GetEnergy(f=fts)

                Extra = [(2., 2., Engine.kh, Engine.kl, 'K')]
                Eex, Fex = Engine.GetExtra(Extra=Extra)
                Ess = Ets + Eex

                if Single:
                    Fts += Fex

                epsilon = Engine.epsilon
                C = Engine.C

                epsilon_, C_ = Engine.SolveFock(Fts, k0=Engine.kl)
                Engine.epsilon[Engine.kl:] = epsilon_
                Engine.C[:,Engine.kl:] = C_

                if (np.abs(Ets-Ets_old)<1e-6) and (step>5): break

                Ets_old = Ets
            print("Excitation en = %10.2f %10.2f"\
                  %(eV*(Ets - Egs), eV*(Ess - Egs)))

        psi4.geometry("0 3\n"+MolStr)
        psi4.set_options({
            'reference': 'rohf' if DFA=='scf' else 'uhf',
        })
        E1 = psi4.energy(DFA)

        print("%10.5f %10.5f %10.5f %10.5f"%(E0, Egs, eV*(E1-E0), eV*(Ets-Egs)))

    print(Engine.SymHelp.s_sorted)
