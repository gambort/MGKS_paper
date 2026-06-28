__version__ = 1.6
__author__ = "Tim Gould"

from Broadway.EDFT import CoreExcitationHelper
from Broadway.LDAFits import *

import numpy as np
import scipy.linalg as la
import numpy.random as ra
import scipy.optimize as opt

eV = 27.211
tau_min = 1e-3

###########################################################
# Routines for safe handling of Fermi statistics

def safeexp(x):
    return np.exp(np.minimum(86,x))

def safelog(x):
    return np.log(np.maximum(x, 1e-36))

def safelog1exp(x, T=43):
    # safe handling of log(1 + exp(x))
    f = x*1.
    f[x<T] = np.log(1 + np.exp(x[x<T]))
    f[x<-T] = np.exp(x[x<-T])
    return f

"""
Evaluate Fermi-Diract thermal occupations given an array of
epsilon values, a float temperature tau and float electron number N

This algorithm seems to be very hard to break - I think I caught
every edge case in it
"""
def ThermalOcc(epsilon, tau, N,
               w_f = 1., # Weights, if needed
               muOnly = False,
               eta = 1e-8,
               eta_eps = 1e-4, NBisect=150):
    # Make sure epsilon is an array of floats
    epsilon = np.array(np.atleast_1d(epsilon), dtype=float)

    # Then sort - we will undo this at the end
    kk = np.argsort(epsilon)
    kk_inv = np.argsort(kk)
    epsilon = epsilon[kk]
    
    if N > 2*len(epsilon): # Return an error if N is incompatible with epsilon
        print("Cannot get %.4f electrons from %d orbitals"\
              %(N, len(epsilon)))
        return None
    elif N==2*len(epsilon): # All occupied
        if muOnly: return epsilon[0]-50*tau
        return 0*epsilon + 2.
    elif N <=0: # No occupied
        if muOnly: return epsilon[-1]+50*tau
        return 0*epsilon
        

    # Make sure tau is always non-zero to avoid order of limits
    tau = max(1e-8, tau)

    def qf(mu):
        return 2./(1 + safeexp((epsilon-mu)/tau))
    
    # (1) First get the unique eigenvalues
    # i) round to the nearest tau/10 or eta_eps (greater)
    rr = max(tau/10, eta_eps)
    eps_u = np.round(epsilon/rr)*rr
    # ii) get unique values
    eps_u = np.sort(np.unique(eps_u))

    # Then get the N between those values
    eps_b = (eps_u[1:]+eps_u[:-1])/2
    N_b = np.array([np.sum(qf(mu)) for mu in eps_b])

    # (2) Then evaluate initial window
    if N<N_b[0]: # Before first value
        window = (np.min(epsilon)-20*tau, eps_b[0])
    elif N>=N_b[-1]: # After second value
        window = (eps_b[-1], np.max(epsilon)+20*tau)
    else: # Some other size
        for k in range(len(N_b)-1):
            if (N>=N_b[k]) and (N<N_b[k+1]):
                window = (eps_b[k], eps_b[k+1])

    # (3) Then use bisection to get mu
    def QErr(mu): return np.sum(w_f*qf(mu))-N
    
    mu = [window[0], (window[0]+window[1])/2, window[1]]
    dN = [QErr(mu_) for mu_ in mu]

    while dN[0]*dN[2]>0.:
        h = mu[2]-mu[0]
        mu[0] -= h/2
        mu[2] += h/2
        dN = [QErr(mu_) for mu_ in mu]    


    # Check if already really good
    if np.min(np.abs(dN))<eta:
        k = np.argmin(np.abs(dN))
        mu0 = mu[k]
    else: # Otherwise bisect
        for step in range(NBisect):
            if dN[0]*dN[1]<=0.: k0, k1 = 0, 1
            else: k0, k1 = 1, 2

            mu0 = (mu[k0]+mu[k1])/2
            dN0 = QErr(mu0)

            if np.abs(dN0)<eta:
                break

            mu = [mu[k0], mu0, mu[k1]]
            dN = [dN[k0], dN0, dN[k1]]

    f = qf(mu0)
    if np.abs(np.sum(w_f*f)-N)<eta:
        if muOnly: return mu0
        return f[kk_inv]
    else:
        print("Left with %.8f electrons not %.8f as desired"\
              %(np.sum(w_f*f), N))
        return None


def Entropy(f):
    fh = f/2.
    return -2.*(fh.dot(safelog(fh)) + (1-fh).dot(safelog(1-fh)))


class ZPropsHelper:
    def __init__(self, epsilon, f0, E0, Low):
        self.epsilon = epsilon

        self.f0 = f0*1.
        self.N0 = np.sum(f0)
        self.Es0 = np.dot(f0, epsilon[:len(f0)])
        self.E0 = E0

        X = np.array(Low)
        self.NLow = X.shape[0]
        # [(D, N-N0,E-E0,Es-Es0,Indx)]
        self.Degen = X[:,0]
        self.dN = X[:,1]
        self.dE = X[:,2]
        self.dEs = X[:,3]

        #self.Singlet = np.min(self.dE)>=-1e-6
        self.Singlet = True

    def GetBounds(self, tau):
        mu0 = ThermalOcc(self.epsilon, tau, self.N0, muOnly=True)
        dmu_d = max(tau, 0.01)
        dmu_u = tau

        while dmu_d<0.1:
            Np = self.Get(mu0+dmu_u, tau, Show=False)[0]
            Nm = self.Get(mu0-dmu_d, tau, Show=False)[0]
            if np.abs(Np-Nm)>0.1: break
            dmu_u *= 1.1
            dmu_d *= 1.1

        return mu0-dmu_d, mu0+dmu_u
    
    def Solve(self, tau):
        mu0, mu1 = self.GetBounds(tau)
        muh = (mu0+mu1)/2.

        mu_ = [mu0, muh, mu1]
        dN_ = [self.Get(mu,tau)[0]-self.N0 for mu in mu_]
        for step in range(80):
            #print(mu_, dN_)
            if dN_[0]*dN_[1]<0.: k0, k1 = 0, 1
            else: k0, k1 = 1, 2
            muh = (mu_[k0] + mu_[k1])/2
            dNh = self.Get(muh,tau)[0]-self.N0

            mu_ = [mu_[k0], muh, mu_[k1]]
            dN_ = [dN_[k0], dNh, dN_[k1]]

            if (np.abs(dNh)<1e-10) or (mu_[2]-mu_[0])<1e-8:
                break

        return muh

    def Get(self, mu, tau, Show=False):
        LZ0 = -(self.Es0-mu*self.N0)/tau
        LZM = 2.*np.sum(safelog1exp(-(self.epsilon-mu)/tau))

        exp_zero = max(LZM-LZ0, 0.)

        ZM = safeexp(LZM-LZ0 - exp_zero)
        fM = 2./(1 + safeexp((self.epsilon-mu)/tau))
        NM = np.sum(fM)
        EM = np.dot(fM, self.epsilon) - self.Es0

        LW  = -(self.dE -mu*self.dN)/tau
        LWs = -(self.dEs-mu*self.dN)/tau
        Z  = self.Degen*np.exp(LW  - exp_zero)
        Zs = self.Degen*np.exp(LWs - exp_zero)

        ZTot = ZM + np.sum(Z-Zs)
        WM = ZM/ZTot
        W  = Z /ZTot
        Ws = Zs/ZTot

        DW = np.sum(W-Ws)
        DN = np.dot(W-Ws, self.N0+self.dN)
        DE = np.sum(W*self.dE-Ws*self.dEs)

        DE_Full = DE + DW*self.E0

        NTot = WM*NM + DN
        ETot = self.E0 + WM*EM + DE

        if Show:
            print("%7.4f | %6.4f %7.4f %8.4f %6.4f %7.4f %8.4f"\
                  %(mu, WM, NM, EM, DW, DN/DW, DE/DW))

        Props = { 'WM': WM, 'W': W, 'Ws': Ws,
                  'fM': fM, 'DN': DN, 'DE': DE_Full,}

        return NTot, ETot, Props


###########################################################
# Mermin EDFT helper

class MerminExcitationHelper(CoreExcitationHelper):
    # Quick Mermin calculaion
    def QuickMerminEnergyFock(self, f, C=None):
        if C is None: C = self.CE

        if len(f)<C.shape[1]:
            f_ = f*1.
            f = np.zeros((C.shape[1],))
            f[:len(f_)] = np.maximum(f_, 0.)

        C_tau = C*np.sqrt(f/2.)[None,:] # effective C for spin
        CL_tau = C*f[None,:] # Left hand for energies

        # Get the Fock (*C) and Hamiltonian
        FTVC = self.Engine.Get_TC(C) + self.Engine.Get_VC(C)
        ETV = np.einsum('pk,pk', CL_tau, FTVC)

        EHx, FHxC = self.Engine.GetHx_FC(C, Ca=C_tau)
        Exc, FxcC = self.Engine.GetDFA_FC(C, Ca=C_tau)

        E = ETV + EHx + Exc + self.Engine.Enn
        FC = FTVC + FHxC + FxcC

        return E, FC

    # Full Mermin calculaion
    def FullMerminEnergyFock(self, f, tau, C=None):
        if C is None: C = self.CE

        if len(f)<C.shape[1]:
            f_ = f*1.
            f = np.zeros((C.shape[1],))
            f[:len(f_)] = np.maximum(f_, 0.)

        C_tau = C*np.sqrt(f/2.)[None,:] # effective C for spin
        CL_tau = C*f[None,:] # Left hand for energies

        # Get the Fock (*C) and Hamiltonian
        FTVC = self.Engine.Get_TC(C) + self.Engine.Get_VC(C)
        ETV = np.einsum('pk,pk', CL_tau, FTVC)

        EHx, FHxC = self.Engine.GetHx_FC(C, Ca=C_tau)
        Exc, FxcC = self.Engine.GetDFA_FC(C, Ca=C_tau)

        E = ETV + EHx + Exc + self.Engine.Enn
        FC = FTVC + FHxC + FxcC

        # Get the in and out densities projected onto self.C0 basis
        SC0 = self.Engine.Get_SC(self.C0)
        X = (C_tau.T).dot(SC0)
        D_in = (X.T).dot(X)
        # solve the Fock equation to get new fo and Co
        epsilono, Uo = la.eigh((C.T).dot(FC))
        fo = ThermalOcc(epsilono, tau, np.sum(f))
        Co = C.dot(Uo)
        # Project onto self.C0 basis
        D_out = (SC0.T).dot(np.einsum('pk,qk,k->pq', Co, Co, fo)).dot(SC0)

        return E, FC, D_in, D_out
    
    def eps_C_to_D(self, tau, N0, epsilon=None, C=None):
        f = ThermalOcc(epsilon, tau, N0)
        return np.einsum('pk,qk,k->pq', C, C, f)

    
    def D_to_D(self, tau, N0, D,
                 f_cut = 1e-10):
        Sym0 = self.Engine.Sym_k*1
        AllSyms = set(self.Engine.Sym_k)
        C0 = self.C0
        SC0 = self.Engine.Get_SC(C0)

        # Form the density matric using C0 basis with symmetries
        D0 = np.zeros((C0.shape[1], C0.shape[1]))
        f0 = np.zeros((C0.shape[1], ))
        for S in AllSyms:
            kk_S = Sym0==S
            D0_S = (SC0[:,kk_S].T).dot(D).dot(SC0[:,kk_S])
            f0_S, U0 = la.eigh(-D0_S + 1e-12*np.eye(D0_S.shape[0]))
            f0_S = np.einsum('pk,qk,pq->k', U0, U0, D0_S)

            D0[np.ix_(kk_S,kk_S)] += D0_S # Note, this is indexing trickery
            f0[kk_S] = f0_S

            kk_cut = f0_S>f_cut
            C0_tau_S = C0[:,kk_S].dot(U0[:,kk_cut])*np.sqrt(f0_S[kk_cut]/2.0)[None,:]

            if S==0: C0_tau = C0_tau_S
            else: C0_tau = np.hstack((C0_tau, C0_tau_S))

        # Get the Fock (*C) and Hamiltonian
        FTVC = self.Engine.Get_TC(C0) + self.Engine.Get_VC(C0)

        ETV = np.vdot(D0, (C0.T).dot(FTVC))

        EHx, FHxC = self.Engine.GetHx_FC(C0, Ca=C0_tau)
        Exc, FxcC = self.Engine.GetDFA_FC(C0, Ca=C0_tau)

        #print("Energies: ETV = %.6f EHx = %.6f Exc = %.6f"%(ETV, EHx, Exc))

        E = ETV + EHx + Exc + self.Engine.Enn
        F_C = (C0.T).dot(FTVC + FHxC + FxcC) # F in the C basis

        if len(AllSyms)>1:
            # Ensure the symmetry is preserved
            epsilon = np.zeros((C0.shape[1],))
            C = np.zeros((C0.shape[0], C0.shape[1]))
            for S in AllSyms:
                kk_S = Sym0==S
                eps_S, U_S = la.eigh(F_C[np.ix_(kk_S,kk_S)])
                epsilon[kk_S] = eps_S
                C[:,kk_S] = C0[:,kk_S].dot(U_S)
        else:
            epsilon, U = la.eigh(F_C)
            C = C0.dot(U)

        f = ThermalOcc(epsilon, tau, N0)
        D = np.einsum('pk,qk,k->pq', C, C, f)

        #print(f0)
        S = Entropy(f0)

        Props = {
            'E': E, 'FE': E - tau*S, 'tauS': tau*S,
            'epsilon': epsilon*1., 'f': f*1., 'C': C*1.,
        }


        return D, Props


    def eps_to_Groups(self, epsilon, SymCut=1e-3):
        Ne = len(epsilon)
        SMax = 5
        k = 0
        Groups = {}
        while k<Ne:
            e0 = epsilon[k]
            Groups[k] = [k,]
            for k1 in range(k+1,min(k+SMax,Ne)):
                if np.abs(epsilon[k1]-e0)<SymCut:
                    Groups[k] += [k1]
            k = np.max(Groups[k])+1
        return Groups
    
    def Groups_to_eps(self, epsilon, Groups):
        for k in Groups:
            kk = Groups[k]
            epsilon[kk] = np.mean(epsilon[kk])
        return epsilon

    def Solver(self, **kwargs):
        return self.SolverDIIS(**kwargs)


    # Generic Mermin DIIS routine
    def SolverDIIS(self, Plan=None, tau=0, N0=None,
                NDIIS = 4, DIISEvery = 4, Mix_D = 0.3,
                MaxIter = 200, ShowIter = 5,
                ECut = 1e-5, FECut = 1e-8,
                Reset = True,
               **kwargs):
        tau = max(tau, tau_min)
        if N0 is None and Plan is not None:
            f0 = Plan['1RDM'].np
            N0 = np.sum(f0)
        elif N0 is None:
            N0 = self.Na + self.Nb

        if Reset:
            self.CE = 1.*self.C0
            self.epsilonE = 1.*self.epsilon0

        NBas = self.C0.shape[0]
        D_In  = np.zeros((NDIIS,NBas,NBas))
        D_Out = np.zeros((NDIIS,NBas,NBas))

        def DIISSolve(eta = 1e-6):
            DD = D_Out - D_In

            A = np.ones((NDIIS+1, NDIIS+1))
            A[:NDIIS,:NDIIS] = np.tensordot(DD, DD, axes=((1,2),(1,2))) + eta*np.eye(NDIIS)
            A[-1,-1] = 0.
            c = np.zeros((NDIIS+1))
            c[-1] = 1.
            b = la.solve(A, c)[:NDIIS]

            return b, np.tensordot(b, D_In, axes=((0,),(0,)))

        def QShow(step, tau, Props, Props_Old=None):
            E = Props['E']
            FE = Props['FE']
            k0 = max(self.kh-3,0)
            k1 = k0 + 6
            fStr = " ".join("%.3f"%(x) for x in Props['f'][k0:k1])
            if Props_Old is None:
                print("%3d %6.3f %10.5f %10.5f %s"%(step, tau, E, FE, fStr))
            else:
                DE = E - Props_Old['E']
                DFE = FE - Props_Old['FE']
                print("%3d %6.3f %10.2e %10.2e %s"%(step, tau, DE, DFE, fStr))

        D = self.eps_C_to_D(tau, N0, self.epsilonE, self.CE)
        for step in range(MaxIter):
            Dp, Props = self.D_to_D(tau, N0, D)
            D_In[step%NDIIS,:,:] = D
            D_Out[step%NDIIS,:,:] = Dp

            if step>=NDIIS and (step%DIISEvery==0):
                b, D = DIISSolve()
                D = (1-Mix_D)*D + Mix_D*Dp
            else:
                D = (1-Mix_D)*D + Mix_D*Dp

            if step==0:
                Breaking = False
            else:
                Breaking = (np.abs(Props['E']-Props_Old['E'])<ECut*(1-Mix_D)) \
                            and (np.abs(Props['FE']-Props_Old['FE'])<FECut*(1-Mix_D))
            
            E = Props['E']
            if step%ShowIter==0 and (step>0):
                QShow(step, tau, Props, Props_Old)

            if step==0 or Breaking:
                QShow(step,tau, Props)

            Props_Old = { x: Props[x]*1. for x in Props }


            if Breaking: 
                self.Converged = True
                break

            

        self.epsilonE = Props['epsilon']*1.
        self.CE = Props['C']*1.
        self.Lastf = Props['f']*1.
        self.Lasttau = tau

        # Store properties of this run
        if not(Plan is None):
            self.LastPlan = Plan
            Occ = Plan['1RDM']
            self.LastD = Occ.D(self.CE)
        else:
            self.LastD = D*1.

    
        if self.Converged:
            return E
        else:
            if self.Report>0: print("Warning! Failed to converge - returning last calc")
            return E


    # Generic Mermin fixed point routine
    def SolverFixedPoint(self, Plan, tau=0, N0=None,
                MaxIter = 200, ShowIter = 5,
                ECut = 1e-7, fCut = 1e-4,
                Mix_F = 0.3, Mix_eps = 0.5,
                ForceSym = False, SymCut = 1e-3,
                Reset = True,
               **kwargs):
        tau = max(tau, tau_min)

        if Reset:
            self.CE = 1.*self.C0
            self.epsilonE = 1.*self.epsilon0

        if ForceSym:
            Groups = self.eps_to_Groups(self.epsilon0, SymCut=SymCut)

        if N0 is None:
            f0 = Plan['1RDM'].np
            N0 = np.sum(f0)

        E_Old = 1e4
        epsilon_Old = self.epsilonE
        F_Old = None
        epsilon = epsilon_Old*1.
        C = self.CE*1.


        def QShow(tau, E, f):
            k0 = max(self.kh-3,0)
            k1 = k0 + 6
            fStr = " ".join("%.3f"%(x) for x in f[k0:k1])
            print("%3d %6.3f %10.5f %s"%(step, tau, E, fStr))

        f = ThermalOcc(epsilon, tau, N0)

        k_DIIS = 0

        for step in range(MaxIter):
            E, FC = self.QuickMerminEnergyFock(f, C)

            if (self.Report>1) and (step%ShowIter)==0:
                QShow(tau, E, f)

            # The f cutoff is a bit challenging if low-occ oscillate
            if (step>0) and (np.abs(E-E_Old)<ECut) \
                and (np.max(np.abs(f*(f-f_Old)))<fCut):
                if self.Report>0: QShow(tau, E, f)

                self.Converged = True
                break

            if not(F_Old is None):
                FC = (1-Mix_F)*FC + Mix_F*F_Old.dot(C)
            F_Old = FC.dot(self.Engine.Get_SC(C).T)

            epsilon, U = la.eigh((C.T).dot(FC))
            C = C.dot(U)

            if ForceSym:
                epsilon = self.Groups_to_eps(epsilon, Groups)

            epsilon = (1-Mix_eps)*epsilon + Mix_eps*epsilon_Old
            epsilon_Old = epsilon*1.
            E_Old = E*1.
            f_Old = f*1.

            f = ThermalOcc(epsilon, tau, N0)

        self.epsilonE = epsilon*1.
        self.CE = C*1.
        self.Lasttau = tau

        # Store properties of this run
        self.LastPlan = Plan
        self.Lastf = f

        # Get epsilonE and CE by rediagonalising down blocks

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
        
    def PostMermin(self, tau=None, ListOfPlans=None, GSPlan=None, N0=None):
        """Apply the post Mermin CDFA correction

        Args:
            tau (float, optional): Temperature in Ha. Defaults to last value.
            ListOfPlans (_type_, optional): _description_. Defaults to None.
            GSPlan (_type_, optional): _description_. Defaults to None.
            N0 (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if tau is None: tau = self.Lasttau
        tau = max(tau, tau_min)

        if not(N0 is None):
            1
        elif not(GSPlan is None):
            N0 = np.sum(GSPlan['1RDM'].np)
        else:
            N0 = self.Na + self.Nb

        if ListOfPlans is None:
            # Defaults to first triplet and singlet
            # Note - this is bad to use for atoms
            ListOfPlans = [
                (1, self.SolveGS(PlanOnly=True)),
                (3, self.SolveTS(PlanOnly=True)),
                (1, self.SolveSX(PlanOnly=True)),
            ]

        def QuickVals(Plan):
            f = Plan['1RDM'].np
            N = np.sum(f)
            Es = np.dot(self.epsilonE[:len(f)], f)
            E = self.SolverFrozen(Plan)
            return N, E, Es

        #N0, E0, Es0 = QuickVals(GSPlan)

        Low = np.zeros((len(ListOfPlans),4))
        for Indx,(D, Plan) in enumerate(ListOfPlans):
            N, E, Es = QuickVals(Plan)
            Low[Indx] = [D,N,E,Es]
                
        Indx0 = np.argmin(Low[:,2] + 1e-4*Low[:,3] + 1e4*np.abs(Low[:,1]-N0))
        N0, E0, Es0 = tuple(Low[Indx0,1:])
        Low[:,1]-=N0
        Low[:,2]-=E0
        Low[:,3]-=Es0
        f0 = ListOfPlans[Indx0][1]['1RDM'].np

        if self.Report>=3:
            for Indx,(D, Plan) in enumerate(ListOfPlans):
                N, E, Es = QuickVals(Plan)
                print("|%2d_s> -> dN = %2d D = %d, E = %8.4f Es = %8.4f"\
                      %(Indx, N-N0, D, E-E0, Es-Es0))

                
        ZPH = ZPropsHelper(self.epsilonE, f0, E0, Low)
                
        def QProps(mu):
            return ZPH.Get(mu, tau)

        mu = ZPH.Solve(tau)
        NTot, ETot, Props = QProps(mu)

        if np.abs(NTot-N0)>1e-6:
            return E0, Props

        Props['f'] = Props['WM']*Props['fM']
        for Indx,(D, Plan) in enumerate(ListOfPlans):
            dW = Props['W'][Indx] - Props['Ws'][Indx]
            f_ = Plan['1RDM'].np
            Props['f'][:len(f_)] += dW*f_

        self.LastMerminProps = Props
        if self.Report>0:
            print("%8.4f %6.4f [ %s ]"%(NTot, 1.-np.sum(Props['W']),
                                    " ".join("%6.4f"%(x) for x in Props['W'])))
        return ETot, Props



