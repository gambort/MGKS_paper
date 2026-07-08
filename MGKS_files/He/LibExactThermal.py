import psi4

import numpy as np
import scipy.linalg as la
import scipy.optimize as opt
import numpy.random as ra

eV = 27.211399

def ReadCFG(FName):
    Defaults = {
        'DFA': 'svwn',
        'kbT0': 1e-4,
        'kbT1': 0.5,
        'kbTN': 26,
        'kbTGrid': 'linear',
        'RBox': 5.0,
        'Basis': 'aug-cc-pvtz',
    }

    F = open(FName)
    for L in F:
        if not('=' in L): continue
        X = L.split('=')
        Key = X[0]
        try:
            Val = float(X[1])
        except:
            Val = X[1].strip()

        for DKey in Defaults:
            if Key.lower()==DKey.lower():
                Defaults[DKey] = Val
    F.close()

    if Defaults['kbTGrid'][:3].lower()=='lin':
        Defaults['kbT'] = np.linspace(Defaults['kbT0'], Defaults['kbT1'], 
                                      int(Defaults['kbTN']))
    else:
        l0 = np.log10(Defaults['kbT0'])
        l1 = np.log10(Defaults['kbT0'])
        Defaults['kbT'] = np.logspace(l0, l1, 
                                      int(Defaults['kbTN']))


    return Defaults

def NiceOcc(f):
    return " ".join("%6.4f"%(x) for x in f[f>1e-5])

def NiceEps(f):
    return " ".join(["%6.3f"%(x) for x in f])

def safelog(x, xmin=1e-8):
    return np.log(np.maximum(x, xmin))


def GetOcc(kbT, epsilon, eta=1e-5, N0=2., deps=False,
           eps_degen=0):
    if eps_degen>0.:
        epsilon = np.round(epsilon/eps_degen)*eps_degen

    kl = min(int(np.ceil(N0/2))+2, len(epsilon)-1)
    mu0 = epsilon[0]-20*kbT
    mu1 = epsilon[kl]+20*kbT

    
    def qf(mu):
        x = np.minimum((epsilon-mu)/kbT,86.)
        return 2./(1. + np.exp(x))
    def Err(mu): 
        return np.sum(qf(mu))-N0
    
    mu = [mu0, 0, mu1]
    dN = [Err(mu0), 0, Err(mu1)]

    while dN[0]*dN[1]>0.:
        dmu = mu[2]-mu[0]
        mu[0] = mu[0] - dmu/2
        mu[2] = mu[2] + dmu/2
        dN = [Err(mu0), 0, Err(mu1)]


    for iter in range(30):
        mu_ = (mu[0]+mu[2])/2.
        dN_ = Err(mu_)
        mu[1] = mu_
        dN[1] = dN_

        #print(NiceEps(mu), NiceEps(dN))

        if (mu[2]-mu[1])<1e-8 or np.min(np.abs(dN))<1e-12: break

        if (dN[0]*dN[1])<0.: k0, k1 = 0,1
        else: k0, k1 = 1,2

        mu = [mu[k0], 0, mu[k1]]
        dN = [dN[k0], 0, dN[k1]]

    k = np.argmin(np.abs(dN))
    mu0 = mu[k]
    f = qf(mu0)

    if np.abs(dN[k])>1e-5:
        print(np.array(mu))
        print(np.array(dN))

    if deps: return epsilon-mu0

    return f

def GetDegen(epsilon, deps=1e-5):
    Exclude = {}
    for k in range(len(epsilon)-1):
        if (epsilon[k+1]-epsilon[k])<deps:
            if not(k in Exclude): Exclude[k] = []
            Exclude[k] += [k,k+1]
            Exclude[k+1] = [k,k+1]
    
    Unique = []
    for k in range(len(epsilon)-1):
        if not(k in Exclude): Unique += [k,]

    Degen = {}
    for k in Exclude:
        X = set(Exclude[k])
        for iter in range(8):
            for kp in X:
                X = X.union(Exclude[kp])
        Degen[np.min(list(X))] = list(X)

    return Unique, Degen


def Clusters(f, df=0.02):
    Omega = f - np.hstack((f[1:],0.))

    Omega_N = {}
    Map_N = {}

    # First full the clusters
    Nf = int(np.ceil(2/df))+2
    ft = 2 + df/2.
    for _ in range(Nf):
        fb = ft - df
        kk = np.argwhere((f>=fb) & (f<ft)).reshape((-1,))
        if len(kk)>0:
            B = np.max(kk)+1
            Map_N[B] = kk

        ft = fb

    # Then compute Omega effective
    for B in Map_N:
        kk = Map_N[B]
        Omega_N[B] = np.sum(Omega[kk])

    return Omega, Omega_N, Map_N


def DegenTrim(E, D, round=0.001):
    N = min(D.shape[0], len(E))
    E = E[:N]
    EC = E[-1]-round
    NRoots = len(E[E<EC])
    print("Trimmed from %d to %d"%(N, NRoots))
    return E[:NRoots], D[:NRoots,:,:]

class FCIHelper:
    def __init__(self, Basis, DV, NRoots, Force=False, Kind='H2',
                 Delta3=0.0, # Correction for 3el system (for -> CBS/FCI)
                 HFOnly = False):
        self.Basis = Basis
        self.HFOnly = HFOnly
        if self.HFOnly:
            self.CacheFileFCI = "Data/__%s_HF_%s.npz"%(Kind, Basis)
        else:
            self.CacheFileFCI = "Data/__%s_FCI_%s.npz"%(Kind, Basis)

        if Kind.upper()=='H2': 
            self.Kind = 'H2'
            self.GeomStr = "H\nH 1 0.74\n"
            self.N0 = 2
        elif Kind.upper()=='H':
            self.Kind = 'H'
            self.GeomStr = 'H\n'
            self.N0 = 1
        elif Kind.upper()=='HE':
            self.Kind = 'He'
            self.GeomStr = 'He\n'
            self.N0 = 2
        else:
            print("Kind = %s not recognised"%(Kind))
            quit()

        try:
            X = np.load(self.CacheFileFCI, allow_pickle=True)
            self.Data = X['Data'][()]
        except:
            Force = True
        
        self.DV = DV
        self.NRoots = NRoots

        if Force:
            self.Data = {}
        else:
            del self.Data[1][2]

        if str(Delta3).lower()=='auto':
            self.Delta3 = {'aug-cc-pvtz':-0.5/eV, 'def2-qzvppd':-0.1/eV,
                        'def2-qzvppd-decon':-0.1/eV}[Basis]
            print("Basis = %s, Delta3 = %.2f eV"%(Basis, self.Delta3*eV))
        else:
            self.Delta3 = Delta3
            
        for N, D in [(1,2),(2,1),(2,3),(3,2),(3,4),(4,1),(4,3)]:
            self.Compute(N, D)

        self.Patch()

    def Patch(self):
        for N in self.Data:
            if N<3: continue
            
            for D in self.Data[N]:
                self.Data[N][D]['E_Box'] += self.Delta3


    def Compute(self, N, D):
        if isinstance(self.NRoots, dict):
            if (N,D) in self.NRoots:
                NRoots = self.NRoots[(N,D)]
            else:
                return
        else:
            NRoots = self.NRoots
            if NRoots<1: return

        if not(N in self.Data): self.Data[N] = {}

        # Leave if already got N, D
        if D in self.Data[N]: return

        print("Computing FCI for %d %d for %d roots"%(N,D, NRoots))

        if self.Kind=='H':
            ExcitationLevel = N
        elif N==3:
            ExcitationLevel = 2
        elif N>=4:
            ExcitationLevel = 1
        else:
            ExcitationLevel = 2

        if self.HFOnly: ExcitationLevel = 1

        DiagMethod = {'H2':'sem', 'H':'RSP', 'He':'RSP'}[self.Kind]
        #if N>=3: DiagMethod = 'DAVIDSON'

        psi4.set_options({
            'basis': self.Basis,
            'reference': 'rohf',
            'opdm': True,
            'diag_method': DiagMethod,
            'ex_level': ExcitationLevel,
            'ci_maxiter': 100,
        })
        #if N==3: psi4.set_options({'diag_method':'sem',})

        if N==1:
            # 1 electron = trivial
            mol = psi4.geometry('%s 2\n%s\nsymmetry c1'%(self.N0-1, self.GeomStr))
            _, wfn_CI = psi4.energy('scf', return_wfn = True)
            self.S = wfn_CI.S().np

            NRoots = self.S.shape[0]

            E_Box, C = la.eigh(wfn_CI.H().np + self.DV, b = wfn_CI.S().np)
            E_Box += mol.nuclear_repulsion_energy()

            D_Box = np.zeros((NRoots,C.shape[0],C.shape[0]))
            for k in range(NRoots):
                D_Box[k,:,:] = np.outer(C[:,k], C[:,k])

            #E_Box, D_Box = DegenTrim(E_Box, D_Box)

            self.Data[1][2] = { 'E_Box': E_Box, 'D_Box': D_Box, }
            np.savez(self.CacheFileFCI, Data=self.Data)


            psi4.core.clean()
            return
        

        if N==3:
            psi4.set_output_file("__He_%d_%d.out"%(N,D))
            psi4.geometry('0 1\n%s\nsymmetry c1'%(self.GeomStr))
            _, wfn_2 = psi4.energy('scf', return_wfn = True)
            C_2 = wfn_2.Ca().to_array()
            F_2 = wfn_2.Fa().to_array()
            eps_2 = wfn_2.epsilon_a().to_array()

            print(eps_2[:10])
            

            # Swap 2s,3s with 2p, 3p
            c_2p = C_2[:,1:4]*1.
            c_2s = C_2[:,4]*1.
            c_3p = C_2[:,5:8]*1.
            c_3s = C_2[:,8]*1.
            C_2[:,1] = c_2s
            C_2[:,2] = c_3s
            C_2[:,3:6] = c_2p
            C_2[:,6:9] = c_3p

            print((C_2.T).dot(F_2).dot(C_2)[:8,:8])


        # Solve for 2 and 3
        psi4.geometry('%d %d\n%s\nsymmetry c1'%(self.N0-N, D, self.GeomStr))

        psi4.set_options({ 'num_roots': NRoots, 'S': D, 'calc_s_squared': True,})
        _, wfn_HF = psi4.energy('scf', return_wfn = True)
        wfn_HF.H().np[:,:] += self.DV

        if N==3:
            wfn_HF.Ca().np[:,:] = C_2
            wfn_HF.Cb().np[:,:] = C_2


        _, wfn_CI = psi4.energy('detci', return_wfn = True, ref_wfn = wfn_HF)
        self.S = wfn_CI.S().np


        v = wfn_CI.variables()
        C = wfn_CI.Ca().np

        E_Box = np.zeros((NRoots,))
        D_Box = np.zeros((NRoots,C.shape[0],C.shape[0]))
        for k in range(NRoots):
            E_Box[k] = v['CI ROOT %d TOTAL ENERGY'%(k)]
            D_Box[k,:,:] = C.dot(wfn_CI.get_opdm(k,k,'SUM',True).np).dot(C.T)

        E_Box, D_Box = DegenTrim(E_Box, D_Box)

        self.Data[N][D] = { 'E_Box': E_Box, 'D_Box': D_Box, }

        np.savez(self.CacheFileFCI, Data=self.Data)

        psi4.core.clean()

    def FixSinglets(self):
        E1 = self.Data[2][1]['E_Box']
        E3 = np.unique(self.Data[2][3]['E_Box'])
        kk = []
        for k, e1 in enumerate(E1):
            if np.min(np.abs(e1-E3))>1e-5:
                kk += [k,]

        kk = np.array(kk)
        print("From %3d to %3d"%(len(E1), len(kk)))
        self.Data[2][1]['E_Box'] = self.Data[2][1]['E_Box'][kk]
        self.Data[2][1]['D_Box'] = self.Data[2][1]['D_Box'][kk,:,:]

        np.savez(self.CacheFileFCI, Data=self.Data)
        


    def ComputeMu(self):
        if self.N0==2:
            if 3 in self.Data:
                return -(self.Data[1][2]['E_Box'][0] - self.Data[3][2]['E_Box'][0])/2
            else:
                return 0.
        else:
            return -(0. - self.Data[2][3]['E_Box'][0])/2
        
    def Report(self, Raw=True):
        def TArr(X): return " ".join(["%6.2f"%(x) for x in X])
        if self.N0==2: E0 = self.Data[2][1]['E_Box'][0]
        else: E0 = self.Data[1][2]['E_Box'][0]
        if Raw:
            mu = 0.
        else:
            mu = self.ComputeMu()
        for N in self.Data:
            for D in self.Data[N]:
                DE = self.Data[N][D]['E_Box'] - E0 - mu*(N-self.N0)
                print("%d with %d - %s ... %s"%(N, D, TArr(eV*DE[:5]), TArr(eV*DE[-5:])))

    def Natural(self, C=None):
        for N in self.Data:
            for D in self.Data[N]:
                print("# electrons = %d, Mult = %d"%(N, D))

                D = self.Data[N][D]['D_Box']
                for k in range(D.shape[0]):
                    if C is None:
                        f, _ = la.eigh(-self.S.dot(D[k,:,:]).dot(self.S), b=self.S)
                        f = -f
                    else:
                        SC = self.S.dot(C)
                        f = np.einsum('pk,pq,qk->k', SC, D[k,:,:], SC)
                    print(f)

    def Flatten(self, E_Dict=None, NList=None, Weights=False):
        if E_Dict is None:
            E_Dict = {}
            for N in self.Data:
                E_Dict[N] = {}
                for D in self.Data[N]:
                    E_Dict[N][D] = self.Data[N][D]['E_Box']

        if NList is None: NList = list(E_Dict)

        # Add in the zero state
        D_All = [1]
        N_All = [0]
        E_All = [0.]

        for N in NList:
            for D in self.Data[N]:
                EE = E_Dict[N][D]
                D_All += [D]*len(EE)
                N_All += [N]*len(EE)
                E_All += list(EE)

        if Weights:
            E_All = np.array(E_All)
            E_All[0] = 1. - np.sum(E_All)
            return E_All


        return np.array(D_All), np.array(N_All), np.array(E_All)
    
    def Unflatten(self, V, D_All, N_All):
        VD = {}
        for N in set(N_All):
            if N==0: continue
            VD[N] = {}
            for D in set(D_All):
                vv = V[(N_All==N) & (D_All==D)]
                if len(vv)>0: VD[N][D] = vv

        return VD


    def Solve(self, kbT, Canonical=False, E_Dict=None):
        if kbT<1e-5 or Canonical:
            D_All, N_All, E_All = self.Flatten(E_Dict=E_Dict, NList = (int(self.N0),))
        else:
            D_All, N_All, E_All = self.Flatten(E_Dict=E_Dict)
        
        def QN(mu, WOnly = False):
            x = (E_All - mu*N_All)
            x -= x.min()
            W = D_All*np.exp(-x/kbT)
            if WOnly: return W/np.sum(W)
            return 100000*(np.dot(W, N_All)/np.sum(W) - self.N0)**2
        
        mu0 = self.ComputeMu()

        res = opt.minimize_scalar(QN)
        mu = res.x

        W = QN(mu, WOnly = True)
        self.mu_Last = mu

        # Renormalize
        W *= self.N0/np.dot(W, N_All)

        E_T = np.dot(W, E_All)
        W_All = self.Unflatten(W, D_All, N_All)

        D_T = 0.
        for N in W_All:
            for D in W_All[N]:
                D_T += np.einsum('k,kpq->pq', W_All[N][D], self.Data[N][D]['D_Box'])

        return W_All, E_T, D_T

    def EstimateErrors(self, T0=0.01, T1=10):
        kbT_all = np.linspace(max(T0,0.01), T1, 20)/eV

        def GetNS(X, eps=0.0016):
            Q = np.abs(X-X[-1])
            for k in range(1, len(Q)):
                if Q[k]<eps: break
            return k+1
        
        print("%5s %7s %5s %5s %5s [eV]" % ('kbT', 'F-F0', '0.01', '0.1', '5%'))

        E_0 = None
        for kbT in kbT_all:
            W_T, E_T, D_T = self.Solve(kbT)
            D_All, N_All, E_All = self.Flatten()
            W_All = self.Flatten(W_T, Weights=True)

            # Incorporate the chemical potential into the energies
            E_All = E_All - self.mu_Last*(N_All-2)

            # Sort from highest to lowest weights
            kk = np.argsort(W_All)[::-1]
            D_All = D_All[kk]
            W_All = W_All[kk]
            E_All = E_All[kk]

            if E_0 is None:
                NState = len(E_All)
                E_0 = np.dot(W_All, E_All)

            W_Cum = np.maximum(np.cumsum(W_All), 1e-5)

            E = np.cumsum(W_All*E_All)/W_Cum
            TS = -kbT*np.cumsum(W_All*safelog(W_All/D_All))/W_Cum
            F = E - TS
            dF = F[-1] - E_0

            eps_5 = np.abs(dF*0.05)

            CountStr = ""
            for eps in (0.01/eV, 0.1/eV, eps_5):
                pE = GetNS(E, eps)
                pTS = GetNS(TS, eps)
                pF = GetNS(F, eps)

                k0 = max(pE, pTS, pF)
                CountStr += "%4.0f%% "%(100.*k0/NState)
            print("%5.2f %7.3f %s" % (kbT*eV, dF*eV, CountStr))



class KSThermalHelper:
    def __init__(self, Engine, XHelp, N0=2.):
        self.Engine = Engine
        self.XHelp = XHelp
        self.N0 = N0

        self.Last_F = None
        self.Last_D = None
        self.Last_Q = None
        self.VBas = None

    def GetRefs(self, kbT, FCIHelper):
        # First do the interacting system if possible
        if not(FCIHelper is None):
            W_T, E_T, D_T = FCIHelper.Solve(kbT)
            D_All, N_All, E_All = FCIHelper.Flatten()
            W_All = FCIHelper.Flatten(W_T, Weights=True)
            tauS = -kbT*np.dot(W_All, safelog(W_All/D_All))

            FE_T = E_T - tauS
        else:
            print("Cannot deal with entropy terms in interacting system")
            quit()

        return E_T, D_T, tauS, FE_T

    def Solve(self, kbT, FCIHelper = None,
              Fresh = True, EGKS=False,
              F0 = None,
              Return_Densities = False, p4D = None,
              Mix=2.8, MaxIter=6000, ShowIter=200,
              EnCut=5e-8, dfCut=1e-6,
              fMin=1e-10, OneEl = False
              ):
        E_T, D_T, tauS, FE_T = self.GetRefs(kbT, FCIHelper)

        # Convert epsilon to f, D and Cf = (C sqrt(f))
        def f_D_Cf(epsilon, C):
            if OneEl:
                f = np.exp(-epsilon/kbT)/np.sum(np.exp(-epsilon/kbT))
            else:
                f = GetOcc(kbT, epsilon, N0=self.N0)
            D = np.einsum('pk,qk,k->pq', C, C, f)
            kk = f>fMin
            Cf = C[:,kk]*np.sqrt(f[kk])[None,:]
            return f, D, Cf

        # For simplicity
        Engine = self.Engine

        # Check that we have the right number of electrons
        N_T = np.vdot(Engine.S_ao, D_T)
        if np.abs(N_T-self.N0)>fMin:
            print("Warning, e- number off by %.3e for %.3f on %.3f"\
                  %(N_T-self.N0, N_T, self.N0))
            quit()

        # Convert D_T into C_T so that (C_T).dot(C_T.T) = D_T
        r_T, C_T = la.eigh(D_T)
        r_T = np.maximum(r_T, 0.)
        C_T = C_T * np.sqrt(r_T)[None,:]

        # Initial Hartree potential
        V_T = Engine.GetFJ(C_T)

        # Initialise to bare Hamiltonian (N<=1.2) or interacting
        if not(F0 is None):
            1 # Use a starting guess
        elif Fresh or (self.Last_F is None) or (EGKS):
            if N_T>1.2:
                F0 = Engine.H_ao + V_T/2.
            else:
                F0 = Engine.H_ao*1.
        else:
            F0 = self.Last_F * 1.

        F00 = F0*1.

        # Solve the inverse KS problem
        f_Old = 0.

        F_Min = None
        En_Min = 1000.

        VNL = 0.

        #F_Cache = np.zeros((NCache,F0.shape[0],F0.shape[1]))
        #En_Cache = np.zeros((NCache,))
        for iter in range(MaxIter):
            # Diagonalize the Fock operator
            epsilon, C = la.eigh(F0 + VNL, b=Engine.S_ao)
            f, D, Cf = f_D_Cf(epsilon, C)

            if EGKS:
                VNL =  0.5*(self.Engine.GetFJ(Cf) - self.Engine.GetFK(Cf))
            else:
                VNL = 0.


            # Calculate the difference in potentials and Hartree energy of diff
            DV = (Engine.GetFJ(Cf) - V_T)
            En = 0.5*np.vdot(DV, D - D_T)

            if (En<En_Min):
                En_Min = En
                F_Min = F0 + VNL

            if iter>1 and (En>0.1):
                print("Something seems to have gone wrong")
                F0 = F_Min
                Mix *= 0.7

            # Change in occupation factors
            dfErr = np.max(np.abs(f-f_Old))
            f_Old = f

            if (iter<5) or (iter%ShowIter)==0:
                print("%4d, En = %10.7f df = %10.7f, [%s] [%s]"\
                      %(iter, En, dfErr, 
                        NiceOcc(f[:2]), NiceEps(epsilon[:2])))

            if (En<EnCut) and (dfErr<dfCut) and (iter>=5): break


            F0 += Mix * DV


        epsilon, C = la.eigh(F_Min, b=Engine.S_ao)
        f, D, Cf = f_D_Cf(epsilon, C)
        DV = (Engine.GetFJ(Cf) - V_T)
        En = 0.5*np.vdot(DV, D - D_T)
        print("%4d, En = %10.7f,"%(iter, En), f[:4])
        print("f   = [ %s ]"%(NiceOcc(f)))
        print("eps = [ %s ]"%(NiceEps(epsilon[:8])))

        if False:
            VNL = 0.*F_Min + VNL
            VNL_C = (C.T).dot(VNL).dot(C)
            print(VNL_C[:4,:4])
                 


        self.Last_F = F_Min*1.
        self.Last_C = C*1.
        self.Last_epsilon = epsilon*1.
        self.last_D = D*1.
        self.Last_f = f*1.

        # Derive the H, J and K integrals (latter two with factor 0.5)
        NBas = C.shape[1]
        EJ = np.zeros((NBas,NBas))
        EK = np.zeros((NBas,NBas))
        H  = (C.T).dot(self.XHelp.Engine.H_ao).dot(C)
        for j in range(NBas):
            FJ = self.XHelp.Engine.GetFJ(C[:,j])
            FK = self.XHelp.Engine.GetFK(C[:,j])

            EJ[:,j] = 0.5*np.einsum('pk,qk,pq->k',C,C,FJ)
            EK[:,j] = 0.5*np.einsum('pk,qk,pq->k',C,C,FK)

        Props = {
            'f': f*1., 'epsilon': epsilon*1., 'C': C*1.,
            'H': H, 'EJ': EJ, 'EK': EK,
            }
        
        self.InvProps = {
            'f': f, 'epsilon': epsilon, 'C': C, 'kbT': kbT,
        }

        # E_Core = Ts + (n,v) + E_nn
        E_Core = np.vdot(Engine.H_ao, D) + Engine.Enn

        # EHxc_G = E_T - E_Core = enthalpic part of Hxc
        EHxc_G = E_T - E_Core

        # EHxc = FE_T - E_Core + tau Ss
        tauSs = 2.*(np.dot(f/2,safelog(f/2))+np.dot(1-f/2,safelog(1-f/2.)))
        EHxc = FE_T - (E_Core - tauSs)
        EHx  = f.dot(EJ).dot(f) - 0.5*f.dot(EK).dot(f)
        EH   = f.dot(EJ).dot(f)
        Ex   =  - 0.5*f.dot(EK).dot(f)

        # EHxc_S = EHxc - EHxc_G = entropic part of Hxc
        EHxc_S = EHxc - EHxc_G

        AllEns = {
            'E_T': E_T, 'FE_T': FE_T, 'tauS': tauS,
            'E_Core': E_Core, 'tauSs': tauSs,
            'EHxc': EHxc, 'EHxc_G': EHxc_G, 'EHxc_S': EHxc_S,
            'EH': EH, 'Ex': Ex, 'EHx': EHx,
        }

        # Calculate the Hartree and Fock exchange potentials
        VH = Engine.GetFJ(Cf)
        Vx =  - 0.5*Engine.GetFK(Cf)
        VHx = VH + Vx

        # Calculate the Hx
        EH_HF = 0.5*np.vdot(VH, D)
        Ex_HF = 0.5*np.vdot(Vx, D)
        EHx = 0.5*np.vdot(VHx, D)

        AllEns.update({
            'EHx': EHx, 'EH_HF': EH_HF, 'Ex_HF': Ex_HF,
        })

        # Calculate regular DFA xc and Hxc
        Exc_DFA, _ = Engine.GetDFA(Da=D/2)
        EHxc_DFA = EH_HF + Exc_DFA
        
        # Calculate the CLDA modified analogue
        Exc_CTDFA, _ = Engine.GetDFA(Da=D/2*(2/f[0])**(1/3))
        Exc_CTDFA *= f[0]/2.*Exc_CTDFA
        EHxc_CTDFA = EH_HF + Exc_CTDFA

        AllEns.update({
            'Exc_DFA': Exc_DFA, 'EHxc_DFA': EHxc_DFA,
            'Exc_CTDFA': Exc_CTDFA, 'EHxc_CTDFA': EHxc_CTDFA,
        })

        # Calculate Ex_FDT and through it EH_FDT and Ec_DD
        Ex_HF = EHx - EH_HF
        Ex_FDT = 0.
        for j in range(C.shape[1]):
            for k in range(C.shape[1]):
                Ex_FDT -= f[max(j,k)] * EK[j,k]
        EH_FDT = EHx - Ex_FDT
        Ec_DD = 0.32*(EH_HF - EH_FDT)

        AllEns.update({
            'EH_FDT': EH_FDT, 'Ex_FDT': Ex_FDT, 'Ec_DD': Ec_DD
        })

        # Calculate the combination rule DFA
        Exc_CDFA_SD = 0.
        Exc_CDFA_Pol = 0.
        Omega = f - np.hstack((f[1:],0.))
        for k in range(C.shape[1]):
            if np.abs(Omega[k])>1e-7:
                D_k = (C[:,:(k+1)]).dot(C[:,:(k+1)].T)
                Exc_k, _ = Engine.GetDFA(Da=D_k)
                Exc_k_Pol, _ = Engine.GetDFA(Da=D_k, Db=0.*D_k)
                if np.abs(Omega[k])>1e-4:
                    print("%10.8f %10.4f"%(Omega[k], Exc_k))
                Exc_CDFA_SD += Omega[k]/2.*Exc_k
                Exc_CDFA_Pol += Omega[k]*Exc_k_Pol

        EHxc_CDFA_SD = EH_FDT + Exc_CDFA_SD
        EHxc_CDFA = EHxc_CDFA_SD + Ec_DD

        AllEns.update({
            'Exc_CDFA_SD': Exc_CDFA_SD, 'EHxc_CDFA_SD': EHxc_CDFA_SD,
            'Exc_CDFA_Pol': Exc_CDFA_Pol, 'EHxc_CDFA_Pol': EH_FDT + Exc_CDFA_Pol,
            'EHxc_CDFA': EHxc_CDFA,
        })

        if not(Return_Densities) or (p4D is None):
            return AllEns, Props
        
        rho = p4D.Density(D)
        
        Omega, Omega_N, Map_N = Clusters(f, df=0.01)
        print("Reduced to %d densities"%(len(list(Omega_N))))
        rho_N = {}
        D_Cum = 0.
        Exc_CDFA_Test, rho_Test = 0., 0.
        for N in sorted(Map_N):
            kk = Map_N[N]
            D_N = 0.
            for k in kk:
                D_Cum += np.outer(C[:,k], C[:,k])
                D_N += Omega[k] * D_Cum
            D_N /= Omega_N[N]

            rho_N[N] = p4D.Density(D_N)

            rho_Test += Omega_N[N] * rho_N[N]
            Exc_N = Engine.GetDFA(Da=D_N)[0]
            if np.abs(Omega_N[N])>1e-4:
                print("%10.8f %10.4f"%(Omega_N[N], Exc_N))
            Exc_CDFA_Test += Omega_N[N]/2. * Exc_N

        
        if np.abs(Exc_CDFA_Test - Exc_CDFA_SD)>0.001:
            print("**** Error in Exc_CDFA = %8.5f and rho = %8.5f ****"\
                %(Exc_CDFA_Test - Exc_CDFA_SD, np.max(np.abs(rho_Test - rho))))
            if np.abs(Exc_CDFA_Test - Exc_CDFA_SD)>0.005:
                quit()
        
        RhoProps = {
            'rho': rho,  
            'Omega': Omega, 'Omega_N': Omega_N, 'Map_N': Map_N,
            'rho_N': rho_N,
        }

        return AllEns, Props, RhoProps


def LDAExchange(p4D, w, Props):
    f = Props['f']
    C = Props['C']
    EJ = Props['EJ']
    EK = Props['EK']

    EH_HF = (f).dot(EJ).dot(f)
    EHx = EH_HF - 0.5*(f).dot(EK).dot(f)
    EH_FDT = EHx
    for j in range(len(f)):
        for k in range(len(f)):
            EH_FDT += f[max(j,k)]*EK[j,k]

    Omega = f - np.hstack((f[1:],0.))


    rho = 0.
    Ex_CDFA = 0.
    rho_k = 0.
    for k in range(len(Omega)):
        rho_k +=  p4D.Density(np.outer(C[:,k],C[:,k]))
        rho += Omega[k]*rho_k

        Ex_CDFA -= Omega[k]/2*np.dot(w, 0.7385587663820224*(rho_k*2.)**(4/3))

    Ex_DFA = -np.dot(w, 0.7385587663820224*rho**(4/3))

    print("Ex_DFA  = %10.5f Ex_CDFA  = %10.5f"%(Ex_DFA, Ex_CDFA))
    print("EHx_DFA = %10.5f EHx_CDFA = %10.5f"%(EH_HF+Ex_DFA, EH_FDT+Ex_CDFA))

    return EH_HF+Ex_DFA, Ex_DFA, EH_FDT+Ex_CDFA, Ex_CDFA

class HFThermalHelper(KSThermalHelper):
    def QSolve(self, F, kbT, alpha=1., gamma=1., Avg=False):
        epsilon, C = la.eigh(F, b = self.Engine.S_ao)

        if Avg: epsilon[1:4]==np.mean(epsilon[1:4])

        f = GetOcc(kbT, epsilon)

        kk = f>1e-10

        fC = C[:,kk] * np.sqrt(f[kk])[None,:]
        D_T = (fC).dot(fC.T)
        VH = self.Engine.GetFJ(fC)
        Vx = -0.5*self.Engine.GetFK(fC)
        EH = 0.5*np.vdot(D_T, VH)
        Ex = 0.5*np.vdot(D_T, Vx)

        ECore = np.vdot(D_T, self.Engine.H_ao) + self.Engine.Enn

        E_T = ECore + EH + Ex
        t = f/2.
        tauS = -kbT*2*np.sum(t*safelog(t) + (1-t)*safelog(1-t))

        Props = {
            'f': f, 'epsilon': epsilon, 'C': C, 'kbT': kbT,
            'ECore': ECore, 'EH': EH, 'Ex': Ex, 
            'E': ECore+EH+Ex, 'FE': ECore+EH+Ex-tauS, 'tauS': tauS,
        }   

        return gamma*VH+alpha*Vx, D_T, E_T, tauS, Props

    def GetRefs(self, kbT, FCIHelper=None):
        F = self.Engine.F*1.
        epsilon_Old = 0.
        f_Old = 0.
        for step in range(100):
            VHx, D_T, E_T, tauS, Props = self.QSolve(F, kbT)
            epsilon = Props['epsilon']*1.
            f = Props['f']*1.
            #print(epsilon)

            FE_T = E_T - tauS

            #print("FE_T = %8.4f, E_T = %8.4f tauS = %8.4f"%(FE_T, E_T, tauS))

            F = self.Engine.H_ao + VHx

            Err_eps = np.max(np.abs(epsilon-epsilon_Old))
            Err_f = np.max(np.abs(f - f_Old))
            if Err_eps<1e-5 and Err_f<1e-6:
                break

            epsilon_Old = epsilon*1.
            f_Old = f*1.


        print("FE_T = %8.4f, E_T = %8.4f tauS = %8.4f"%(FE_T, E_T, tauS))
        self.HFProps = { K: Props[K]*1. for K in Props }
        print(NiceOcc(self.HFProps['f']))

        self.F_HF = F*1.
        self.epsilon_HF = epsilon*1.

        return E_T, D_T, tauS, FE_T

    def ComputeEns(self, PP, Report=False):
        kbT = PP['kbT']
        f = PP['f']
        C = PP['C']
        fC = C * np.sqrt(f)
        D_T = fC.dot(fC.T)
        ECore = self.Engine.Enn + np.vdot(self.Engine.H_ao, D_T)

        VH = self.Engine.GetFJ(fC)
        Vx = -0.5*self.Engine.GetFK(fC)
        EH = 0.5*np.vdot(D_T, VH)
        Ex = 0.5*np.vdot(D_T, Vx)

        Ex_FDT = 0.
        f_ = f[f>1e-8]
        for i in range(len(f_)):
            Fi = self.Engine.GetFK(C[:,i])
            for j in range(i, len(f_)):
                Jij = 0.5*(C[:,j]).dot(Fi).dot(C[:,j])
                Ex_FDT -= f_[max(i,j)]*Jij
        EH_FDT = EH + Ex - Ex_FDT


        E_T = ECore + EH + Ex
        t = f/2.
        tauS = -kbT*2*np.sum(t*safelog(t) + (1-t)*safelog(1-t))
        FE_T = E_T - tauS

        # deps = epsilon - mu
        eps_mu = GetOcc(PP['kbT'], PP['epsilon'], deps=True)


        if Report:
            print("FE_T = %8.4f, E_T = %8.4f tauS = %8.4f"%(FE_T, E_T, tauS))
            print("  - EH_HF = %8.4f Ex_HF = %8.4f EH_FDT = %8.4f Ex_FDT = %8.4f"\
                  %(EH, Ex, EH_FDT, Ex_FDT))
            print("  - epsilon = " + " ".join(["%6.3f"%(x) for x in eps_mu[:8]]))

        return {'E_T': E_T, 'FE_T': FE_T, 'tauS': tauS,
                'ECore': ECore,
                'EH': EH, 'Ex': Ex,
                'EH_FDT': EH_FDT, 'Ex_FDT': Ex_FDT,
                'epsilon': PP['epsilon'], 'eps_mu': eps_mu, 'f': PP['f'], }
    
    def SolveHF(self, kbT, Mix=0.2, Mix2=0.5, MaxStep=200):
        if not(self.Last_F is None):
            F = self.Last_F*1.
        else:
            # Iterate to near the self-consistent 0K solution
            F = self.Engine.F*1.
            for step in range(5):
                epsilon, C = la.eigh(F, b=self.Engine.S_ao)
                F = self.Engine.H_ao + 0.5*self.Engine.GetFJ(C[:,0])
            ######

        E_Old = 0.
        epsilon_Old = 0.
        f_Old = 0.
        for step in range(MaxStep):
            VHx, D_T, E_T, tauS, Props = self.QSolve(F, kbT)
            epsilon = Props['epsilon']*1.
            f = Props['f']*1.

            FE_T = E_T - tauS

            if step>(MaxStep//2):
                print("FE_T = %8.4f, E_T = %8.4f tauS = %8.4f"%(FE_T, E_T, tauS))

            F2 = F
            if step>0:
                F = (1-Mix)*(self.Engine.H_ao + VHx) + Mix*F
            else:
                F = (1-Mix)*(self.Engine.H_ao + VHx) + Mix*((1-Mix2)*F + Mix2*F2)

            

            Err_E = np.abs(E_T-E_Old)/(1.-Mix)
            Err_eps = np.max(np.abs(epsilon-epsilon_Old))/(1.-Mix)
            Err_f = np.max(np.abs(f - f_Old))/(1.-Mix)

            if Err_E<1e-6 and Err_eps<1e-6 and Err_f<1e-8:
                break

            E_Old = E_T*1.
            epsilon_Old = epsilon*1.
            f_Old = f*1.

        if step>(MaxStep-10):
            print("!!!! WARNING !!!! - this is very slow convergence")

        self.Last_F = F*1.

        print("FE_T = %8.4f, E_T = %8.4f tauS = %8.4f"%(FE_T, E_T, tauS))
        self.HFProps = { K: Props[K]*1. for K in Props }
        print(NiceOcc(self.HFProps['f']))

        self.F_HF = F*1.
        self.D_HF = D_T*1.
        self.epsilon_HF = epsilon*1.

        return E_T, D_T, tauS, FE_T


    def SolveEXX(self, kbT, UseDFAStart=False):
        # Creat the potential basis on the first step
        if self.VBas is None:
            # Use the DF basis for potentials   
            wfn = self.Engine.wfn
            basis = wfn.basisset()

            #BName = wfn.basisset().name()
            BName = 'hepotbasis'
            aux_basis = psi4.core.BasisSet.build\
                (wfn.molecule(), "DF_BASIS_MP2", "",
                'RIFIT', BName)
            
            zero_basis = psi4.core.BasisSet.zero_ao_basis_set()
            mints = psi4.core.MintsHelper(basis)

            # Get all basis functions
            self.VBas = np.squeeze(mints.ao_eri(aux_basis, zero_basis, basis, basis))

            # Get the normalization for df basis
            self.Norm = np.squeeze(mints.ao_overlap(aux_basis, zero_basis))

            # Scale the VBas
            self.VBas = self.VBas/self.Norm[:,None,None]

            # Trim to s only
            SList = []
            for s in range(aux_basis.nshell()):
                l = aux_basis.shell(s).am
                if l==0:
                    SList += [aux_basis.shell_to_ao_function(s)]

            self.VBas = self.VBas[SList,:,:]

            print("Potential basis dimension = %d"%(self.VBas.shape[0]))

        # Starting guess @ HF density
        C = self.HFProps['C']
        fC = C * np.sqrt(self.HFProps['f'])[None,:]
        D = (fC).dot(fC.T)
        if UseDFAStart: # From DFA
            FCore = self.Engine.H_ao + self.Engine.GetFJ(fC) \
                + self.Engine.GetDFA(Da=D/2.)[1]
        else:
            FCore = self.Engine.H_ao + self.Engine.GetFJ(fC)/2.

        def QuickEn(Q, AsF=False, AsFE=False, w_eps = 0, zeta = 1e-4):
            F = FCore + np.tensordot(Q, self.VBas, axes=((0,),(0,)))
            if AsF: return F

            VHx, D_T, E_T, tauS, Props = self.QSolve(F, kbT, Avg=False)
            if AsFE: return E_T-tauS, E_T, tauS

            # Nudge the 2ps together
            epsilon = Props['epsilon']
            Err_eps = ((epsilon[2]-epsilon[1])**2 + (epsilon[3]-epsilon[1])**2)

            #print("%8.5f %8.5f %8.5f"%(E_T-tauS, E_T, tauS))

            
            # Return the free energy with a small contribution from eigenvalues
            return 1000*(E_T - tauS - FE0) + w_eps*Err_eps + zeta*np.sum(Q)**2
        
        NAux = self.VBas.shape[0]
        Q0 = np.zeros((NAux,))
        
        FE0, E0, tauS0 = QuickEn(Q0, AsFE=True)

        #res = opt.minimize(lambda Q: QuickEn(Q,w_eps=1e6), Q0)
        #Q0 = res.x
        res = opt.minimize(lambda Q: QuickEn(Q), Q0) 
        Q = res.x
        self.Last_Q = Q*1.

        print(self.Last_Q)
        
        F = QuickEn(Q, AsF=True)

        VHx, D_T, E_T, tauS, Props = self.QSolve(F, kbT)
        FE_T = E_T - tauS

        FE1, E1, tauS1 = QuickEn(Q0, AsFE=True)
        #print("**** FE0 = %8.5f E0 = %8.5f FE1 = %8.5f E1 = %8.5f ****"\
        #      %(FE0, E0, FE1, E1))

        print("FE_T = %8.4f, E_T = %8.4f tauS = %8.4f"%(FE_T, E_T, tauS))
        self.KSProps = { K: Props[K]*1. for K in Props }
        print(NiceOcc(self.KSProps['f']))
        return self.KSProps, F*1.


class EGKSThermalHelper(HFThermalHelper):
    def QSolve(self, F, kbT, alpha=1.):
        epsilon, C = la.eigh(F, b = self.Engine.S_ao)
        f = GetOcc(kbT, epsilon)

        kk = f>1e-9

        fC = C[:,kk] * np.sqrt(f[kk])[None,:]
        D_T = (fC).dot(fC.T)
        VH = self.Engine.GetFJ(fC)
        Vx = -0.5*self.Engine.GetFK(fC)
        EH = 0.5*np.vdot(D_T, VH)
        Ex = 0.5*np.vdot(D_T, Vx)

        E_1, F_1 = self.Engine.GetHx(Ca = fC/np.sqrt(2))
        E_2, F_2 = self.Engine.GetDFA(Da = D_T/2.)
        EHxc = E_1 + E_2
        VHxc = F_1 + F_2


        ECore = np.vdot(D_T, self.Engine.H_ao) + self.Engine.Enn

        E_T = ECore + EHxc
        t = f/2.
        tauS = -kbT*2*np.sum(t*safelog(t) + (1-t)*safelog(1-t))

        Props = {
            'f': f, 'epsilon': epsilon, 'C': C, 'kbT': kbT,
            'ECore': ECore, 'EH': EH, 'Ex': Ex, 'EHxc': EHxc,
        }   

        return VHxc, D_T, E_T, tauS, Props
