import numpy as np

eV = 27.211
Kelvin = 315777
kKelvin = Kelvin/1000

RefBasis = 'def2-qzvppd'

def safeexp(x):
    return np.exp(np.minimum(x, 86.))

def safelog(x):
    return np.log(np.maximum(x, 1e-16))

def f_to_Ss(f):
    fh = f/2.
    return -2.*np.sum(fh*safelog(fh) + (1-fh)*safelog(1-fh))

def eps_to_Occ(eps, tau, N, Cut=86., d=None):
    if d is None: d = np.ones((len(eps),))
    
    tau = max(tau, 1e-4)

    eps = eps - eps.min()
    epsl, epsr = eps.min()-8*tau, eps.max()+8*tau

    def dN(mu):
        f = 2./(1. + np.exp(np.minimum((eps-mu)/tau,Cut)))
        return np.dot(d, f)-N

    mu0 = np.array([epsl, (epsl+epsr)/2, epsr])
    dN0 = np.array([dN(mu) for mu in mu0])
    
    for iter in range(40):
        k= np.argmin(np.abs(dN0))
        mu = mu0[k]
        if np.abs(dN0[k])<1e-8: break

        if dN0[0]*dN0[1]<0.: k0, k1 = 0, 1
        else: k0, k1 = 1, 2

        mu_ = (mu0[k0] + mu0[k1])/2
        dN_ = dN(mu_)

        mu0 = np.array([mu0[k0], mu_, mu0[k1]])
        dN0 = np.array([dN0[k0], dN_, dN0[k1]])

    return 2./(1. + np.exp(np.minimum((eps-mu)/tau,Cut)))


class HeHFHelper:
    def __init__(self, Basis=RefBasis,):
        self.Basis = Basis
        FileName = "Data/He_HF_Thermal_%s_pbe.npz"%(self.Basis)
        X = np.load(FileName, allow_pickle=True)
        self.Data = X['Data'][()]

        self.tau = np.sort(list(self.Data))
        self.Methods = list(self.Data[self.tau[0]])
        self.Terms = list(self.Data[self.tau[0]][self.Methods[0]])

        self.Rename = {}

    def AddExact(self):
        FileName = "Data/He_Exact_Thermal_%s_pbe.npz"%(self.Basis)
        X = np.load(FileName, allow_pickle=True)
        Exact_EKS  = X['Data'][()]
        Exact_EGKS = X['Data_EGKS'][()]
        Extra_EKS  = X['ExtraData'][()]
        Extra_EGKS = X['ExtraData_EGKS'][()]

        ########################################################
        # Store the reference energies
        ########################################################
        EE = X['RefData'][()]['Ens']
        NAll = 2
        for N in EE:
            for D in EE[N]: NAll += len(EE[N][D])
        self.ExactEns = {
            'N': np.zeros((NAll,),dtype=float),
            'D': np.ones((NAll,),dtype=float),
            'E': np.zeros((NAll,),dtype=float),
        }
        K = 1
        for N in EE:
            for D in EE[N]:
                n = len(EE[N][D])
                self.ExactEns['N'][K:(K+n)] = N
                self.ExactEns['D'][K:(K+n)] = D
                self.ExactEns['E'][K:(K+n)] = EE[N][D]
                K += n

        # Add a 4 at high energy
        self.ExactEns['N'][K] = 4
        self.ExactEns['D'][K] = 1
        self.ExactEns['E'][K] = np.max(self.ExactEns['E'])

        ########################################################
        # Store the other reference properties
        ########################################################
        tau_ = np.sort(list(Exact_EKS))

        Rename = {'tauSs': 'tauS', 'E_Core': 'ECore',}

        for ExactKey, Exact, Extra in \
            [ ('Exact', Exact_EKS, Extra_EKS),
              ('Exact_EGKS', Exact_EGKS, Extra_EGKS)]:
            T = {}
            for Key in ('E_T', 'FE_T', 'E_Core', 'tauSs',
                        'EHx', 'EH', 'Ex', 'EHxc', 'EHxc_S', 'EHxc_G'):
                if Key in Rename: OKey = Rename[Key]
                else: OKey = Key

                y = np.array([Exact[t][Key] for t in tau_])
                T[OKey] = np.interp(self.tau, tau_, y)
                if Key=='tauSs': T[OKey] = -T[OKey]

            T['f'] = [Extra[t]['f'] for t in tau_]
            T['epsilon'] = [Extra[t]['epsilon'] for t in tau_]

            for tau in self.tau:
                self.Data[tau][ExactKey] = {}

            for Key in T:
                for tau, V in zip(self.tau, T[Key]):
                    self.Data[tau][ExactKey][Key] = V


    def FixFEGrowth(self, FECut=1e-4):
        AllMethods = list(self.Data[self.tau[0]])

        Mask = {}
        for Method in AllMethods:
            FE = self.Get(Method, 'FE_T', Zero=True)
            kk = np.argwhere(FE>FECut).reshape((-1,))
            kkp = np.argwhere(FE<=FECut).reshape((-1,))

            if len(kk)==0:
                Mask[Method] = []
                continue

            Mask[Method] = kk
            print("%s has increasing FE @ %s"\
                  %(Method,
                    " ".join(["%.1f"%(x)
                              for x in self.tau[kk]*315.777]
                    )))

            Keys = list(self.Data[self.tau[0]][Method])
            ArrKeys = ('epsilon', 'f', 'eps_mu')
            
            for Key in Keys:
                X = np.array([
                    self.Data[tau][Method][Key] for tau in self.tau
                ])
                if len(X.shape)==1:
                    X = np.interp(self.tau, self.tau[kkp], X[kkp])
                else:
                    Y = X*0.
                    for p in range(X.shape[1]):
                        Y[:,p] = np.interp(self.tau, self.tau[kkp], X[kkp,p])
                    X = Y

                for k, tau in enumerate(self.tau):
                    self.Data[tau][Method][Key] = X[k]

        return Mask

    def Get(self, Method, Term, Zero=False):
        # Replace interacting reference by MKS
        if Method in ('Int', 'Ref'): Method = 'Exact'
        
        Key = Method+"___"+Term
        if Zero: Key += "___zerod"

        #print(list(self.Data[self.tau[0]][Method]))

        if Term in ('EEXX', 'E_EXX', 'E_HF'):
            EC = self.Get(Method, 'ECore', Zero=Zero)
            EH = self.Get(Method, 'EH', Zero=Zero)
            Ex = self.Get(Method, 'Ex', Zero=Zero)
            return EC+EH+Ex

        if Term=='Orc':
            EEXX = self.Get(Method, 'EEXX', Zero=Zero)
            FE = self.Get('Exact', 'FE_T', Zero=Zero)
            tauSs = self.Get(Method, 'tauSs', Zero=Zero)
            return FE-tauSs-EEXX


        if Term=='EHx' and not('EHx' in self.Data[self.tau[0]][Method]):
            EH = self.Get(Method, 'EH', Zero=Zero)
            Ex = self.Get(Method, 'Ex', Zero=Zero)
            return EH+Ex

        if Term == 'tauSs':
            E = 0.*self.tau
            for K, tau_ in enumerate(self.tau):
                f = self.Data[tau_][Method]['f']
                E[K] = tau_*f_to_Ss(f)
        elif Term in ('f', 'epsilon'):
            E = []
            for K, tau_ in enumerate(self.tau):
                E += [self.Data[tau_][Method][Term]]
            return np.array(E)
        else:        
            E = 0.*self.tau
            for K, tau_ in enumerate(self.tau):
                if Term in self.Data[tau_][Method]:
                    E[K] = self.Data[tau_][Method][Term]
                else:
                    E[K] = self.Data[tau_][Method][self.Rename[Term]]
                

        if Zero:
            p = np.polyfit(self.tau[:6], E[:6], 2)
            #print(Method, Term, p)
            return E - p[2]
        else:
            return E
            

    def GetGap_Exact(self):
        D = self.ExactEns['D']
        N = self.ExactEns['N']
        E = self.ExactEns['E']

        E2 = np.min(E[N==2])
        E1 = np.min(E[N==1])
        E3 = np.min(E[N==3])

        IP = E1 - E2
        EA = E2 - E3
        FG = IP - EA
        print("IP = %6.2f EA = %6.2f Gap = %6.2f"\
              %(eV*IP, eV*EA, eV*(IP-EA)))

        
        def GetW_N(tau, mu):
            Arg = E - mu*N
            W = safeexp(-(Arg-Arg.min())/tau)
            W /= np.sum(W*D)
            return W, np.dot(W*D, N)

        mu0 = {}
        Et  = { 1:0.*self.tau, 2:0.*self.tau, 3:0.*self.tau }
        for Ktau, tau_ in enumerate(self.tau):
            for Nt in (1,2,3):
                dNBest = 1000
                for mu_ in np.arange(-100,101):
                    dN_ = GetW_N(tau_,mu_)[1]-Nt
                    if np.abs(dN_)<dNBest:
                        dNBest = np.abs(dN_)
                        muBest = mu_

                
                mu = [muBest-1,muBest,muBest+1]
                dN = [GetW_N(tau_,mu[0])[1]-Nt,
                      GetW_N(tau_,mu[1])[1]-Nt,
                      GetW_N(tau_,mu[2])[1]-Nt ]
                    
                for step in range(500):
                    # Break if accidentally excellent
                    if np.abs(dN[1])<1e-8:
                        k0, k1 = 1,1
                        mup = mu[1]
                        dNp = dN[1]
                        break
                    
                    if dN[0]*dN[1]<0:
                        k0,k1=0,1
                    else:
                        k0,k1=1,2

                    mup = (mu[k0] + mu[k1])/2
                    dNp = GetW_N(tau_,mup)[1]-Nt

                    if np.abs(dNp)<1e-8: break

                    mu = [mu[k0], mup, mu[k1]]
                    dN = [dN[k0], dNp, dN[k1]]
                    
                mu0[Nt] = mup
                W, _ = GetW_N(tau_, mu0[Nt])
                Et[Nt][Ktau] = np.dot(W*D, E)


            #print("%6.2f %6.2f %6.2f %6.2f"\
            #      %(eV*tau_, eV*(Et[1]-Et[2]), eV*(Et[2]-Et[3]),
            #        eV*(Et[3]+Et[1]-2*Et[2]) ))

        return FG, Et


            
