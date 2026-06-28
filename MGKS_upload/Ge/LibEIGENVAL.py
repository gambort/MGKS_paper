import numpy as np

def CompareEigs(eps1, eps2, avg=np.mean):
    r0, r1 = -2., 2.

    for iter in range(9):
        VMin = 1e10
        d = (r1-r0)/10
        for r in np.linspace(r0, r1, 11):
            V = avg(np.abs(eps1-(eps2+r)))
            if V<VMin:
                r0, r1 = r-d, r+d
                VMin = V

    r = (r0+r1)/2
    return eps1-(eps2+r)

class EIGENVALHelper:
    def __init__(self, FName=None, Nk=None):
        if not(FName is None):
            self.ReadFile(FName, Nk)
            
    def ReadFile(self, FName=None, Nk=None):
        if FName is None: return
        try:
            with open(FName, "r") as F:
                F.readline()
                F.readline()
                sigma = float(F.readline())
                F.readline()
                F.readline()
                X = [int(x) for x in F.readline().split()]
                _, NK, NBands = tuple(X)

                Raw = {}

                for k in range(NK):
                    F.readline()
                    Hdr = F.readline().split()
                    if len(Hdr)<2: break

                    Content = []
                    for p in range(NBands):
                        Content += [ [float(x) for x in F.readline().split()] ]

                    Raw[k] = { 'Header': Hdr, 'Content': Content }
        except FileNotFoundError:
            print("Error: The file does not exist.")
        except PermissionError:
            print("Error: You do not have permission to access this file.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            
        self.Data = {}
        for k in Raw:
            k1,k2,k3,wk = tuple([float(x) for x in Raw[k]['Header']])

            F = np.array(Raw[k]['Content'])
            epsilon = F[:,1]
            Occ = F[:,2]

            self.Data[k] = {
                'k': np.array([k1,k2,k3]), 'wk': wk,
                'epsilon_k': epsilon,
                'Occ_k': Occ,
            }

        self.k = np.zeros((NK,3))
        self.wk = np.zeros((NK,))
        self.eps_nk = np.zeros((NK,NBands))
        self.N = 0.
        for k in range(NK):
            self.k[k] = self.Data[k]['k']
            self.wk[k] = self.Data[k]['wk']
            self.eps_nk[k,:] = self.Data[k]['epsilon_k']
            self.N += np.sum(self.Data[k]['Occ_k']*self.wk[k])

        self.N = 2.*np.round(self.N)

        if Nk is None:
            # Estimate the grid based on the available points
            print("WARNING! Estimated grids can go wrong - recommend specifying Nk")
            a = np.array([np.min(self.k[self.k[:,p]>0.01,p])
                          for p in range(3)])
            self.Nk = np.round(1/a)
        else:
            self.Nk = np.array(Nk)
            
        self.dN = 1./np.prod(self.Nk)

        return self.Data

    def Occ_N(self, tau, N, With_mu=True):
        def Err_mu(mu):
            F = self.Occ_mu(tau, mu)
            return np.sum(np.dot(self.wk, F)) - N

        eps_min = self.eps_nk.min()
        eps_max = self.eps_nk.max()
        
        Delta = eps_max - eps_min
        bb = [ eps_min - Delta/2, eps_max + Delta/2 ]
        
        mu_ = np.interp([0,0.5,1], [0,1], bb)
        dN_ = np.array([ Err_mu(mu) for mu in mu_ ])

        for iter in range(100):
            if dN_[0]*dN_[1]<0.:
                k0,k1=0,1
            else:
                k0,k1=1,2

            if mu_[k1]-mu_[k0]<1e-10: break

            mun = (mu_[k0] + mu_[k1])/2.
            dNn = Err_mu(mun)

            if np.abs(dNn) < 1e-8: break

            mu_ = [ mu_[k0], mun, mu_[k1] ]
            dN_ = [ dN_[k0], dNn, dN_[k1] ]


        mu = np.interp(0., dN_, mu_)

        f_nk = self.Occ_mu(tau, mu)
        #print("__N = %.8f %.8f"%(self.Avg(f_nk), N))
        if With_mu: return f_nk, mu
        else: return f_nk
    
    def Occ_mu(self, tau, mu):
        x = np.minimum( (self.eps_nk - mu)/tau, 80. )
        return 2./(1. + np.exp( x ))

    def Avg(self, F):
        return np.sum(np.dot(self.wk, F))

    def Hirata(self, tau=1e-5, dN=None, s=3, sL=None, sR=None):
        # By default dN = 1/Nk
        if dN is None: dN = self.dN
        
        # By default sL=sR=s
        if sL is None: sL = s
        if sR is None: sR = s

        # Take the difference of right and left derivatives at EH.N
        # using a fit to dNL points left and dNR points right
        NI = int(np.round(self.N))
        NN = NI + np.arange(-sL, sR+1)*dN
        epsN = 0.*NN
        for K, N in enumerate(NN):
            f = self.Occ_N(tau, N, With_mu=False)
            epsN[K] = self.Avg(f * self.eps_nk)

        # Left fit
        pL = np.polyfit(NN[NN<=NI]-NI, epsN[NN<=NI], min(2, sL))
        # Right fit
        pR = np.polyfit(NN[NN>=NI]-NI, epsN[NN>=NI], min(2, sR))

        if (pL[-1]-pR[-1])>5e-4:
            print("Warning, do not approach same value, try s=1")

        return pR[-2] - pL[-2]
    

class MultiEIGENVALHelper(EIGENVALHelper):
    def __init__(self, DFA=None, tau = 0.0,
                 EIGENVALDir='Outputs/',
                 Nk=None):
        self.EIGENVALDir = EIGENVALDir
        self.Nk = Nk
        
        if not(DFA is None):      
            self.ReadFileByID(DFA=DFA, tau=tau)

    def GetBestFName(self, DFA, tau):
        import glob
        Core = self.EIGENVALDir + 'EIGENVAL_%s_'%(DFA)

        tauP = []
        FP = []
        for F in glob.glob(Core + "*"):
            Q = open(F)
            NF = len(list(Q))
            Q.close()
            
            if NF<10: continue
            
            FP += [F]
            tauP += [float(F.split('_')[-1])]

        if len(tauP)>1:
            kk = np.argsort(np.abs(np.array(tauP)-tau))
        else:
            kk = [0,0]
        self.last_FName = [ FP[k] for k in kk ]
        self.last_tau = [ tauP[k] for k in kk ]

        k = kk[0]               

        return FP[k]

    def InterpFiles(self, DFA='PBE', tau = 0.0):
        self.GetBestFName(DFA, tau)

        FName1, tau1 = self.last_FName[0], self.last_tau[0]
        FName2, tau2 = self.last_FName[1], self.last_tau[1]

        if tau1==tau2:
            self.ReadFile(FName=FName1, Nk=self.Nk)
            return


        self.ReadFile(FName=FName1, Nk=self.Nk)
        eps_nk_1 = self.eps_nk*1.

        self.ReadFile(FName=FName2, Nk=self.Nk)
        eps_nk_2 = self.eps_nk*1.

        W1 = (tau-tau2)/(tau1-tau2)
        W2 = (tau1-tau)/(tau1-tau2)
        
        #print("Interpolating on %.3f %s and %.3f %s"%(W1, FName1, W2, FName2))

        self.eps_nk = W1*eps_nk_1 + W2*eps_nk_2

    def ReadFileByID(self, DFA='PBE', tau = 0.0,
                 FName=None):
        if FName is None:
            FName = self.GetBestFName(DFA, tau)
        else:
            self.ReadFile(FName, self.Nk)

        self.ReadFile(FName, self.Nk)


if __name__ == '__main__':
    # Note, this function uses the default of eV energy units
    # from VASP

    print('='*72)
    print("Example for a single file")

    # Note, could also do:
    #   EH = EIGENVALHelper("EIGENVAL_Demo", Nk=(6,6,6))
    # in one line

    # Set up the EIGENVAL helper
    EH = EIGENVALHelper()
    # Read an individual file (helps to specify k grid)
    EH.ReadFile("EIGENVAL_Demo", Nk=(6,6,6))
    
    # Get the Hirata gap at various temperatures
    for tau in np.linspace(1e-5, 0.08, 9): # tau is in eV
        HG = EH.Hirata(tau)
        HG_S = EH.Hirata(tau, s=1) # Test using one point only

        
        print("Hirata gap = %8.3f eV at tau = %.3f eV [default 3 point]"\
              %(HG, tau))
        if np.abs(HG-HG_S)>5e-4:
            HG_2 = EH.Hirata(tau, s=2) # Test using 2 points
            print(" => %8.3f eV diff at one point"%(HG_S-HG))
            print(" => %8.3f eV diff at two points"%(HG_2-HG))
    

    print('='*72)
    print("Example for a series of files (OT-RSH)")
    DFA = 'OT'
    
    EHM = MultiEIGENVALHelper(EIGENVALDir='Outputs/', Nk=(6,6,6))
    # Files are [EIGENVALDir]/EIGENVAL_[DFA]_[tau]
    EHM.ReadFileByID(DFA=DFA, tau=0.0) # Reading a file populates last_tau

    # Available  tau is in eV
    for tau in EHM.last_tau:
        EHM.ReadFileByID(DFA=DFA, tau=tau)
        print("Hirata gap = %8.3f eV at tau = %.3f eV"\
              %(EHM.Hirata(tau), tau))
