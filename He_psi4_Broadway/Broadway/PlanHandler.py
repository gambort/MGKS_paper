__version__ = 1.6
__author__ = "Tim Gould"

import numpy as np
#import scipy.linalg as la

zero_round = 1e-7

np.set_printoptions(precision=5, suppress=True)

#############################################################################
# Auxilliary functions
#############################################################################

# Get occupation factor safely by setting to zero if out
# of bounds
def safeocc(f, k):
    if k>=len(f): return 0.
    else: return f[k]


# Convert an ordered list into an ordered list of swaps to
# generate the list
def GetSwaps(kk):
    if not(len(set(kk))==len(kk)):
        print("List has duplicates - quitting!")
        quit()

    Done = np.zeros((len(kk),))
    Swaps = []
    for k, kp in enumerate(kk):
        # Don't do anything if done
        if Done[k]>0 or Done[kp]>0: continue

        # No swap necessary
        if k==kp: continue

        # Idempotent swap
        if kk[kp] == k: 
            Done[k] = 1
            Done[kp] = 1
            Swaps += [(k,kp)]
        else:
            # Build the chain
            kn = kp
            Chain = [k,kp]
            while not(kn==k):
                kn = kk[kn]
                Chain += [kn,]
            Chain = list(set(Chain))
            if len(Chain)==3:
                for k in Chain: Done[k]=1
                Swaps += [(Chain[0], Chain[1]), (Chain[1], Chain[2])]
            else:        
                print("Chains >3 not implemented - results will be bad")

    return Swaps

# Get the parity of a Slater determinant from a list

def QParity(occupied_orbitals, num_orbitals):
    """
    Calculates the parity of a Slater determinant.

    Args:
        occupied_orbitals (list): A list of integers representing the indices of occupied orbitals.
        num_orbitals (int): The total number of orbitals.

    Returns:
        int: The parity of the Slater determinant (1 for even, -1 for odd).
    """

    if not all(0 <= i < num_orbitals for i in occupied_orbitals):
      raise ValueError("Orbital indices must be within the valid range.")

    virtual_orbitals = sorted(list(set(range(num_orbitals)) - set(occupied_orbitals)))
    permutation = list(occupied_orbitals) + list(virtual_orbitals)

    length = len(permutation)
    elements_seen = [False] * length
    cycles = 0
    for index, already_seen in enumerate(elements_seen):
        if already_seen:
            continue
        cycles += 1
        current = index
        while not elements_seen[current]:
            elements_seen[current] = True
            current = permutation[current]
    return ((length-cycles) % 2)*2-1


#############################################################################
# Occupation helper
#############################################################################

class OccHelper:
    def __init__(self, f=None, WList=None, NDbl=1):
        """Class for handling orbital occupations"""
        if not(WList is None):
            self.FromWList(WList)
        elif not(f is None):
            self.FromOcc(f)            
        else:
            self.FromOcc([2.]*NDbl)

    def Copy(self):
        return OccHelper(f=self.npRaw)

    def __str__(self): return self.AsStr()

    def __mul__(self, x):
        return OccHelper(f=x*self.npRaw)
    
    def __truediv__(self, x):
        return OccHelper(f=self.npRaw/x)
    
    def __add__(self, x):
        return OccHelper(WList=[(1.,self),(1.,x)])

    def __sub__(self, x):
        return OccHelper(WList=[(1.,self),(-1.,x)])
    
    def __len__(self):
        return len(self.npRaw)

    def NiceRaw(self, Raw): # Handles raw matrices
        Raw = np.array(Raw, dtype=float)
        
        if len(Raw.shape)==1: # Ensure is NOcc x 1 of 1D
            Raw = Raw.reshape((-1,1))

        s = Raw.shape
        if not(s[1] in (1,2)):
            print("Occupation array not of valid shape - must be NOcc x (1 or 2)")
            quit()

        if s[1]==1: # Add spin information of NOcc x 1
            sRaw = np.zeros((s[0],2))
            sRaw[:,0] = np.minimum(Raw[:,0], 1.)
            sRaw[:,1] = Raw[:,0] - sRaw[:,0]
            return sRaw
        else:
            return Raw

    def FromOcc(self, f):
        if isinstance(f, OccHelper):
            self.npRaw = self.NiceRaw(f.npRaw)
        else:
            self.npRaw = self.NiceRaw(f)
        self.Process()

        return self
    
    def FromWList(self, WList):
        """ WList is [(w1,OccHelper1), (w2,OccHelper2), ...]"""
        N = 0
        for w,f in WList:
            N = max(N, len(f))

        self.npRaw = np.zeros((N,2))
        for w,f in WList:
            self.npRaw[:len(f.npRaw),:] += w*f.npRaw

        self.Process()
        
        return self
            
    
    def Promote(self, i, a, Copy=False,
                df=1., N=0):
        """Return a new occupation with an orbital promoted from i to a
        df is the occupation transfered (e.g. df=2 for both spins)
        Pads _existing_ and new state to max(N,a+1)
        """
        if Copy: return self.Copy().Promote(i,a,df=df,N=N,Copy=False)

        self.PadTo(max(N,a+1))
        x = self.np*1.

        df = min(df, x[i], 2-x[a])
        x[i] -= df
        x[a] += df

        return OccHelper(x)

    def PadTo(self, N=0):
        if N<=len(self.npRaw):
            self.Process()
            return self
        else:
            x = np.zeros((N,2))
            if len(self.npRaw)>0:
                x[:len(self.npRaw),:]=self.npRaw
            self.npRaw = x*1.
            self.Process()
            return self

    def Reorder(self, kk):
        self.Process()
        NNew = np.max(kk[:self.N])+1
        npNew = np.zeros((NNew,2))
        npNew[kk[:self.N],:] = self.npRaw
        self.npRaw = npNew
        self.Process()
        return self

    def Process(self):
        self.N = len(self.npRaw)
        self.npa = self.npRaw[:,0]
        self.npb = self.npRaw[:,1]
        self.np = self.npa + self.npb

        #print("a", self.npa)
        #print("b", self.npb)
        #print("T", self.np)

        # Get number of occupied orbitals overall and in alpha and beta
        self.NOcc = 0
        for k in range(self.N):
            if self.np[k] >= zero_round: self.NOcc = k+1
            if self.npa[k] >= zero_round: self.NaOcc = k+1
            if self.npb[k] >= zero_round: self.NbOcc = k+1

        # Get the core orbitals, i.e. those doubly occupied appearing 
        # before partly occupied orbitals
        self.NCore = 0
        for k in range(len(self.np)):
            if self.np[k]>2.-zero_round:
                self.NCore = k+1
            else:
                break

        return self
    
    def Swap(self,a,b):
        """ Switch occupation of a and b """
        if a==b: return
        #print("Swapping %d and %d in %s"%(a,b,self.AsStr()))
        if self.N<max(a,b)+1:
            self.PadTo(max(a,b)+1)
        Olda = self.npRaw[a,:]*1.
        Oldb = self.npRaw[b,:]*1.
        self.npRaw[a,:] = Oldb
        self.npRaw[b,:] = Olda
        self.Process()
        return self
    
    def SwapSpin(self, a):
        """ Swap the spin occupancies of a"""
        self.npRaw[a,0], self.npRaw[a,1] \
            = self.npRaw[a,1], self.npRaw[a,0]
        self.Process()
        return self
    
    def SmearSpin(self):
        self.np = self.npRaw[:,0] + self.npRaw[:,1]
        self.npa = np.minimum(self.np, 1.)
        self.npb = self.np - self.npa
        return self

    def f(self): return self.np*1.
    def fa(self): return self.npa*1.
    def fb(self): return self.npb*1.

    def F2H(self, N=None):
        if not(N is None): self.PadTo(N)
        return np.outer(self.f(),self.f())
    def F2x(self, N=None): 
        if not(N is None): self.PadTo(N)
        return -(np.outer(self.fa(), self.fa()) + np.outer(self.fb(), self.fb()))
    def F2xFDT(self, N=None):
        if not(N is None): self.PadTo(N)
        F2 = np.zeros((self.N,self.N))
        f = self.f()
        for j1 in range(self.N):
            for j2 in range(self.N):
                F2[j1,j2] = -f[max(j1,j2)]
        return F2

    def IsInt(self): return np.max(np.abs(self.np-np.round(self.np)))<zero_round
    def IsDbl(self): return np.max(self.np*(2-self.np))<zero_round

    def SameKS(self, f2):
        """Check if self represents the same total occupations as f2"""
        if isinstance(f2, OccHelper): f2 = f2.np
        f1 = self.np

        # Make sure len(f1) < len(f2)
        if len(f1)==len(f2): return max(np.abs(f1-f2)).max()
        elif len(f1)>len(f2): f1, f2 = f2, f1
        return max(np.abs(f1-f2[:len(f1)]).max(), 
                   np.abs(f2[len(f1):])).max()<zero_round
    
    def SameAs(self, f2):
        """Check if self represents the same spin occupations as f2"""
        if not(isinstance(f2, OccHelper)): f2 = OccHelper(f2)

        N1 = f1.npRaw.shape[0]
        N2 = f2.npRaw.shape[0]

        N = max(N1,N2)
        df = np.zeros((N,2))
        df[:N1,:]  = f1.npRaw
        df[:N2,:] -= f2.npRaw

        return np.max(np.abs(df))<zero_round

    def Parity(self):
        # Parity is only defined for integers
        if not(self.IsInt()): return None

        # Get the list of occupied up orbitals
        ku = np.argwhere(self.npRaw[:,0]==1).reshape((-1,))
        # Get the list of occupied down orbitals
        kd = np.argwhere(self.npRaw[:,1]==1).reshape((-1,))

        if len(ku)==0:
            N = kd.max()+1
            return QParity(kd, N)
        elif len(kd)==0:
            N = ku.max()+1
            return QParity(ku, N)

        N = max(ku.max(), kd.max()) + 1


        return QParity(ku, N)*QParity(kd, N)

    
    def OccBlocks(self, NBas=0, NFrozen=0,
                  Empty = -1, 
                  WithMap=False):
        """Get the blocks of similar orbitals
        NBas is the basis set size (defaults to # occ)
        NFrozen removes frozen from the Map (not used otherwise)
        Set Empty to -1 to use the same model for holes and virtuals
        Set Empty to -2 to use the next occupied for holes
        WithMap returns a map
        """
        self.PadTo(max(NBas,len(self.np)))
        self.Process()
        NCore = self.NCore
        NOcc = self.NOcc

        if np.sum(self.np)<zero_round:
            kk = np.array([-1]*self.N, dtype=int)
            if WithMap: return kk, { -1: np.arange(NFrozen,self.N)}
            else: return kk

        if Empty>=0 or Empty<=-1.5: Empty = -2
        else: Empty = -1

        kk = np.zeros((self.N,), dtype=int) + Empty
        if WithMap:Map = {}

        kk[:NCore] = NCore-1
        if WithMap: Map[NCore-1] = list(range(NFrozen, NCore))

        kk[NOcc:] = -1
        if WithMap: Map[-1] = list(range(NOcc,self.N))

        for k in range(NCore,NOcc):
            # Hand unoccupied later
            if self.np[k]<zero_round: continue
            kTo = k
            kk[k] = kTo
            if WithMap:
                # Note, the map is Map[kTo] = list of from so needs care
                if kTo in Map:
                    Map[kTo] = Map[kTo] + [k]
                else:
                    Map[kTo] = [k]

        for k in range(NCore,NOcc):
            if kk[k]==-2:
                for kp in range(k+1,NOcc):
                    if self.np[kp]>=zero_round:
                        kTo = kp
                        break
                if WithMap: Map[kTo] += [k]
                kk[k] = kTo

        if Empty == -1: Map[-1] = np.argwhere(kk==-1).reshape((-1,))

        # Convert map into arrays
        for k in Map:
            if len(Map[k])>0:
                Map[k] = np.array(Map[k], dtype=int)
            else:
                Map[k] = np.array([kTo], dtype=int)


        if WithMap:
            return kk, Map
        return kk

    #### Routines for helping with wavefunctions
    def D(self, C, f=None):
        """Return the 1RDM based on C
        Here, C must have enough columns to capture all occupied orbitals 
        """
        CL, CR = self.CLR(C,f)
        return (CL).dot(CR.T)

    def CLR(self,C,f=None):
        """Return CL and CR where CL is C for occupied orbitals
        and CR is C * f_k for occupied orbitals.
        Here, C must have enough columns to capture all occupied orbitals 
        
        Note, D = CL (CR.T) is then the correct 1RDM
        """
        if f is None: f = self.np

        kk = np.argwhere(f>zero_round).reshape((-1,))
        CL = C[:,kk]
        CR = np.einsum('pk,k->pk',CL,f[kk])
        return CL, CR
    
    def Cab(self,C,f=None):
        """Return Ca and Cb for alpha and beta.
        Note, if f_spin in [0,1] it returns the occupied orbitals
        otherwise, it yield sqrt(f_spin)*phi for the two spin channels
        """
        if f is None: 
            f = self.np
            fa = self.npa
            fb = self.npb
        else:
            fa = np.minimum(f,1.)
            fb = f - fa

        if np.max(np.abs(f-np.round(f)))<zero_round:
            ka = np.argwhere(np.abs(fa-1.)<zero_round).reshape((-1,))
            kb = np.argwhere(np.abs(fb-1.)<zero_round).reshape((-1,))
            return C[:,ka], C[:,kb]
        else:
            kk = np.argwhere(np.abs(f)>zero_round).reshape((-1,))
            Ca = C[:,kk]*np.sqrt(fa[kk])[None,:]
            Cb = C[:,kk]*np.sqrt(fb[kk])[None,:]
            return Ca, Cb

    
    def Dab(self,C,f=None):
        """Return Da and Db.
        Note, can handle partial occupancies but is slower because of it
        """
        if f is None: 
            f = self.np
            fa = self.npa
            fb = self.npb
        else:
            fa = np.minimum(f,1.)
            fb = f - fa

        kk = np.argwhere(f>zero_round).reshape((-1,))
        CT = C[:,kk]
        return np.einsum('pk,qk,k->pq', CT, CT, fa[kk]), np.einsum('pk,qk,k->pq', CT, CT, fb[kk])
    
    def Show(self):
        print(self.AsStr())

    def AsStr(self, Max=10, Spin=False):
        f = self.np
        if (len(f)>Max):
            f = f[:self.NOcc]

            if (len(f)>Max):
                L = len(f)-Max
                Pre, f = "2^%d;"%(L), f[-Max:]
            else: Pre = ''
        else: Pre = ''

        if self.IsInt():
            if Spin:
                OccList = []
                fu = np.round(self.npRaw[:,0])
                fd = np.round(self.npRaw[:,1])
                for k in range(len(fu)):
                    if fu[k]==1 and fd[k]==1:
                        OccList += ["2"]
                    elif fu[k]==1 and fd[k]==0:
                        OccList += ["u"]
                    elif fu[k]==0 and fd[k]==1:
                        OccList += ["d"]
                    else:
                        OccList += ["-"]

                return "["+Pre + ";".join(OccList) + "]"
            else:
                return "["+Pre + ";".join(["%d"%(int(x)) for x in np.round(f)]) + "]"
        else:
            x = np.max(np.abs(self.np-np.round(self.np)))
            M = max(3,int(np.round(np.log10(1/x))))
            Mask = "%%.%df"%(M)
            return "["+Pre + ";".join([Mask%(x) for x in f]) + "]"

#############################################################################
# Lists of occupation factors
#############################################################################
class OccList(list):
    def __init__(self, List):
        self.List = []
        self.AddList(List)

    # Handle the list as a list
    def __item__(self, k): return self.List[k]
    def __iter__(self): return self.List.__iter__()
    def __next__(self): return self.List.__next__()
    def __len__(self): return len(self.List)
    
    def __str__(self):
        return self.AsStr()
    
    def __add__(self, B):
        """ Add two lists together and merge common entries """
        A = self.Copy()
        A.AddList(B)
        return A
    
    def __mul__(self, x):
        """ Multiply just affects the weights """
        C = []
        for W, Occ in self.List:
            C += [(W*x,Occ)]
        return OccList(C)
    
    def __truediv__(self, x):
        """ Divide just affects the weights """
        return self.__mul__(1/x)
    
    def Copy(self):
        """ Return a copy of the current object """
        return OccList(self.List)

    def AddList(self, List):
        """ Add List to the current list """
        # Add the items
        for t in List:
            if len(t)==2: W, Occ = t
            elif len(t)==3: W, _, Occ = t

            self.List += [(W,Occ.Copy())]
        
        # Trim out double entries
        self.Trim()

        return self
    
    def AddItem(self, W, Occ):
        """ Add a single item to the current list """
        self.List += [(W,Occ.Copy())]
        self.Trim()
        return self

    def Trim(self):
        """ Trim the list by combining identical entries
        e.g. (1.,[1,1,0]) and (1.,[1,1])
        get merged to (2.,[1,1,0])
        """

        # Convert to a dictionary
        Dict = {}
        Occs = {}
        N = 0
        for (W,Occ) in self.List:
            # Unique key
            fu = Occ.npRaw[:Occ.NOcc,0]
            fd = Occ.npRaw[:Occ.NOcc,1]
            Key = ";".join("%.5f"%(x) for x in fu) \
                + '|' + ";".join("%.5f"%(x) for x in fd)

            Occs[Key] = Occ
            N = max(N, Occ.N)

            if Key in Dict: Dict[Key] += W
            else: Dict[Key] = W
        
        # Convert back to a list
        self.List = []
        for Key in Dict:
            W = Dict[Key]
            if np.abs(W)>=zero_round:
                self.List += [(W,Occs[Key].PadTo(N))]

        return self
    
    def Swap(self, a, b):
        """ Swap orbitals a and b in everything """
        if a==b: return
        NList = []
        for W, Occ in self.List:
            Occ.Swap(a,b)
            NList += [(W,Occ)]
        self.List = NList

    def Mean(self):
        """ Return the mean occupation from the list """
        X = OccHelper([])
        for W, Occ in self.List:
            X += Occ*W
        X.SmearSpin()
        return X
    
    def AsStr(self):
        return " & ".join("%.2f %s"%x for x in self.List)

#############################################################################
# Extra ERI terms helper
#############################################################################

class ExtrasHelper:
    def __init__(self, Extras=[], EH=None):
        """ Class for handling extra ERI terms 
        Note, WF is ignored internally """

        self.Content = {}

        if not(EH is None):
            self.FromExtrasHelper(EH)
        elif isinstance(Extras, ExtrasHelper):
            self.FromExtrasHelper(Extras)
        else:
            self.FromExtras(Extras)

    def __str__(self):
        return self.AsStr()
    
    def __mul__(self, Scale):
        New = self.Copy()
        for Kind in New.Content:
            for Pair in New.Content[Kind]:
                New.Content[Kind][Pair] *= Scale
        return New
    
    def __add__(self, Extras):
        New = self.Copy()
        New.AddExtras(Extras.ForEnergy())
        return New

    def Copy(self):
        return ExtrasHelper(EH=self)
    
    def FromExtrasHelper(self, EH):
        self.Content = {}
        for Kind in EH.Content:
            self.Content[Kind]={}
            for Pair in EH.Content[Kind]:
                self.Content[Kind][Pair] = EH.Content[Kind][Pair]
        self.Prune()
        return self

    def Any(self): return len(self.Content)>0

    def FromExtras(self, Extras, Scale=1.):
        if Extras is None: Extras = []

        self.Content = {}
        self.AddExtras(Extras, Scale=Scale)
        return self

    def AddExtras(self, Extras, Scale=1.):
        if Extras is None: Extras = []

        for t in Extras:
            if len(t)==5 : WE, _, j, k, Kind = t
            else: WE, j, k, Kind = t

            Pair = ( min(j,k), max(j,k) )
            if not(Kind) in self.Content: self.Content[Kind] = {}
            if not(Pair) in self.Content[Kind]: self.Content[Kind][Pair] = 0.

            self.Content[Kind][Pair] += WE*Scale
        self.Prune()
        return self

    def Prune(self, Content=None):
        """Setting None will do an in-place pruning and return self
        Otherwise returns the new content"""
        if Content is None: 
            Content = self.Content
            InPlace = True
        else:
            InPlace = False

        # Prune out zero element
        NewContent = {}
        for Kind in Content:
            NC = {}
            for Pair in Content[Kind]:
                WE = Content[Kind][Pair]
                if np.abs(WE)>zero_round:
                    NC[Pair] = WE

            if len(NC)>0:
                NewContent[Kind] = NC
        
        if InPlace:
            self.Content = NewContent
            return self
        else:
            return NewContent
        
    def Swap(self, a, b):
        """ Swap orbitals a and b """
        if a==b: return
        for Kind in self.Content:
            OC = self.Content[Kind]
            NC = {}
            for Pair in list(OC):
                j,k = Pair
                if j==a: j=b
                elif j==b: j=a
                if k==a: k=b
                elif k==b: k=a
                NPair = (min(j,k),max(j,k))
                NC[NPair] = OC[Pair]
            self.Content[Kind] = NC

            
    def PruneJK(self, Content=None):
        """Prune out J->K and return as new content
        This should never be done in place because it matters for Fock"""
        if Content is None: Content = self.Content

        PContent = {}
        for Kind in Content:
            for Pair in Content[Kind]:
                WE = Content[Kind][Pair]
                j1, j2 = Pair
                if j1==j2 and Kind=='K': K = 'J'
                else: K = Kind

                if not(K in PContent): PContent[K]={}
                if not(Pair in PContent[K]): PContent[K][Pair]=0.

                PContent[K][Pair] += WE

        return self.Prune(PContent)
    
    def ToEnergy(self, Scale=1.): return self.ForEnergy(self, Scale)
    def ForEnergy(self, Scale=1.):
        """Get the Extras for an energy calculation"""
        self.Prune()

        Extras = []
        for Kind in self.Content:
            for j1,j2 in self.Content[Kind]:
                WE = self.Content[Kind][(j1,j2)]
                Extras += [(WE*Scale,0.,j1,j2,Kind)]
        return Extras
    
    def ToFock(self, k, Occ=1.): return self.ForFock(self, k, Occ)
    def ForFock(self, k, Occ=1.):
        """Get the Extras for a Fock calculation with orbital k
        Scales by 1/Occ consistent with definitions
        """
        self.Prune()

        Occ = max(Occ, zero_round) # To avoid divide by zero

        Extras = []
        for Kind in self.Content:
            for j1,j2 in self.Content[Kind]:
                WE = self.Content[Kind][(j1,j2)]
                WF = WE/Occ

                if not(k==j1) and not(k==j2):
                    Extras += [(WE, 0., j1, j2, Kind)]
                if (k==j1) and not(k==j2):
                    Extras += [(WE, WF, j2, j1, Kind)]
                if (k==j2) and not(k==j1):
                    Extras += [(WE, WF, j1, j2, Kind)]
                if (k==j1) and (k==j2):
                    Extras += [(WE, 2.*WF, j1, j2, Kind)]

        NewExtras = []
        for (WE,WF,j1,j2,Kind) in Extras:
            if np.abs(WF)>=zero_round:
                NewExtras += [(0., WF, j1, j2, Kind)]
        return NewExtras
    
    def FromF2(self, F2):
        """ Generate the Extras from a set of pair-occupation matrices
        Here, F2 = {'J':FJ, 'K':FK, 'K_w':FK_w } (not all terms are needed)
        is a dictionary of N x N matrices and non-zero terms
        are converted to Extras
        """
        Extras = []
        for Kind in F2:
            for j1 in range(F2[Kind].shape[0]):
                for j2 in range(j1, F2[Kind].shape[0]):
                    if j1==j2: W = F2[Kind][j1,j2]
                    else: W = F2[Kind][j1,j2] + F2[Kind][j2,j1]
                    if np.abs(W)>=zero_round: Extras += [(W,j1,j2,Kind)]
        return Extras
    
    def AddFromF2(self, F2):
        """ Like F2 but adds the extra terms into the current terms """
        self.Extras = self.Extras + self.FromF2(F2)
        self.Prune()
        return self
    
    def AddToF2(self, F2):
        """ Given a set of matrices (like in FromF2) add the
        extra contributions from self
        Adds them symmetrically.
        """
        F2A = { Kind:F2[Kind]*1. for Kind in F2 }
        for Kind in self.Content:
            for j1,j2 in self.Content[Kind]:
                WE = self.Content[Kind][(j1,j2)]
                if not(Kind in F2A): continue

                if j1==j2: F2A[Kind][j1,j1] += WE
                else:
                    F2A[Kind][j1,j2] += WE/2.
                    F2A[Kind][j2,j1] += WE/2.
        return F2A
    
    def AsStr(self, PruneJK=True):
        Str = ""

        if PruneJK:
            PContent = self.PruneJK()     
        else:
            PContent = self.Content

        for Kind in ('J', 'K', 'K_w'):
            if not(Kind in PContent): continue
            
            for Pair in PContent[Kind]:
                j1, j2 = Pair
                Pre = np.round(PContent[Kind][Pair]*1000)/1000
                if Pre==1.: PreStr = " + "
                elif Pre==-1.: PreStr = " - "
                elif Pre>0. and Pre==np.round(Pre): PreStr = " + %d"%(int(Pre))
                elif Pre<0. and Pre==np.round(Pre): PreStr = " - %d"%(int(-Pre))
                elif Pre>0.: PreStr = " + %.3f"%(Pre)
                elif Pre<0.: PreStr = " - %.3f"%(-Pre)
                else: PreStr='+ eta'

                if Kind=='J': Str += PreStr + "[%d %d|%d %d]"%(j1,j1,j2,j2) 
                elif Kind=='K': Str += PreStr + "[%d %d|%d %d]"%(j1,j2,j2,j1) 
                elif Kind=='K_w': Str += PreStr + "[%d %d|%d %d]_rs"%(j1,j2,j2,j1) 

        if Str[:3]==" + ": return Str[3:]
        elif Str[:3] == " - ": return '- '+Str[3:]   
        else: return Str

#############################################################################
# The PlanHandler class automates some aspects of ensemble
# generation.
#############################################################################
class PlanHandler(dict):
    """The PlanHandler class automates some aspects of ensemble generation."""
    
    def __init__(self, Content=None, f=[],
                 Report = -1, xi = 0.,
                 epsilon = None,
                 **kwargs,
                 ):
        # Set the internal parameters
        self.xi = xi # Set the DDC parameter
        self.Report = Report

        # Set some internal variables to null defaults
        self.Recipes = None

        if not(epsilon is None):
            self.InitOrdered(epsilon, Content=Content, f=f, Report=Report, xi=xi, **kwargs)
            return

        if (Content is None):
            if len(f)>0:
                self.FromContent({'Hx':[(1.,f)],})
            else:
                self.NullPlan()
        else:            
            self.FromContent(Content)

##### Operator overrides

    def __str__(self):
        return self.AsStr()    

    def __getitem__(self, ID):
        if ID in self.Content: return self.Content[ID]
        else: return None

    def __setitem__(self, ID, value):
        if ID in self.Content: self.Content[ID] = value
        else: return None

    def __mul__(self, Scale):
        New = self.Copy()
        New.Content['1RDM']=New.Content['1RDM']*Scale
        for Q in ('Hx', 'xcDFA'):
            New.Content[Q] = New.Content[Q]*Scale
        New.Content['Extra'] = self.Content['Extra']*Scale
        New.Content['Extra_no_DDC'] = self.Content['Extra_no_DDC']*Scale
        return New
    
    def __rmul__(self, Scale):
        New = self.Copy()
        New.Content['1RDM']=New.Content['1RDM']*Scale
        for Q in ('Hx', 'xcDFA'):
            New.Content[Q] = New.Content[Q]*Scale
        New.Content['Extra'] = self.Content['Extra']*Scale
        New.Content['Extra_no_DDC'] = self.Content['Extra_no_DDC']*Scale
        return New
    
    def __add__(self, B):
        New = self.Copy()
        New.AddContent(B.Content)
        return New

##### Main routines
    def Copy(self):
        return PlanHandler(Content=self.Content,
                           xi=self.xi, Report=self.Report)
    
    def NullPlan(self):
        self.Content = None

    def InitOrdered(self, epsilon,
                    Content=None, f=[], **kwargs):
        kSort = np.argsort(epsilon)
        kUnsort = np.argsort(kSort)

        if (Content is None):
            if len(f)>0:
                Content = {'Hx':[(1.,OccHelper(f))],}
            else:
                return self.NullPlan()
        else:
            print("Not yet implemented for full content - sorry! Results may be bad")

        # Sort everything
        if 'Hx' in Content:
            NMax = 0
            N = []
            for W, f in Content['Hx']:
                f = OccHelper(f)
                f.Reorder(kSort)
                NMax = max(NMax, f.N)
                N += [(W,f)]
            Content['Hx'] = N
            Content['AutoExtra'] = True
        else:
            print("Must specify Hx at least - quitting!")
            quit()

        # Create on sorted
        self.FromContent(Content)
        
        # Unsort everything
        # First build a list of swaps
        Swaps = GetSwaps(kUnsort)
        for a,b in Swaps:
            self.Swap(a,b)

        return self


    def AddContent(self, B):
        A = self.Content

        C = {}
        C['Singlet'] = A['Singlet'] and B['Singlet']
        for Q in ('Hx', 'xcDFA'):
            C[Q] = A[Q] + B[Q]
        C['1RDM'] = C['Hx'].Mean()
        C['Extra'] = A['Extra'] + B['Extra']
        C['Extra_no_DDC'] = A['Extra_no_DDC'] + B['Extra_no_DDC']

        kTo = max(max(A['kTo']), max(B['kTo']))

        f = C['1RDM'].f()
        while kTo>=len(f) or f[kTo]<zero_round:
            kTo -= 1
        C['kTo'] = (kTo,)

        self.Content = C
        return self
    
    # Create from a occ (light feature)
    def FromOcc(self, f):
        return self.FromContent({'Hx':[(1,OccHelper(f))]})
    
    def FromContent(self, Content):
        self.Content = {}

        if not('Hx' in Content):
            print("Must specify at least Hx")
            quit()

        self.Content['Hx'] = OccList(Content['Hx'])
        f_Hx = self.Content['Hx'].Mean()

        if '1RDM' in Content: self.Content['1RDM'] = OccHelper(Content['1RDM'])
        else: self.Content['1RDM'] = f_Hx

        if 'Singlet' in Content: self.Content['Singlet'] = Content['Singlet']
        else: self.Content['Singlet'] = np.max(f_Hx.np*(2-f_Hx.np))<zero_round

        if 'kTo' in Content: self.Content['kTo'] = Content['kTo']
        else: self.Content['kTo'] = (f_Hx.NOcc-1,)

        if 'xcDFA' in Content: self.Content['xcDFA'] = OccList(Content['xcDFA'])
        else: self.Content['xcDFA'] = self.Content['Hx'].Copy()

        if 'Extra_no_DDC' in Content:
            # When adding things together we take the non-DDC extras
            self.Content['Extra'] = Content['Extra'].Copy()
            self.Content['Extra_no_DDC'] = Content['Extra_no_DDC'].Copy()
        else:
            if 'Extra' in Content:
                self.Content['Extra'] = ExtrasHelper(Content['Extra'])
            else:
                self.Content['Extra'] = ExtrasHelper([])

            if 'AutoExtra' in Content and not(Content['AutoExtra'] in (False, 'f', 'F', 'false', 'False')):
                self.Content['Extra'].AddExtras(self.Hx_to_Extra())

            # The extra terms without a DDC correction
            self.Content['Extra_no_DDC'] = self.Content['Extra'].Copy()

            self.ApplyDDC()

        return self
    
    def Swap(self, a,b):
        """ Swap orbitals a and b """
        if a==b: return

        # kTo needs to be handled here
        kTo = list(self.Content['kTo'])
        NkTo = list(kTo)
        if a in kTo: NkTo[kTo.index(a)]=b
        if b in kTo: NkTo[kTo.index(b)]=a
        self.Content['kTo'] = tuple(NkTo)

        # The other terms already have their own swap routines
        self.Content['Hx'].Swap(a,b)
        self.Content['1RDM'] = self.Content['Hx'].Mean()
        self.Content['xcDFA'].Swap(a,b)
        self.Content['Extra'].Swap(a,b)
        self.Content['Extra_no_DDC'].Swap(a,b)
    
    ##### Special initalisations
    def Polarised(self, **kwargs): return self.Polarized(**kwargs)
    def Polarized(self, Na=1, Nb=1,
                  f = None,
                  **kwargs):
        """ Return a polarized ground state with Na and Nb occupied (polarised works too)
        f can also be directly specfied, in which case it is just a 'ground state' with
        whatever occupations you feed it
        """
        if f is None:
            ND = min(Na,Nb)
            NS = max(Na,Nb)-ND
            Occ = OccHelper([2]*ND+[1]*NS)
        else:
            Occ = OccHelper(f)
        return self.FromContent({'Hx':[(1.,Occ),], })
    
    def Triplet(self, NEl, From=[], To=[], 
                Auto=True, # Set false to ignore auto setup (not recommended)
                **kwargs):
        """ Return a triplet ground state with NEl
        From, To say which orbitals to promote and no testing is done so be careful
        Can also generate quintuplets
        """
        if not(NEl%2)==0:
            print("Triplet must have even electron number not %d"%(NEl))
            quit()

        Occ = OccHelper([2]*(NEl//2))

        kTo = 0
        for i,a in zip(From, To):
            Occ = Occ.Promote(i,a)
            kTo = max(kTo,a)
        return self.FromContent({'kTo':(kTo,), 'Hx':[(1.,Occ)],
                                 'AutoExtra':Auto, })
    
    def Doublet(self, NEl, From=[], To=[],
                Auto=True, # Set false to ignore auto setup (not recommended)
                **kwargs):
        Na, Nb = (NEl+1)//2, (NEl-1)//2
        Occ = OccHelper([2]*Nb + [1]*(Na-Nb))

        kTo = 0
        for i,a in zip(From, To):
            Occ = Occ.Promote(i,a)
            kTo = max(kTo,a)
        return self.FromContent({'kTo':(kTo,), 'Hx':[(1.,Occ)],
                                 'AutoExtra':Auto, })


    def Singlet(self, NEl, From=[], To=[],
                  **kwargs):
        """ Return a triplet ground state with NEl
        From, To say which orbitals to promote and bad promotions will not work
        """
        if not(NEl%2)==0:
            print("Singlet must have even electron number not %d"%(NEl))
            quit()
        
        Occ = OccHelper([2]*(NEl//2))

        if len(From)==0:
            # No excitations - ground state
            return self.FromContent({'Hx':[(1.,Occ)]})
        elif len(From)==1 and len(To)==1:
            # Single excitation
            i, a = From[0], To[0]
            Occ = Occ.Promote(i,a,Copy=True)
            return self.FromContent({'kTo':(a,), 'Singlet':True,
                                     'Hx':[(1.,Occ)],
                                     'Extra':[(2., i, a, 'K')],
                                     'AutoExtra':True,})
        elif len(From)==2 and len(To)==2:
            # Double excitation (different modes)
            i1, a1 = From[0], To[0]
            i2, a2 = From[1], To[1]
            if (i1==i2) and (a1==a2):  # i^2 -> a^2
                Occ_ts = Occ.Promote(i1,a1,Copy=True)
                return self.FromContent({'kTo':(a1,), 'Singlet':True,
                                         'Hx':[(2.,Occ_ts),(-1.,Occ)],
                                         'AutoExtra':True,})
            elif (i1==i2):  # i^2 -> a1 a2
                Occ_ts1 = Occ.Promote(i1,a1,Copy=True)
                Occ_ts2 = Occ.Promote(i1,a2,Copy=True)
                return self.FromContent({'kTo':(a1,a2), 'Singlet':True,
                                         'Hx':[(1.,Occ_ts1),(1.,Occ_ts2),(-1.,Occ)],
                                         'Extra':[(2., a1, a2, 'K')],
                                         'AutoExtra':True,})
            elif (a1==a2):  # i1 a2 -> a^2
                Occ_ts1 = Occ.Promote(i1,a1,Copy=True)
                Occ_ts2 = Occ.Promote(i2,a1,Copy=True)
                return self.FromContent({'kTo':(a1,), 'Singlet':True,
                                         'Hx':[(1.,Occ_ts1),(1.,Occ_ts2),(-1.,Occ)],
                                         'Extra':[(2., i1, i2, 'K')],
                                         'AutoExtra':True,})
            else:  # Double split i1 i2 -> a1 a2
                print("Double split double excitation not implemented")
                quit()
        else:
            # Triple or more
            print("Triple+ excitation not implemented")
            quit()

    def Convert_to_CDFA(self):
        f = self['1RDM'].np
        Omega = f - np.hstack((f[1:],0.))
        xc = []
        for k in range(len(f)):
            if np.abs(Omega[k])>1e-5:
                Occ = OccHelper(np.ones((k+1,2)))
                xc += [(Omega[k]/2., Occ)]
        self['xcDFA'] = OccList(xc)
        return self

    def Hx_to_Extra(self):
        """
        Auto-evaluate most of the ensemble Hartree correction
        1) Calculate Hx from the excited state and then subtract
        the weighted average of Hx from the list
        2) Convert this into an extras
        """

        # Treat tht total RDM as a pure state
        F2 = {}
        Occ = self.Content['1RDM']
        F2['J'] = 0.5*Occ.F2H()
        F2['K'] = 0.5*Occ.F2x()
        NT = Occ.N

        # Subtract out the weighted average of naive Hartree and FDT exchange
        for W, Occ in self.Content['Hx']:
            F2['J'] -= 0.5*W*Occ.F2H(NT)
            F2['K'] -= 0.5*W*Occ.F2xFDT(NT)

        # What is left is the FDT Hartree extras
        Extras = ExtrasHelper().FromF2(F2)

        return Extras
    
    def ApplyDDC(self, xi = None):
        """
        Auto-evaluate the DDC correction
        1) Calculate H from the list and add extras
        2) Subtract the classical H
        3) DDC is -xi * Difference
        """
        if xi is None: xi = self.xi

        F2 = {}
        F2_SCE = {}

        # Get the classical Hartree
        Occ = self.Content['1RDM']
        NT = Occ.N
        Z = np.zeros((NT,NT), dtype=float)
        F2_SCE['J'] = 0.5*Occ.F2H()
        F2_SCE['K'] = Z*0.

        # Get the ensemble Hartree
        F2 = { 'J': Z*0., 'K': Z*0. }
        for W, Occ in self.Content['Hx']:
            F2['J'] += W*0.5*Occ.F2H(NT)
        DF2 = self.Content['Extra_no_DDC'].AddToF2(F2)

        # Multiply the difference by -xi
        DF2 = {Kind:-xi*(DF2[Kind]-F2_SCE[Kind]) for Kind in DF2}

        # Convert to extras and add to the existing extras
        DDCExtras = ExtrasHelper().FromF2(DF2)

        # Add to the existing Hx Extras passed at creation
        self.Content['Extra'] = self.Content['Extra_no_DDC'].Copy().AddExtras(DDCExtras)

    ##### Helper routines for energy calculators
    def GenerateFockRecipes(self, NOrb = 0,  NBas=0):
        """
        Generate recipes for evaluating energies and Fock matrices
        for different orbitals.

        Must specify NOrb=# basis functions to identify what
        we optimize on.

        NOrb is the number of orbitals (defaults to NBas)
        """

        if NOrb==0: NOrb = NBas

        Occ = self.Content['1RDM']
        f = Occ.f()*1.

        self.Recipes = {}
        
        # Get the occupation blocks from the 1RDM
        kk, kMap = Occ.OccBlocks(NBas=NOrb, WithMap=True)
        
        # Make a list of unique k by taking the blocks + kTo elements
        kUnique = list(self.Content['kTo'])
        for k in list(kMap):
            if k>=0: kUnique += [k]

        # Unique only via a set
        kUnique = sorted(list(set(kUnique)))

        # Deal with the future case that 'H' and 'x' are separated
        if 'H' in self.Content:
            Keys = ('H', 'x', 'xcDFA')
        else:
            Keys = ('Hx', 'xcDFA')
        
        # Initilaise the Recipes
        self.Recipes['Map'] = kMap
        self.Recipes['FockOccs'] = { Q:[] for Q in Keys }
        self.Recipes['FockWeights'] = {}
        self.Recipes['FockExtra'] = {}

        for k in kUnique:
            self.Recipes['FockWeights'][k] = {}
            for Q in Keys:
                N = len(self.Content[Q])
                self.Recipes['FockWeights'][k][Q] = [(0.,0.,0.)]*N

        # Process the Hx and xcDFA terms by using the functional chain
        # rule applied to spin-1RDMs

        for Q in Keys:
            for K, (W, Occ) in enumerate(self.Content[Q]):
                self.Recipes['FockOccs'][Q] += [Occ]

                for k in kUnique:
                    if f[k]==0.: continue
                    if not(k in self.Recipes['FockWeights']): 
                        self.Recipes['FockWeights'][k] = { 'Hx':[], 'xcDFA':[], }

                    ta = safeocc(Occ.fa(), k)/max(f[k], zero_round)
                    tb = safeocc(Occ.fb(), k)/max(f[k], zero_round)
                    self.Recipes['FockWeights'][k][Q][K] = (W,W*ta,W*tb)

        # The Extras Helper implements handling for extras
        for k in kUnique:
            self.Recipes['FockExtra'][k] = self.Content['Extra'].ForFock(k, f[k])

        # Virtual orbitals use kTo if it's defined, otherwise the highest defined
        if len(self.Content['kTo'])>0:
            self.Recipes['Virtual'] = self.Content['kTo']
        else:
            self.Recipes['Virtual'] = (max(kUnique),)
        
        return self.Recipes

    def ExtraList(self):
        """ Get the Extras stuff as a lsit for energy calculations """
        return self.Content['Extra'].ForEnergy()

    def GetFJK(self):
        """ Return matrices (non-zero elements only) of total J and K
        weights including Hx and extras terms"""

        f_avg = self.Content['1RDM'].f()
        N = len(f_avg)
        FJ  = np.zeros((N,N), dtype='float')
        FK  = np.zeros((N,N), dtype='float')
        FKx = np.zeros((N,N), dtype='float')
        for W, Occ in self.Content['Hx']:
            fa = Occ.fa()
            fb = Occ.fb()
            f  = Occ.f()
            FJ[:len(f),:len(f)] += 0.5*W*np.outer(f, f)
            FK[:len(fa),:len(fa)] -= 0.5*W*np.outer(fa, fa)
            FK[:len(fb),:len(fb)] -= 0.5*W*np.outer(fb, fb)


        for W,_,j,k,Kind in self.Content['Extra_no_DDC'].ForEnergy():
            if Kind=='J':
                FJ[j,k]+=W/2.
                FJ[k,j]+=W/2.
            else:
                FK[j,k]+=W/2.
                FK[k,j]+=W/2.

        for j in range(N):
            for k in range(N):
                FKx[j,k] = -f_avg[max(j,k)]/2

        return FJ, FK, FKx
               

    def AsStr(self):
        """ Show the Plan as a string """
        if self.Content is None:
            return("Null plan")
        
        if self.xi==0.: xiStr = ""
        else: xiStr = ", xi=%.2f"%(self.xi)

        Str = ["kTo = %s, Singlet = %s%s, f = %s"\
            %(str(self.Content['kTo']), str(self.Content['Singlet'])[0],
              xiStr,
              self.Content['1RDM'].AsStr() ) ]
        for Key in ('Hx', 'xcDFA'):
            Str += [ "%-5s : %s"%(Key, self.Content[Key]) ]
            #Occs = " & ".join(["%.3f %s"%E for E in self.Content[Key]])
            #Str += [ '%-5s : %s'%(Key, Occs) ]

        if not(self.Content['Extra'] is None) and self.Content['Extra'].Any():
            Str += [ "Extra : %s"%(self.Content['Extra']) ]
        
        return "\n".join(Str)
    
class CSFHandler(PlanHandler):
    # Convert a CSF - expressed as a list of:
    #     [ (C1, Occ1), (C2, Occ2), ... ]
    # into a plan
    # Note, will normalise if required

    def __init__(self, CSF=[(1., OccHelper(np.ones((3,2))))],
                 Report = -1, xi = 0.,
                 epsilon = None,
                 **kwargs,
                 ):
        # Set the internal parameters
        self.xi = xi # Set the DDC parameter
        self.Report = Report

        # Set some internal variables to null defaults
        self.Recipes = None

        self.SetCSF(CSF)
        self.EvalHx(**kwargs)

    def SetCSF(self, CSF):
        # Get the maximum number of orbitals
        self.NOrb = 0
        for C, Occ_ in CSF:
            Occ = OccHelper(Occ_)
            self.NOrb = max(self.NOrb, Occ.npRaw.shape[0])

        # Condense duplicates
        Done = []
        Dupes = {}
        for J in range(len(CSF)):
            if (J in Done): continue

            Dupes[J] = [J]
            for K in range(J+1,len(CSF)):
                if (K in Done): continue

                Diff, _ = self.Diff(OccHelper(CSF[J][1]), OccHelper(CSF[K][1]))
                if Diff==0:
                    Dupes[J] += [K]
                    Done += [K]

        # Normalize it
        Norm = 0.
        for J in sorted(list(Dupes)):
            C = np.sum([CSF[K][0] for K in Dupes[J]])
            Norm += C**2

        # Store internally as [(C, Occ, Parity(Occ)), ...]
        self.CSF = []
        for J in sorted(list(Dupes)):
            C = np.sum([CSF[K][0] for K in Dupes[J]])/np.sqrt(Norm)
            Occ = OccHelper(CSF[J][1])
            self.CSF += [(C, Occ, Occ.Parity())]
        



    def Diff(self, Occ1, Occ2):
        # Returns number of differences (3 for more than 2) and info
        # E.g. info is nothing if 0 or 3 differences but
        f1 = np.zeros((self.NOrb,2))
        f2 = np.zeros((self.NOrb,2))
        f1[:len(Occ1.npRaw),:] = np.round(Occ1.npRaw)
        f2[:len(Occ2.npRaw),:] = np.round(Occ2.npRaw)
        df = f1 - f2

        Delta = np.sum(np.abs(df))

        # Check if they're the same
        if Delta==0.: return 0, ()

        # Check if they differ by spin
        for s in (0,1):
            DNs = np.sum(f1[:,s])-np.sum(f2[:,s])
            if not(DNs)==0.: return 3, ()

        # Single difference
        if Delta==2.:
            kf = np.argwhere(df==-1.)
            kt = np.argwhere(df== 1.)
            return 1, (kf, kt)
        elif Delta==4.:
            kf = np.argwhere(df==-1.)
            kt = np.argwhere(df== 1.)

            # Ensure the spins align
            if not(kf[0,1]==kt[0,1]):
                kt = kt[[1,0],:]
            
            # Return the delta
            return 2, (kf, kt)
        else:
            return 3, ()


    def EvalHx(self, CSF1=None, CSF2=None,
               Simplified = True, HxPolarised=True,
               eta = 1e-6, Polarised=False,
               **kwargs):
        if CSF1 is None: CSF1 = self.CSF
        if CSF2 is None: CSF2 = self.CSF

        fRDM =np.zeros((self.NOrb,))
        F2J = np.zeros((self.NOrb, self.NOrb))
        F2K = np.zeros((self.NOrb, self.NOrb))
        Extra = []

        Hx_Direct = []
        for C1, Occ1, P1 in CSF1:
            for C2, Occ2, P2 in CSF2:
                D, Info = self.Diff(Occ1, Occ2)
                if D>=3: continue

                W = C1*C2
                f1 = np.zeros((self.NOrb,2))
                f2 = np.zeros((self.NOrb,2))
                f1[:len(Occ1.npRaw),:] = np.round(Occ1.npRaw)
                f2[:len(Occ2.npRaw),:] = np.round(Occ2.npRaw)

                if D==0:
                    if HxPolarised:
                        ff = f1*0.
                        ft = f1.sum(axis=1)
                        fu = np.minimum(ft,1.)
                        fd = ft - fu
                        ff[:,0] = fu
                        ff[:,1] = fd
                    else:
                        ff = f1
                    Hx_Direct += [(W,OccHelper(ff))]
                    F2J += W * np.outer(ff[:,0]+ff[:,1], ff[:,0]+ff[:,1])
                    F2K -= W * (np.outer(ff[:,0],ff[:,0]) + np.outer(ff[:,1],ff[:,1]))
                    fRDM += W * (ff[:,0]+ff[:,1])
                elif D==1:
                    print("Single promotion not yet implemented - results are wrong")
                elif D==2:
                    kf, kt = Info
                    kp, sp = tuple(kf[0,:])
                    kq, sq = tuple(kf[1,:])
                    kr, sr = tuple(kt[0,:])
                    ks, ss = tuple(kt[1,:])

                    if sp==sq: # All same spin
                        print("Same spin double promotion not yet implemented - results are wrong")
                    else: # Different spin
                        l = (min(kp,kq),max(kp,kq))
                        r = (min(kr,ks),max(kr,ks))
                        if l==r:
                            F2K[l[0],l[1]] += 2*W*P1*P2
                            F2K[l[1],l[0]] += 2*W*P1*P2
                        elif l[0]==l[1] and r[0]==r[1]:
                            F2J[l[0],r[0]] += 2*W*P1*P2
                            F2J[r[0],l[0]] += 2*W*P1*P2
                        else:
                            Extra += [(2*W*P1*P2, l[0], l[1], r[0], r[1])]

        # Calculate the spin-resolved RDM
        fRDMu = np.minimum(fRDM, 1)
        fRDMd = fRDM - fRDMu

        fRDMs = np.zeros((self.NOrb,2))
        fRDMs[:,0] = fRDMu
        fRDMs[:,1] = fRDMd

        # Calculate the frontier orbital
        kh = 0
        for k in range(self.NOrb):
            if fRDM[k]>0.: kh = k

        # Calculate the Hx and DFT contribution
        xcDFA = []
        F2J_MF = 0.*F2J
        F2K_MF = 0.*F2K

        for k in range(self.NOrb):
            if k==(self.NOrb-1): Omega = fRDM[k]
            else: Omega = fRDM[k] - fRDM[k+1]

            if np.abs(Omega)>eta:
                f_k = np.ones((k+1,2), dtype=float)
                if not(Polarised):
                    W_k = Omega/2.
                else:
                    W_k = Omega
                    f_k[:,1] = 0.

                Occ_k = OccHelper(f_k)
                xcDFA += [ (W_k, Occ_k) ]

                F2J_MF[:(k+1),:(k+1)] += W_k*Occ_k.F2H() # np.outer(f_k[:,0]+f_k[:,1], f_k[:,0]+f_k[:,1])
                F2K_MF[:(k+1),:(k+1)] += W_k*Occ_k.F2x() # np.einsum('js,ks->jk',f_k, f_k)

        if not(Simplified):
            Hx = xcDFA
        else:
            Hx = Hx_Direct
            F2J_MF = 0.*F2J
            F2K_MF = 0.*F2K
            for W, f in Hx_Direct:
                F2J_MF += W*f.F2H()
                F2K_MF += W*f.F2x()


        # Calculate the extra components (i.e. beyond RDM)
        self.DF2 = { 'J': (F2J - F2J_MF)/2, 'K': (F2K - F2K_MF)/2 }

        Extra = []
        for j in range(self.NOrb):
            for k in range(j, self.NOrb):
                if np.abs(self.DF2['J'][j,k])>eta: Extra += [ (2*self.DF2['J'][j,k], j, k, 'J')]
                if np.abs(self.DF2['K'][j,k])>eta: Extra += [ (2*self.DF2['K'][j,k], j, k, 'K')]

        Occ = OccHelper(fRDMs)
        self.DF2['J'] = 0.5*(F2J -Occ.F2H())

        self.FromContent({
            'kTo':(kh,), 'Singlet':True,
            '1RDM': Occ,
            'Hx': Hx,
            'xcDFA': xcDFA,
            'Extra': Extra,
            'AutoExtra':False,
            })

        self.ApplyDDC()

    def ApplyDDC(self, xi=None):
        if xi is None: xi = self.xi

        F2 = {}
        F2_SCE = {}

        # Get the classical Hartree
        Occ = self.Content['1RDM']
        NT = Occ.N
        Z = np.zeros((NT,NT), dtype=float)
        F2_SCE['J'] = 0.5*Occ.F2H()
        F2_SCE['K'] = Z*0.

        # Multiply the difference of ensemble Hartree and regular by -xi
        DF2 = {Kind:-xi*self.DF2[Kind] for Kind in self.DF2}

        # Convert to extras and add to the existing extras
        DDCExtras = ExtrasHelper().FromF2(DF2)

        # Add to the existing Hx Extras passed at creation
        self.Content['Extra'] = self.Content['Extra_no_DDC'].Copy().AddExtras(DDCExtras)


class ThermalHandler:
    def __init__(self, N0=2., kbT=0.01, fCut=1e-6, OmegaCut=1e-6):
        self.SetN(N0)
        self.SetTemp(kbT)
        self.SetfCut(fCut, OmegaCut)

    def SetN(self, N0):
        # N0 can be a total electron number or electrons per spin
        try:
            self.Na, self.Nb = tuple(N0)
        except:
            self.Na, self.Nb = N0/2, N0/2

    def SetTemp(self, kbT):
        self.kbT = kbT

    def SetfCut(self, fCut, OmegaCut=None):
        if OmegaCut is None: OmegaCut = fCut
        self.fCut = fCut
        self.OmegaCut = OmegaCut

    def CutOcc(self, f_):
        N_ = np.sum(f_)
        f = f_[f_>=self.fCut]
        N = np.sum(f)
        dN = np.sum(f*(1-f))
        if dN>0.:
            f += (N_-N)/dN * f*(1-f)
        return f

    def GetOcc(self, epsilon, N0=None, eta_dN = 1e-6, eta_mu = 1e-5):
        # Automatic occupations
        if N0 is None:
            if self.Na==self.Nb:
                return 2.*self.GetOcc(epsilon, N0=self.Na)
            else:
                return self.GetOcc(epsilon, N0=self.Na) + self.GetOcc(epsilon, N0=self.Nb)
            
        epsilon = np.array(epsilon)
        
        xm = 3.*np.abs(np.log(1/self.fCut)) # Threshold for x
        def QuickN(mu, f_only=False):
            x = (epsilon-mu)/self.kbT
            x = np.minimum(x, xm)
            f = 1/(1. + np.exp(x))
            if f_only: return f

            return np.sum(f)-N0
            
        # Initialise window to floor and ceiling
        ku = max(int(np.ceil(N0))-1, 1)
        kd = ku-1
        mu0 = (epsilon[ku]+epsilon[kd])/2
        dmu = max(np.abs(epsilon[ku]-epsilon[kd]), 0.1)
        
        mu = [ mu0-dmu, mu0, mu0+dmu ]

        dN = [QuickN(mu_) for mu_ in mu]
        # Ensure the window changes sign
        Huh = 0
        while (dN[0]*dN[2])>0:
            dmu *= 2.
            mu = [ mu0-dmu, mu0, mu0+dmu ]
            dN = [QuickN(mu_) for mu_ in mu]
            Huh +=1
            if Huh>10:
                print("Could not find mu")
                quit()

        Huh = 0
        while (np.min(np.abs(dN))>eta_dN) and (mu[2]-mu[0])>eta_mu:
            if dN[0]*dN[1]<0:
                mud, muu = mu[0], mu[1]
                dNd, dNu = dN[0], dN[1]
            else:
                mud, muu = mu[1], mu[2]
                dNd, dNu = dN[1], dN[2]

            mu = [mud, (mud+muu)/2  , muu]
            dN = [dNd, QuickN(mu[1]), dNu]

            Huh += 1
            if Huh>50:
                print("Could not find mu")
                quit()

        kOpt = np.argmin(np.abs(dN))

        f = QuickN(mu[kOpt], f_only=True)

        return self.CutOcc(f)
    
    def FormPlan(self, epsilon, xi=0., OmegaCut=None):
        if OmegaCut is None: OmegaCut = self.OmegaCut

        # Get the occupation factos
        fAll = self.GetOcc(epsilon)

        # Get the number of occupied orbitals
        NOcc = 1
        for k in range(len(fAll)):
            if fAll[k]>=self.fCut: NOcc = k+1

        self.fAll = fAll
        self.NOcc = NOcc
        self.fOcc = fAll[:NOcc]
        self.epsilonOcc = epsilon[:NOcc]

        f = fAll[:NOcc]

        # Values of Omega
        Omega = f - np.hstack((f[1:],0.))

        HxList = []
        if OmegaCut==0.:
            for k in range(len(Omega)):
                f_k = OccHelper( np.ones((k+1,2)) )
                HxList += [ (Omega[k]/2., f_k)]
        else:
            # Find the maximum with a realistic contribution
            km = 0
            for k, om in enumerate(Omega):
                if np.abs(om)>OmegaCut: km = k
            #print(len(Omega), km, Omega, OmegaCut)

            for k in range(len(Omega)):
                # Trim anything but the last point on low Omega
                if (k==km) or np.abs(Omega[k])>OmegaCut:
                    #print("k = %2d, Omega = %9.6f"%(k, Omega[k]))
                    f_k = OccHelper( np.ones((k+1,2)) )
                    HxList += [ (Omega[k]/2., f_k)]
        
        QuickPlan = {
            '1RDM': OccHelper(fAll),
            'kTo': (NOcc-1,), 'Singlet': True,
            'Hx': HxList,
            'AutoExtra': True,
        }

        return PlanHandler(Content=QuickPlan, xi = xi)

if __name__ == "__main__":
    if False:
        print('='*72)
        print(CSFHandler([ (1., [[1,0],[1,0]])] ))

        print('='*72)
        P1 = [[1,1],[1,0],[0,1]]
        P2 = [[1,1],[0,1],[1,0]]
        print(CSFHandler([ (1., P1), (1., P2) ]))

        print('='*72)
        P1 = [[1,1]]
        P2 = [[0,0],[1,1]]
        P3 = [[0,0],[0,0],[1,1]]
        print(CSFHandler([ (1., P1), (1., P2), (1., P3) ]))

    TH = ThermalHandler(N0=[2,2], kbT = 0.01, fCut = 1e-3)
    Plan = TH.FormPlan([-0.1, -0.05, -0.02, -0.01, 0.01, 0.02, 0.05])
    print(Plan)
