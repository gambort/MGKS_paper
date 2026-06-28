__version__ = 1.0
__author__ = "Tim Gould"

import psi4
import numpy as np
import scipy.linalg as la

eV = 27.211
zero_round = 1e-5

#################################################################################################
# This code handles degeneracies detected and used by psi
#################################################################################################

class SymHelper:
    def __init__(self, wfn):
        self.NSym = wfn.nirrep()
        self.NBasis = wfn.nmo()

        self.eps_so = wfn.epsilon_a().to_array()
        self.C_so = wfn.Ca().to_array()
        self.ao_to_so = wfn.aotoso().to_array()

        if self.NSym>1:
            self.eps_all = np.hstack(self.eps_so)
            self.k_all = np.hstack([ np.arange(len(self.eps_so[s]), dtype=int)
                                       for s in range(self.NSym)])
            self.s_all = np.hstack([ s * np.ones((len(self.eps_so[s]),), dtype=int)
                                       for s in range(self.NSym)])
        else:
            self.eps_all = self.eps_so * 1.
            self.k_all = np.array(range(len(self.eps_all)))
            self.s_all = np.zeros((len(self.eps_all),), dtype=int)

        self.ii_sorted = np.argsort(self.eps_all)
        self.eps_sorted = self.eps_all[self.ii_sorted]
        self.k_sorted = self.k_all[self.ii_sorted]
        self.s_sorted = self.s_all[self.ii_sorted]

        self.ks_map = {}
        for q in range(len(self.ii_sorted)):
            self.ks_map[(self.s_sorted[q], self.k_sorted[q])] = q

    # Do a symmetry report to help identifying orbitals
    def SymReport(self, kh, eta=zero_round):
        epsh = self.eps_sorted[kh] + eta
        print("Orbital indices by symmetry - | indicates virtual:")
        for s in range(self.NSym):
            Str = "Sym%02d : "%(s)
            eps = self.eps_so[s]
            if not(hasattr(eps, '__len__')) or len(eps)==0: continue

            kk_occ = []
            kk_unocc = []
            for k, e in enumerate(eps):
                if e<epsh: kk_occ += [ self.ks_map[(s,k)] ]
                else: kk_unocc += [ self.ks_map[(s,k)] ]

            Arr = ["%3d"%(k) for k in kk_occ] + [" | "] \
                + ["%3d"%(k) for k in kk_unocc]
            if len(Arr)<=16:
                print("%-8s"%(Str) + " ".join(Arr))
            else:
                for k0 in range(0, len(Arr), 16):
                    kf = min(k0+16, len(Arr))
                    if k0==0:
                        print("%-8s"%(Str) + " ".join(Arr[k0:kf]))
                    else:
                        print(" "*8 + " ".join(Arr[k0:kf]))

    # Report all epsilon
    def epsilon(self):
        return self.eps_sorted

    # Report a given orbital, C_k
    def Ck(self, k):
        if self.NSym==1:
            return self.C_so[:,k]
        else:
            s = self.s_sorted[k]
            j = self.k_sorted[k]

            return self.ao_to_so[s].dot(self.C_so[s][:,j])

    # Report all C
    def C(self, CIn=None):
        if CIn is None: CIn = self.C_so

        if self.NSym==1:
            return CIn * 1.
        else:
            C = np.zeros((self.NBasis, self.NBasis))
            k0 = 0
            for k in range(self.NSym):
                C_k = self.ao_to_so[k].dot(CIn[k])
                dk = C_k.shape[1]
                C[:,k0:(k0+dk)] = C_k
                k0 += dk
            return C[:,self.ii_sorted]

    # Convert the so matrix to dense form
    def Dense(self, X):
        if self.NSym==1:
            return X
        else:
            XX = 0.
            for s in range(self.NSym):
                XX += self.ao_to_so[s].dot(X[s]).dot(self.ao_to_so[s].T)
            return XX

    # Solve a Fock-like equation using symmetries
    # if k0>0 use only the subspace spanned by C[:,k0:]
    def SolveFock(self, F, S=None, k0=-1):
        # Note, k0>=0 means to solve only in the basis from C[:,k0:]
        if self.NSym==1:
            if k0<=0:
                return la.eigh(F, b=S)
            else:
                # FV = SVw
                # V=CU
                # FCU = SCUw
                # (C^TFC)U = (C^TSC)Uw
                # XU = Uw

                C = self.C()[:,k0:]
                F_C = (C.T).dot(F).dot(C)
                w, U = la.eigh(F_C)
                return w, C.dot(U)
        else:
            k0s = [0]*self.NSym
            ws = [None]*self.NSym
            Cs = [None]*self.NSym

            if k0>0:
                # Use no terms
                for s in range(self.NSym):
                    k0s[s] = self.NBasis

                # Evaluate the smallest k value for each symmetry
                for i in range(k0,self.NBasis):
                    s = self.s_sorted[i]
                    k0s[s] = min(k0s[s],self.k_sorted[i])

            for s in range(self.NSym):
                # Project onto the subset starting at k0s
                C_ao = self.ao_to_so[s].dot(self.C_so[s][:,k0s[s]:])
                F_C = (C_ao.T).dot(F).dot(C_ao)
                if not(S is None):
                    S_C = (C_ao.T).dot(S).dot(C_ao)
                else: S_C = None

                if F_C.shape[0]>0:
                    ws[s], Us = la.eigh(F_C, b=S_C)
                    Cs[s] = C_ao.dot(Us)
                else:
                    ws[s] = []
                    Cs[s] = [[]]

            # Project back onto the main set
            k0 = max(k0,0)
            w = np.zeros((self.NBasis - k0,)) + 200. # If errors, make sure they're high energy
            C = np.zeros((self.NBasis,self.NBasis - k0))

            #for i in self.ii_sorted[k0:]: # Old and I am sure not correct
            for i in range(k0, self.NBasis): # Index of orbital
                s = self.s_sorted[i] # Its symmetry
                k = self.k_sorted[i] # Its k value in the symmetry
                w[i-k0] = ws[s][k-k0s[s]] # Copy in - k0s is the smallest value in the subset
                C[:,i-k0] = Cs[s][:,k-k0s[s]] # Like above

            return w, C

#################################################################################################
# This code handles degeneracies using a custom-brew model
#################################################################################################

class SymHelperInternal:
    def __init__(self, wfn):
        self.NSym = wfn.nirrep()
        self.NBasis = wfn.nmo()

        self.S_so = wfn.S().to_array()
        self.H_so = wfn.H().to_array()
        self.F_so = wfn.Fa().to_array()
        self.ao_to_so = wfn.aotoso().to_array()

        self.S_ao = self.Dense(self.S_so)
        self.H_ao = self.Dense(self.H_so)
        self.F_ao = self.Dense(self.F_so)

        self.PrepareSyms()

    def PrepareSyms(self, eps_degen=1e-6):
        # Get the eigenvalues of the core Hamiltonian
        w, CR = la.eigh(self.H_ao, b=self.S_ao)
        # Round them
        w = np.round(w/eps_degen)*eps_degen
        # Get unique
        wu = set(w)
        kk = np.arange(len(w))
        Layers = {}
        for w_ in sorted(wu):
            ii = w==w_
            k_ = kk[ii]
            if len(k_) not in Layers:
                Layers[len(k_)] = []

            Layers[len(k_)] += list(k_)

        print(Layers)

        quit()



    # Do a symmetry report to help identifying orbitals
    def SymReport(self, kh, eta=zero_round):
        epsh = self.eps_sorted[kh] + eta
        print("Orbital indices by symmetry - | indicates virtual:")
        for s in range(self.NSym):
            Str = "Sym%02d : "%(s)
            eps = self.eps_so[s]
            if not(hasattr(eps, '__len__')) or len(eps)==0: continue

            kk_occ = []
            kk_unocc = []
            for k, e in enumerate(eps):
                if e<epsh: kk_occ += [ self.ks_map[(s,k)] ]
                else: kk_unocc += [ self.ks_map[(s,k)] ]

            Arr = ["%3d"%(k) for k in kk_occ] + [" | "] \
                + ["%3d"%(k) for k in kk_unocc]
            if len(Arr)<=16:
                print("%-8s"%(Str) + " ".join(Arr))
            else:
                for k0 in range(0, len(Arr), 16):
                    kf = min(k0+16, len(Arr))
                    if k0==0:
                        print("%-8s"%(Str) + " ".join(Arr[k0:kf]))
                    else:
                        print(" "*8 + " ".join(Arr[k0:kf]))

    # Report all epsilon
    def epsilon(self):
        return self.eps_sorted

    # Report a given orbital, C_k
    def Ck(self, k):
        if self.NSym==1:
            return self.C_so[:,k]
        else:
            s = self.s_sorted[k]
            j = self.k_sorted[k]

            return self.ao_to_so[s].dot(self.C_so[s][:,j])

    # Report all C
    def C(self, CIn=None):
        if CIn is None: CIn = self.C_so

        if self.NSym==1:
            return CIn * 1.
        else:
            C = np.zeros((self.NBasis, self.NBasis))
            k0 = 0
            for k in range(self.NSym):
                C_k = self.ao_to_so[k].dot(CIn[k])
                dk = C_k.shape[1]
                C[:,k0:(k0+dk)] = C_k
                k0 += dk
            return C[:,self.ii_sorted]

    # Convert the so matrix to dense form
    def Dense(self, X):
        if self.NSym==1:
            return X
        else:
            XX = 0.
            for s in range(self.NSym):
                XX += self.ao_to_so[s].dot(X[s]).dot(self.ao_to_so[s].T)
            return XX

    # Solve a Fock-like equation using symmetries
    # if k0>0 use only the subspace spanned by C[:,k0:]
    def SolveFock(self, F, S=None, k0=-1):
        # Note, k0>=0 means to solve only in the basis from C[:,k0:]
        if self.NSym==1:
            if k0<=0:
                return la.eigh(F, b=S)
            else:
                # FV = SVw
                # V=CU
                # FCU = SCUw
                # (C^TFC)U = (C^TSC)Uw
                # XU = Uw

                C = self.C()[:,k0:]
                F_C = (C.T).dot(F).dot(C)
                w, U = la.eigh(F_C)
                return w, C.dot(U)
        else:
            k0s = [0]*self.NSym
            ws = [None]*self.NSym
            Cs = [None]*self.NSym

            if k0>0:
                # Use no terms
                for s in range(self.NSym):
                    k0s[s] = self.NBasis

                # Evaluate the smallest k value for each symmetry
                for i in range(k0,self.NBasis):
                    s = self.s_sorted[i]
                    k0s[s] = min(k0s[s],self.k_sorted[i])

            for s in range(self.NSym):
                # Project onto the subset starting at k0s
                C_ao = self.ao_to_so[s].dot(self.C_so[s][:,k0s[s]:])
                F_C = (C_ao.T).dot(F).dot(C_ao)
                if not(S is None):
                    S_C = (C_ao.T).dot(S).dot(C_ao)
                else: S_C = None

                if F_C.shape[0]>0:
                    ws[s], Us = la.eigh(F_C, b=S_C)
                    Cs[s] = C_ao.dot(Us)
                else:
                    ws[s] = []
                    Cs[s] = [[]]

            # Project back onto the main set
            k0 = max(k0,0)
            w = np.zeros((self.NBasis - k0,)) + 200. # If errors, make sure they're high energy
            C = np.zeros((self.NBasis,self.NBasis - k0))

            #for i in self.ii_sorted[k0:]: # Old and I am sure not correct
            for i in range(k0, self.NBasis): # Index of orbital
                s = self.s_sorted[i] # Its symmetry
                k = self.k_sorted[i] # Its k value in the symmetry
                w[i-k0] = ws[s][k-k0s[s]] # Copy in - k0s is the smallest value in the subset
                C[:,i-k0] = Cs[s][:,k-k0s[s]] # Like above

            return w, C

