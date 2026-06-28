import psi4

from psi4Engine.Engine import psi4Engine, GX24
from SafeThermalOcc import ThermalOcc, safelog

import numpy as np
import scipy.linalg as la


eV = 27.211

def NiceOcc(f):
    return " ".join(["%5.3f"%(x) for x in f])

def Niceq(q):
    return " ".join(["%7.4f"%(x) for x in q])

# Old name and convention
def GetOcc(tau, epsilon, N0=6):
    return ThermalOcc(epsilon, tau, N0)

class LiHelper:
    def __init__(self, NLi, a=2.67,
                 DFA='hse06', DFADict = None,
                 Basis='def2-smsvp',
                 NCache = 4,
                 ):
        self.NLi = NLi

        Suff = '_%s_%s'%(DFA.upper(), Basis.lower())
        self.CacheFile = "Data/Li_EGKS_%03d%s.npz"%(NLi, Suff)

        try:
            X = np.load(self.CacheFile, allow_pickle=True)
            self.AllData = X['Data'][()]
        except:
            self.AllData = {}

        if not(DFADict is None): self.AllData = {}


        self.kl = NLi*3//2
        self.kh = self.kl-1

        self.Basis = Basis
        self.DFA = DFA

        psi4.set_output_file('__cluster_egks.out')
        psi4.set_options({
            'basis': Basis,
            'dft_spherical_points': 110,
        })

        if NLi>2:
            F = open('LiClusters/Li_%03d_J.xyz'%(NLi))
            GeomStr = ''.join(list(F)[2:])
            F.close()
        else:
            GeomStr = "Li\nLi 1 %.3f"%(a)

        psi4.geometry(GeomStr)

        if not(DFADict is None):
            E0, wfn = psi4.energy('scf', dft_functional=DFADict,
                                  return_wfn=True)
        else:
            E0, wfn = psi4.energy(DFA, return_wfn=True)

        self.Engine = psi4Engine(wfn)
        self.F = self.Engine.F*1.
        self.FLoc = None

        self.NCache = NCache
        self.NBas = self.F.shape[0]

        self.DCache = np.zeros((self.NCache,self.NBas,self.NBas))
        self.VCache = np.zeros((self.NCache,self.NBas,self.NBas))
        self.FCache = np.zeros((self.NCache,self.NBas,self.NBas))


    def F_to_Stuff(self, kbT, F, HF=False):
        epsilon, C = la.eigh(F, b=self.Engine.S_ao)

        f = GetOcc(kbT, epsilon, N0=3*self.NLi)
        Cf = C * np.sqrt(f)[None,:]
        D = Cf.dot(Cf.T)

        t = f/2.
        tauS = -kbT*2.*np.sum(t*safelog(t)+(1-t)*safelog(1-t))

        ECore = self.Engine.Enn + np.vdot(D, self.Engine.H_ao)


        if HF:
            VH = self.Engine.GetFJ(Cf)
            Vx = -0.5*self.Engine.GetFK(Cf)
            EH = 0.5*np.vdot(D, VH)
            Ex = 0.5*np.vdot(D, Vx)
            EHx = EH + Ex

            FE = ECore + EHx - tauS
            FNew = self.Engine.H_ao + VH + Vx

            Props = {
                'f': f, 'C': C, 'epsilon': epsilon, 'D_T': D,
                'VH': VH, 'Vx': Vx,
                'FE_T': FE, 'E_T': ECore+EHx, 'tauS': tauS,
                'ECore': ECore, 'EH':EH, 'Ex': Ex, 'EHx': EHx,
            }
        else:
            VH = self.Engine.GetFJ(Cf)
            EHx, VHx = self.Engine.GetHx(Ca=Cf/np.sqrt(2.))
            Exc, Vxc = self.Engine.GetDFA(Da=D/2.)

            FE = ECore + EHx + Exc - tauS
            FNew = self.Engine.H_ao + VHx + Vxc

            Props = {
                'f': f, 'C': C, 'epsilon': epsilon, 'D_T': D,
                'VHx': VHx, 'Vxc': Vxc, 'VH': VH,
                'FE_T': FE, 'E_T': ECore+EHx+Exc, 'tauS': tauS,
                'ECore': ECore, 'EHx':EHx, 'Exc': Exc, 'EHxc': EHx+Exc,
            }

        return FNew, FE, Props
    
    def QReport(self, Props):
        Gap = Props['epsilon'][self.kl] - Props['epsilon'][self.kh]
        print("Gap = %8.4f eV"%(Gap*eV))
        print("Core = " + NiceOcc(Props['f'][:self.NLi]))
        print("Fron = " + NiceOcc(Props['f'][self.NLi:(2*self.NLi)]))

    def SolveEGKS(self, kbT,
                  Mix_SCF = 0.1,
                  df_Cut = 1e-5, E_Cut = 1e-6,
                  Report = 0,
                  ):
        if not(kbT in self.AllData):
            self.AllData[kbT] = {}
        elif 'HF' in self.AllData[kbT]:
            print('Cached EGKS problem')
            self.QReport(self.AllData[kbT]['HF'])
            return True


        print('Solving the EGKS problem')
        fOld = None
        FEOld = None
        Pulay_Step = 1
        for step in range(50):
            FNew, FE, Props = self.F_to_Stuff(kbT, self.F)

            if not(fOld is None):
                df = Props['f'] - fOld
                df_Err = np.max(np.abs(df))
                ef_Err = np.sum(Props['f']*Props['epsilon'] - efOld)
                FE_Err = np.abs(FE - FEOld)
            else:
                df_Err = 2.
                ef_Err = 2.
                FE_Err = 2.

            if step>3 and df_Err<=df_Cut*Scale_SCF and FE_Err<=E_Cut*Scale_SCF:
                break

            if (Report>2) or ((Report==0) and (step%5==0)):
                print("%4d %10.5f %10.7f %10.7f %10.7f"\
                      %(step, FE, df_Err, ef_Err, FE_Err))
    
            fOld = Props['f']
            efOld = Props['f']*Props['epsilon']
            FEOld = FE

            if Mix_SCF is None:
                self.VCache[step%self.NCache,:,:] = Props['D_T']*1.
                DOld = self.VCache[(step-1)%(self.NCache),:,:]

                self.FCache[step%self.NCache,:,:] = self.F*1.
                self.DCache[step%self.NCache,:,:] = Props['D_T'] - DOld

                if step<10:
                    Scale_SCF = 0.2
                    self.F = (1.-Scale_SCF)*self.F + Scale_SCF*FNew                    
                elif (step-Pulay_Step)<self.NCache:
                    self.F = FNew
                    Scale_SCF = 1.
                else:
                    Pulay_Step = step

                    X = np.einsum('Apq,Bpq->AB', self.DCache, self.DCache)

                    Y = np.ones((self.NCache+1,self.NCache+1))
                    Y[:self.NCache,:self.NCache] = X
                    Y[-1,-1] = 0.
                    c = np.zeros((self.NCache+1,))
                    c[-1] = 1.

                    q = la.solve(Y + 1e-7*np.eye(self.NCache+1), c)[:self.NCache]
                    q /= np.sum(q)

                    if Report>2:
                        print(X)
                        print("q = " + Niceq(q))

                    
                    self.F = np.einsum('A,Apq->pq', q, self.FCache)
                    Scale_SCF = 1.-np.max(q)
            else:
                self.F = (1-Mix_SCF)*FNew + Mix_SCF*self.F
                Scale_SCF = (1-Mix_SCF)

        if df_Err>df_Cut*Scale_SCF or FE_Err>E_Cut*Scale_SCF:
            return False
        
        print("%4d %10.5f %10.7f %10.7f %10.7f"%(step, FE, df_Err, ef_Err, FE_Err))
        self.AllData[kbT]['HF'] = Props

        self.QReport(Props)
        np.savez(self.CacheFile, Data=self.AllData)
        return True

    def SolveInvert(self, kbT,
                    Mix_Invert = 0.3,
                    ):
        if not(kbT in self.AllData):
            self.AllData[kbT] = {}
        elif 'Inv' in self.AllData[kbT]:
            print('Cached inverted problem')
            self.QReport(self.AllData[kbT]['Inv'])
            return

        print('Inverting the EGKS problem')
        RefProps = self.AllData[kbT]['HF']

        VRef = RefProps['VH']*1.
        DRef = RefProps['D_T']*1.
        Exc, Fxc = self.Engine.GetDFA(Da=DRef/2.)
        FRef = self.Engine.H_ao + VRef + Fxc
        if self.FLoc is None: self.FLoc = 1.*FRef
        
        for step in range(2000):
            _, FE, Props = self.F_to_Stuff(kbT, self.FLoc, HF=True)
            
            self.FCache[step%self.NCache,:,:] = self.FLoc - FRef
            self.DCache[step%self.NCache,:,:] = Props['D_T'] - DRef
            self.VCache[step%self.NCache,:,:] = Props['VH' ] - VRef


            Err = np.vdot(Props['D_T']-DRef, Props['VH']-VRef)
            if (step%50)==0:
                print("%4d %10.5f %10.8f"%(step, FE, Err))

            if (Err/self.NLi)<1e-7:
                break

            if (step>self.NCache) and ((step%self.NCache)==0) and (Err>1e-5):
                X = np.einsum('Apq,Bpq->AB',self.VCache,self.DCache)

                Y = np.ones((self.NCache+1,self.NCache+1))
                Y[:self.NCache,:self.NCache] = X
                Y[-1,-1] = 0.
                c = np.zeros((self.NCache+1,))
                c[-1] = 1.

                q = la.solve(Y + 1e-7*np.eye(self.NCache+1), c)[:self.NCache]
                q /= np.sum(q)

                print("-- pulay mixing %10.6f"%(Err))
                
                self.FLoc = FRef + np.einsum('Apq,A->pq', self.FCache, q)
            else:
                self.FLoc += Mix_Invert*(Props['VH']-VRef)

        _, FE, Props = self.F_to_Stuff(kbT, self.FLoc)
        print("%4d %10.5f %10.8f"%(step, FE, Err))
        self.AllData[kbT]['Inv'] = Props

        self.QReport(Props)
        np.savez(self.CacheFile, Data=self.AllData)
