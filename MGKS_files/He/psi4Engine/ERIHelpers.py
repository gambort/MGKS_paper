__version__ = 1.6
__author__ = "Tim Gould"

import psi4
import numpy as np
import scipy.linalg as la

import numpy.random as np_ra

from psi4Engine.SymHelpers import *

eV = 27.211
zero_round = 1e-6

np.set_printoptions(precision=4, suppress=True, floatmode="fixed")

##### Two-body integrals ####

class JKHelper:
    """
    Simplified python access to the psi4 jk handler
    """
    def __init__(self, wfn, omega=None, mem=None,
                 Debug=False):
        self.NBas = wfn.nmo()
        self.Has_RS = not(omega is None)

        self.JK = None

        # Note, need a beter general solution than just switching off
        NoJK = True
        if NoJK:
            self.NewJK(wfn.basisset(), omega, mem=mem)
        else:
            #self.JK = wfn.jk()
            #self.JK.set_do_wK(True)
            #self.JK.initialize()
            wfn.jk().finalize()
            self.NewJK(wfn.basisset(), omega, mem=mem)

        self.Debug = Debug


    def __del__(self):
        #print("Closing JK helper")
        if not(self.JK is None): self.JK.finalize()

    def NewJK(self, basis, omega, mem=None):
        # Finalize the current if required
        if not(self.JK is None): self.JK.finalize()

        self.JK = psi4.core.JK.build(basis, jk_type="DF",
                                     do_wK=self.Has_RS,
                                     memory=128*1024*1024)
        if mem is None:
            mem = self.JK.memory_estimate()
            MaxMem = int(psi4.get_memory()*0.8)
            if mem>MaxMem:
                print("Need approximately 1024^%4.1f bytes out of 1024^%4.1f"\
                      %(np.log(mem)/np.log(1024), np.log(MaxMem)/np.log(1024) ))
                mem = MaxMem

        self.JK.set_memory( mem )
        self.JK.set_wcombine(False) # Comment out for older psi4
        if self.Has_RS:
            self.JK.set_omega(omega)
            self.JK.set_omega_alpha(0.0) # Comment out for older psi4
            self.JK.set_omega_beta(1.0) # Comment out for older psi4
            self.JK.set_do_wK(True)

        self.JK.initialize()


    def FJ(self, C, CR=None):
        return self.FMaster(C, CR, "J")
    def FK(self, C, CR=None):
        return self.FMaster(C, CR, "K")
    def FK_w(self, C, CR=None):
        return self.FMaster(C, CR, "K_w")

    def FMaster(self, C, CR=None, Mode='J'):
        if self.Debug: print("Getting Fock operator %s"%(Mode))
        if not(CR is None):
            if len(CR.shape)==1:
                CRM = psi4.core.Matrix.from_array(CR.reshape((self.NBas,1)))
            else:
                CRM = psi4.core.Matrix.from_array(CR)

        if len(C.shape)==1:
            CM = psi4.core.Matrix.from_array(C.reshape((self.NBas,1)))
        else:
            CM = psi4.core.Matrix.from_array(C)

        self.JK.C_clear()
        self.JK.C_left_add(CM)
        if CR is None:
            self.JK.C_right_add(CM)
        else:
            self.JK.C_right_add(CRM)

        self.JK.compute()

        if Mode.upper()=='J':
            return self.JK.J()[0].to_array(dense=True)
        elif Mode.upper()=='K':
            return self.JK.K()[0].to_array(dense=True)
        elif Mode.upper() in ('WK', 'KW', 'K_W'):
            if self.Has_RS:
                return self.JK.wK()[0].to_array(dense=True)
            else: return 0.
        else:
            return self.JK.J()[0].to_array(dense=True), \
                self.JK.K()[0].to_array(dense=True)



##### Process density-fitting ####
# THIS ROUTINE IS RETAINED BUT NEVER USED

################################################################################
# Note - all the ERI needs rewriting
# See ~/Molecules/Misc-Code/JK-Tests.py for help
################################################################################

def GetDensityFit(wfn, basis, mints, omega=None,
                  DFName=None, DFMode='RIFIT',
                  return_all = False):
    if DFName is None: DFName = basis.name()
    aux_basis = psi4.core.BasisSet.build\
        (wfn.molecule(), "DF_BASIS_SCF", "",
         DFMode, DFName)
    zero_basis = psi4.core.BasisSet.zero_ao_basis_set()
    SAB = np.squeeze(mints.ao_eri(aux_basis, zero_basis, basis, basis))
    metric = mints.ao_eri(aux_basis, zero_basis, aux_basis, zero_basis)
    metric.power(-0.5, 1e-14)
    metric = np.squeeze(metric)
    ERIA = np.tensordot(metric, SAB, axes=[(1,),(0,)])

    if not(omega is None):
        # Get the density fit business
        # Need to work out how to do density fit on rs part
        IntFac_Apq = psi4.core.IntegralFactory\
            (aux_basis, zero_basis, basis, basis)
        IntFac_AB  = psi4.core.IntegralFactory\
            (aux_basis, zero_basis, aux_basis, zero_basis)
        SAB_w = np.squeeze(
            mints.ao_erf_eri(omega, IntFac_Apq) )
        metric_w = mints.ao_erf_eri(omega, IntFac_AB )
        metric_w.power(-0.5, 1e-14)
        metric_w = np.squeeze(metric_w)

        # ERI in auxilliary - for speed up
        ERIA_w = np.tensordot(metric_w, SAB_w, axes=[(1,),(0,)])
    else:
        ERIA_w, SAB_w, metric_w = None, None, None

    if return_all:
        return ERIA, ERIA_w, SAB, SAB_w, metric, metric_w
    else:
        return ERIA, ERIA_w
