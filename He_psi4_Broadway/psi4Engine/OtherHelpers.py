__version__ = 1.6
__author__ = "Tim Gould"

import psi4
import numpy as np

eV = 27.211
zero_round = 1e-6

np.set_printoptions(precision=4, suppress=True, floatmode="fixed")


def GX24():
    """Custom DFA for GX24"""
    return {'name': 'GX24',
            'x_hf': {'alpha': 0.375, 'beta': 0.625, 'omega': 0.2},
            'x_functionals': {'GGA_X_HJS_PBE': {'alpha': 0.625, 'omega': 0.2}},
            'c_functionals': {'GGA_C_PBE': {'alpha': 1.0}}
            }

def RSHybridParams(alpha, beta, omega):
    """
    One-spot rs-hybrid handler to map from:

    alpha E_x + beta E_x^{lr} + beta E_x^{sr-DFA} + (1-alpha-beta) E_x^{DFA}

    to different variables
    """
    WDFA = 1. - alpha - beta
    WDFA_SR = beta
    WHF = alpha
    WHF_LR = beta

    # See C:\Users\tgoul\Dropbox\Collabs\Ensemble\EGKS\Implementation\psi4-Notes.pdf

    return WDFA, WDFA_SR, WHF, WHF_LR

# Handle PBE0_XX calculations
def TextDFA(DFA):
    """
    Convert short-hand (w)pbe_[alpha](_[omega])(_[beta])
    to a psi4 DFA dictionary.
    """
    if DFA[:5].lower()=="pbe0_":
        X = DFA.split('_')
        alpha = float(X[1])/100.
        if len(X)>2: f_c = max(float(X[2])/100.,zero_round)
        else: f_c = 1.
        return {
            'name':DFA,
            'x_functionals': {"GGA_X_PBE": {"alpha": 1.-alpha, }},
            'c_functionals': {"GGA_C_PBE": {"alpha": f_c, }},
            'x_hf': {"alpha": alpha, },
        }
    elif DFA[:5].lower()=="pbe_h":
        return {
            'name':DFA,
            'x_functionals': {"GGA_X_HJS_PBE": {"alpha": 1., "omega": 10., }},
            'c_functionals': {"GGA_C_PBE": {"alpha": 1., }},
            'x_hf': {"alpha": 0., },
        }
    elif DFA[:5].lower()=="wpbe_":
        # Format
        # wpbe_[alpha%]_[omega]_[beta%]_[corr%]_[lda%]
        #
        # Only alpha needs to be specificiec - others default to:
        #    omega=0.3, beta=1-alpha, corr=100%, lda=0%
        #
        # E.g. wpbe_25_0.5 gives alpha=0.25, omega=0.5, beta=0.75, corr=1.0, lda=0.0


        X = DFA.split('_')
        alpha = float(X[1])/100.
        if len(X)>2: omega = float(X[2])
        else: omega = 0.3
        if len(X)>3: beta = float(X[3])/100
        else: beta = 1.-alpha
        if len(X)>4: WC = float(X[4])/100
        else: WC = 1.
        if len(X)>5: WLDA_SR = float(X[4])/100
        else: WLDA_SR = 0.

        WDFA, WDFA_SR, WHF, WHF_LR = RSHybridParams(alpha, beta, omega)

        DFADef =  {
            'name':DFA,
            'x_hf': {"alpha":WHF, "beta":WHF_LR, "omega":omega, },
            'x_functionals': {"GGA_X_HJS_PBE": {"alpha":WDFA_SR - WLDA_SR, "omega":omega, }, },
            'c_functionals': {"GGA_C_PBE": {"alpha":WC, } },
        }
        if np.abs(WDFA)>zero_round:
            DFADef["x_functionals"]["GGA_X_PBE"] = {"alpha":WDFA, }
        if np.abs(WLDA_SR)>zero_round:
            DFADef["x_functionals"]["LDA_X_ERF"] = {"alpha":WLDA_SR, "omega":omega, }
            print("Short-range LDA does not appear to be implemented in psi4")
            quit()
        return DFADef
    else:
        return DFA



sf_from_dict =  psi4.driver.dft.build_superfunctional_from_dictionary
# My very hacky mask
def sf_RKS_to_UKS(DFA, ScaleDFA_x=1., ScaleDFA_c=1.):
    """
    ##### This is a hack to convert a RKS superfunctional to its UKS equivalent
    # Internal routine
    # https://github.com/psi4/psi4/blob/master/psi4/driver/procrouting/dft/dft_builder.py#L251
    """
    DFA_Dict = { 'name':DFA.name()+'_u'}
    DFA_Dict['x_functionals']={}
    DFA_Dict['c_functionals']={}
    for x in DFA.x_functionals():
        Name = x.name()[3:]
        alpha = x.alpha()
        omega = x.omega()

        if np.abs(alpha)>zero_round:
            if omega==0.:
                DFA_Dict['x_functionals'][Name] = {"alpha": alpha*ScaleDFA_x, }
            else:
                DFA_Dict['x_functionals'][Name] = {"alpha": alpha*ScaleDFA_x, "omega": omega, }
    for c in DFA.c_functionals():
        Name = c.name()[3:]
        alpha = c.alpha()
        omega = c.omega()

        if np.abs(alpha)>zero_round:
            if omega==0.:
                DFA_Dict['c_functionals'][Name] = {"alpha": alpha*ScaleDFA_c, }
            else:
                DFA_Dict['c_functionals'][Name] = {"alpha": alpha*ScaleDFA_c, "omega": omega, }

    if DFA.needs_vv10():
        #DFAU.set_do_vv10(True)
        #DFAU.set_vv10_b(DFA.vv10_b())
        #DFAU.set_vv10_c(DFA.vv10_c())
        DFA_Dict['dispersion'] = { 'type': 'nl', 'params':{'b': DFA.vv10_b(), 'c': DFA.vv10_c() }, }


    npoints = psi4.core.get_option("SCF", "DFT_BLOCK_MAX_POINTS")
    DFAU, _ = sf_from_dict(DFA_Dict,npoints,1,False)


    return DFAU
##### End hack

# Get the degeneracy of each orbital
def GetDegen(epsilon, eta=zero_round):
    Degen = np.zeros((len(epsilon),),dtype=int)
    for k in range(len(epsilon)):
        ii =  np.argwhere(np.abs(epsilon-epsilon[k])<eta).reshape((-1,))
        Degen[k] = len(ii)
    return Degen


# For nice debug printing
def NiceArr(X):
    return "[ %s ]"%(",".join(["%8.3f"%(x) for x in X]))
def NiceArrInt(X):
    return "[ %s ]"%(",".join(["%5d"%(x) for x in X]))
def NiceMat(X):
    N = X.shape[0]
    if N==0:
        return "[]"
    elif N==1:
        return "["+NiceArr(X[0,:])+"]"
    elif N==2:
        return "["+NiceArr(X[0,:])+",\n "+NiceArr(X[1,:])+"]"
    else:
        R = "["
        for K in range(N-1):
            R+=NiceArr(X[K,:])+",\n "
        R+=NiceArr(X[N-1,:])+"]"
        return R


