import numpy as np

def safeexp(x):
    return np.exp(np.minimum(x, 86.))

def safelog(x):
    return np.log(np.maximum(x, 1e-16))

def f_to_Ss(f):
    fh = f/2.
    return -2.*np.sum(fh*safelog(fh) + (1-fh)*safelog(1-fh))

def eps_to_Occ(eps, tau, N, Cut=86.):
    tau = max(tau, 1e-4)

    eps = eps - eps.min()
    epsl, epsr = eps.min()-8*tau, eps.max()+8*tau

    def dN(mu):
        f = 2./(1. + np.exp(np.minimum((eps-mu)/tau,Cut)))
        return np.sum(f)-N

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
