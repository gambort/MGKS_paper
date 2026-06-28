import numpy as np

np.set_printoptions(precision=5, suppress=True)

def safeexp(x):
    return np.exp(np.minimum(86,x))

def safelog(x):
    return np.log(np.maximum(x, 1e-36))

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
        return f
    else:
        print("Left with %.8f electrons not %.8f as desired"\
              %(np.sum(w_f*f), N))
        return None


def Entropy(f):
    fh = f/2.
    return -2.*(fh.dot(safelog(fh)) + (1-fh).dot(safelog(1-fh)))
    
if __name__ == "__main__":
    import numpy.random as ra
    
    for step in range(200):
        L = 5 + ra.randint(80)
        epsilon = np.sort(ra.rand(L)*L)

        # Make degeneraxies more likely
        epsilon = np.round(epsilon/0.01)*0.01
        
        N = ra.randint(2*L+1)

        for tau in (1e-6, 1e-3, 1e-2, 1e-1, 1):
            f = ThermalOcc(epsilon, tau, N)
            if f is None:
                print(epsilon, tau, N)
        

    
