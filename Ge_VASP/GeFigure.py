import numpy as np

from LibEIGENVAL import *

import matplotlib.pyplot as plt
from NiceColours import *

np.set_printoptions(precision=4, suppress=True)

Nk = 216
dN = 1/Nk

def NiceDFA(ID):
    SwapList = {'OT': 'OT-RSH',}
    if ID in SwapList: return SwapList[ID]
    else: return ID

eV_to_K = 11604.59
tau_Melt = 940/eV_to_K # 1000K

tau_all = np.linspace(0, 0.2, 41)
ktau_Plot = [0,8,16,24]

DFAList = ('PBE', 'PBE0', 'HSE06', 'OT')
Cols = {'PBE': 'Purple', 'PBE0': 'Green',
        'HSE06': 'Orange', 'OT': 'Black'}
DFA_Skip = ['PBE']

fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(6,2))

fig_eps, axs_eps = plt.subplots(1, 4, figsize=(12,2), sharey=True)

def QDeriv(x, y):
    if len(x)==2:
        return (y[1]-y[0])/(x[1]-x[0])
    if len(x)==3: d=1
    else: d=2
    
    p = np.polyfit(x, y, d)
    dp = np.polyder(p)
    return np.polyval(dp, 0.)

XTxt, DXTxt = 0.04, 0.025
for KDFA, DFA in enumerate(DFAList):
    print("="*72)
    print(DFA)
    EH = MultiEIGENVALHelper(DFA, tau=0., Nk=(6,6,6))
    eps_nk_0 = 1.*EH.eps_nk

    FG_all = 0.*tau_all


    for k_tau, tau in enumerate(tau_all):
        tau = max(tau, 1e-5)

        EH.InterpFiles(DFA=DFA, tau=tau)

        f_nk, mu = EH.Occ_N(1e-5, EH.N)
        kk_h = f_nk>1.9
        kk_l = f_nk<=1.9
        HLGap = np.min(EH.eps_nk[kk_l]) - np.max(EH.eps_nk[kk_h])

        FG_all[k_tau] = EH.Hirata(tau, s=1)
        
        for s in (3,):
            NN_L = EH.N - dN*np.arange(0,s)
            epsN_L = 0.*NN_L
            for k, N in enumerate(NN_L):
                f_nk, mu = EH.Occ_N(tau, N)
                epsN_L[k] = EH.Avg(f_nk * EH.eps_nk)

            NN_R = EH.N + dN*np.arange(0,s)
            epsN_R = 0.*NN_L
            for k, N in enumerate(NN_R):
                f_nk, mu = EH.Occ_N(tau, N)
                epsN_R[k] = EH.Avg(f_nk * EH.eps_nk)

            deps_m = QDeriv(NN_L-EH.N, epsN_L)
            deps_p = QDeriv(NN_R-EH.N, epsN_R)

        if k_tau in ktau_Plot:

            r = 8
            NN = EH.N + dN*np.arange(-r,r+1)
            epsN = NN*0.
            for k, N in enumerate(NN):
                f_nk, mu = EH.Occ_N(tau, N)
                epsN[k] = EH.Avg(f_nk * EH.eps_nk)

            deps_a = (deps_m + deps_p)/2.
            epsN_T = (NN-EH.N)*deps_a + epsN[r]
                
            ax_eps = axs_eps[KDFA]
            
            cc = NiceColour(ktau_Plot.index(k_tau))
            zo = 1000-k_tau*10
            ax_eps.scatter(NN-EH.N, 1000.*(epsN-epsN_T),
                           color=cc, zorder=zo,)
            ax_eps.plot(NN-EH.N, 1000.*(epsN-epsN_T),
                        color=cc, zorder=zo,
                        label = "%.0f K"%(tau*eV_to_K),
                        )


        if tau<0.09:
            deps = CompareEigs(EH.eps_nk, eps_nk_0, avg=EH.Avg)
            print("tau = %.5f <Delta eps> = %8.5f Gap = %6.2f HL = %6.2f"\
                  %(tau, EH.Avg(np.abs(deps)), deps_p-deps_m, HLGap))



    cc = NiceColour(Cols[DFA])
    ax1.plot(tau_all, FG_all, color=cc)
    
    YTxt = np.interp(XTxt, tau_all, FG_all)
    print(XTxt, YTxt)
    AddBorder(
        ax1.text(XTxt, YTxt, NiceDFA(DFA), color=cc,
                 fontsize=10,
                 ha='center', va='center')
    )

    if not(DFA in DFA_Skip):
        ax2.plot(tau_all, FG_all-FG_all[0], color=cc)

        YTxt = np.interp(XTxt/2.5, tau_all, FG_all-FG_all[0])
        AddBorder(
            ax2.text(XTxt/2.5, YTxt+0.001, NiceDFA(DFA), color=cc,
                     fontsize=10, rotation=60,
                     ha='center', va='center')
        )
        
    XTxt += DXTxt

# Handle the x axis
XTLbl = [0,300,600,900,1200,1500]
XT = np.array(XTLbl)/eV_to_K
XTLbl[1] = 'Room'
XTLbl[3] = 'Melt'
XT[3] = tau_Melt
ax1.set_xticks(XT,XTLbl)
ax1.set_xlim([0,1600/eV_to_K])

XTLbl = [0,100,200,300,400,500,600]
XT = np.array(XTLbl)/eV_to_K
XTLbl[3] = 'Room'
ax2.set_xticks(XT,XTLbl)
ax2.set_xlim([0,650/eV_to_K])

for ax in (ax1, ax2):
    ax.plot([0,2000],[0,0],":k", lw=1)
    AddBorder(
        ax.text(0.02, 0.01, "Ge",
                ha='left', va='bottom', fontsize=14,
                transform = ax.transAxes,)
    )
    ax.set_xlabel("Temperature [K]", fontsize=14)

# Handle the y axis
ax1.set_yticks([0,0.3,0.6,0.9,1.2,1.5])
ax1.set_ylim([-0.05, 1.6])
ax1.set_ylabel("$\\Delta_s^{\\tau}$ [eV]", fontsize=14)

ax2.set_yticks([0,0.003,0.006],[0,3,6])
ax2.set_ylim([-0.002,0.007])
ax2.set_ylabel("$\\Delta_s^{\\tau}-\\Delta_s^0$ [meV]", fontsize=14)

fig.tight_layout(pad=0.2)

# Now do the eps plot
axs_eps[0].legend(ncol=2, loc="center")
axs_eps[0].set_ylabel("$\\Delta\\langle\\epsilon\\rangle^{\\tau}$ [meV]", fontsize=14)
for K_, ax_eps in enumerate(axs_eps):
    ax_eps.axis([-0.039,0.039, -3,30])
    ax_eps.set_xlabel("$\\Delta N$ [unitless]", fontsize=14)
    AddBorder(
        ax_eps.text(0.01, 0.98, NiceDFA(DFAList[K_]),
                    ha='left', va='top', fontsize=14,
                    transform = ax_eps.transAxes,
                    zorder=2000,)
    )
fig_eps.tight_layout(pad=0.2)

fig.savefig("../Fig_Ge.pdf")
fig_eps.savefig("../FigSupp_Ge.pdf")


plt.show()
