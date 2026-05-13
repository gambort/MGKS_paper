import numpy as np
from LibHeHF import *

import matplotlib.pyplot as plt
from NiceColours import *

PlotMode = 1

TUnits = 'kK'

Methods = ['EXX', 'HF', ]
FullMethods = Methods +  []
RefMethods = Methods + ['Ref',]

HHH = HeHFHelper()
HHH.AddExact()

Mask = HHH.FixFEGrowth()

if TUnits=='K':
    x_T = HHH.tau * Kelvin
    XT_, DXT = 30000, 9000
    XNarrow = [0,70000]
    XWide = [0,70000]
elif TUnits=='kK':
    x_T = HHH.tau * Kelvin / 1000
    XT_, DXT = 30, 9.3
    XNarrow = [0,70]
    XWide = [0,70]
else:
    TUnits='eV'
    x_T = HHH.tau * eV
    XT_, DXT = 2.5, 0.9
    XNarrow = [3,6]
    XWide = [0,6]

def GetProps():
    Cols = {'Exact': 'Navy', 'Exact_EGKS': 'Maroon',
            'HF': 'Red', 'EXX': 'Blue',
            'Ref': 'Black',
            'Inv': 'Cyan', 'Mix': 'Purple',
            }
    Dash = {'Exact': (4,2), 'Exact_EGKS': (6,3),
            'EXX': (3,1,1,1), 'HF': (5,1),
            'Ref': (),
            'Inv': (1,1), 'Mix': (3,1,1,2),
            }
    Neat = {'Exact': 'Exact$^{\\alpha=0}$', 'Exact_EGKS': 'Exact$^{\\alpha=1}$',
            'HF': 'HF$^{\\alpha=\\alpha_o=1}$',
            'EXX': 'HF$^{\\alpha=0}$',
            'Ref': 'Reference',
            'Inv': 'Inv. HF', 'Mix': 'Mix',
            }
    return Cols, Dash, Neat

Cols, Dash, Neat = GetProps()

XT = 0.
def LabelPlot(ax, x, y, lbl, color='k', dy=0.0, DXT=0.9,
              mask=[],
              **kwargs):
    global XT

    ax.plot(x, y, color=color, **kwargs)
    if len(mask)>0:
        ax.plot(x[mask], y[mask], 'x', color=color)

    YT = np.interp(XT, x, y)-dy
    AddBorder(
        ax.text(XT, YT, lbl,
                ha='center', va='center',
                color=color,
                fontsize=10,
                ),
    )

    XT += DXT


###############################################
Cols, Dash, Neat = GetProps()
Neat['Exact'] = 'Exact@MKS'


###############################################
if PlotMode==0:
    figEps, (ax_Gap, ax_Hir) \
        = plt.subplots(2,1,figsize=(6,3),sharex=True)

    f_tau, df_tau = {}, {}
    eps_tau, Gap_tau, HGap_tau = {}, {}, {}

    Gap_tau_Ref, Et = HHH.GetGap_Exact()

    HGap_Ref = (Et[3] + Et[1] - 2.*Et[2])
    Gap_Ref = Gap_tau_Ref + 0.*HHH.tau
    Gap_tau_Ref = Et[3] + Et[1] - 2*Et[2]

    print("%-10s %7.1f %7.1f %7.1f %7.1f"\
          %('tau =', x_T[0], x_T[5], x_T[10], x_T[15]))

    print("%-10s %7.3f %7.3f %7.3f %7.3f"\
          %("Inter.", Gap_tau_Ref[0]*eV,
            Gap_tau_Ref[5]*eV, Gap_tau_Ref[10]*eV,
            Gap_tau_Ref[15]*eV))
    
    for Method in FullMethods:
        f_tau[Method] = HHH.Get(Method, 'f')
        df_tau[Method] = f_tau[Method] - f_tau[Method][0,:]
        eps_tau[Method] = HHH.Get(Method, 'epsilon')

        # Test for non-degenerate eigenvalues
        kk = np.argwhere(np.abs(eps_tau[Method][:,1]
                                -eps_tau[Method][:,2])>1e-3)\
                                .reshape((-1,))

        HGap_tau[Method] = 0.*HHH.tau
        for k, tau in enumerate(HHH.tau):
            fp = eps_to_Occ(eps_tau[Method][k,:], tau, 1.)
            fm = eps_to_Occ(eps_tau[Method][k,:], tau, 3.)
            f  = eps_to_Occ(eps_tau[Method][k,:], tau, 2.)
            HGap_tau[Method][k] = np.dot(fp+fm-2*f, eps_tau[Method][k,:])

        Gap_tau[Method] = eps_tau[Method][:,1] - eps_tau[Method][:,0]

        print("%-10s %7.3f %7.3f %7.3f %7.3f"\
              %(Method, HGap_tau[Method][0]*eV,
                HGap_tau[Method][5]*eV, HGap_tau[Method][10]*eV,
                HGap_tau[Method][15]*eV))

    XT = XT_
    for K, Method in enumerate(FullMethods):
        LabelPlot(ax_Hir, x_T, HGap_tau[Method]*eV,
                  Neat[Method], DXT=0.,
                  color=NiceColour(Cols[Method]),
                  dashes = Dash[Method],
                  mask = Mask[Method],
                  )

        LabelPlot(ax_Gap, x_T, Gap_tau[Method]*eV,
                  Neat[Method], DXT=DXT,
                  color=NiceColour(Cols[Method]),
                  dashes = Dash[Method],
                  mask = Mask[Method],
                  )


    XT = XT_ - DXT
    for ax in (ax_Gap, ax_Hir):
        LabelPlot(ax, x_T, HGap_Ref*eV,
                  'Exact', DXT=0.,
                  color=NiceColour('Black'),
                  dashes=(), lw=1)

    ax_Gap.set_ylim([29,44])
    ax_Hir.set_ylim([18,45])
            
        
    ax_Hir.set_xlim(XWide)
    #ax_Hir.set_xticks(range(XWide[0],XWide[1]+1))
    ax_Hir.set_xlabel("Temperature $\\tau$ [%s]"%(TUnits), fontsize=14)


    for ax, Lbl in zip((ax_Gap, ax_Hir),
                       ('$1s\\to 2p$ gap [eV]',
                        'Hirata gap [eV]',
                        )):
        AddBorder(
            ax.text(0.01, 0.11, Lbl,
                    ha='left', va='bottom', fontsize=14,
                    transform = ax.transAxes,
                    )
        )

    figEps.tight_layout(pad=0.2)
    figEps.savefig('Fig_He_Simple_2.pdf')

###############################################
if PlotMode==1:
    figEps, ax_Hir \
        = plt.subplots(1,1,figsize=(6,2),sharex=True)

    f_tau, df_tau = {}, {}
    eps_tau, Gap_tau, HGap_tau = {}, {}, {}

    Gap_tau_Ref, Et = HHH.GetGap_Exact()

    HGap_Ref = (Et[3] + Et[1] - 2.*Et[2])
    Gap_Ref = Gap_tau_Ref + 0.*HHH.tau
    Gap_tau_Ref = Et[3] + Et[1] - 2*Et[2]

    print("%-10s %7.1f %7.1f %7.1f %7.1f"\
          %('tau =', x_T[0], x_T[5], x_T[10], x_T[15]))

    print("%-10s %7.3f %7.3f %7.3f %7.3f"\
          %("Inter.", Gap_tau_Ref[0]*eV,
            Gap_tau_Ref[5]*eV, Gap_tau_Ref[10]*eV,
            Gap_tau_Ref[15]*eV))
    
    for Method in FullMethods:
        f_tau[Method] = HHH.Get(Method, 'f')
        df_tau[Method] = f_tau[Method] - f_tau[Method][0,:]
        eps_tau[Method] = HHH.Get(Method, 'epsilon')

        # Test for non-degenerate eigenvalues
        kk = np.argwhere(np.abs(eps_tau[Method][:,1]
                                -eps_tau[Method][:,2])>1e-3)\
                                .reshape((-1,))

        HGap_tau[Method] = 0.*HHH.tau
        for k, tau in enumerate(HHH.tau):
            fp = eps_to_Occ(eps_tau[Method][k,:], tau, 1.)
            fm = eps_to_Occ(eps_tau[Method][k,:], tau, 3.)
            f  = eps_to_Occ(eps_tau[Method][k,:], tau, 2.)
            HGap_tau[Method][k] = np.dot(fp+fm-2*f, eps_tau[Method][k,:])

        Gap_tau[Method] = eps_tau[Method][:,1] - eps_tau[Method][:,0]

        print("%-10s %7.3f %7.3f %7.3f %7.3f"\
              %(Method, HGap_tau[Method][0]*eV,
                HGap_tau[Method][5]*eV, HGap_tau[Method][10]*eV,
                HGap_tau[Method][15]*eV))

    XT = XT_ - DXT
    #ax_Hir.plot(x_T, HGap_Ref*eV, color='k', dashes=(), lw=1)
    LabelPlot(ax_Hir, x_T, HGap_Ref*eV,
              'Exact', DXT=DXT,
              color=NiceColour("Black"),
              dashes = (),
              )

    for K, Method in enumerate(FullMethods):
        c = NiceColour(Cols[Method])
        c_light = (c[0], c[1], c[2], 0.2)
        LabelPlot(ax_Hir, x_T, Gap_tau[Method]*eV, '',
                  color=c_light,
                  dashes = (1,1),
                  )
        
        LabelPlot(ax_Hir, x_T, HGap_tau[Method]*eV,
                  Neat[Method], DXT=DXT,
                  color=c,
                  dashes = Dash[Method],
                  mask = Mask[Method],
                  )
        

    ax_Hir.set_ylim([22,44])
            
        
    ax_Hir.set_xlim(XWide)
    #ax_Hir.set_xticks(range(XWide[0],XWide[1]+1))
    ax_Hir.set_xlabel("Temperature $\\tau$ [%s]"%(TUnits), fontsize=14)

    ax_Hir.set_ylabel("$\\Delta_{(s)}^{\\tau}$ [eV]", fontsize=14)

    figEps.tight_layout(pad=0.2)
    figEps.savefig('Fig_He_Simple.pdf')

plt.show()
