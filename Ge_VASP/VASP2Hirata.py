import numpy as np

from LibEIGENVAL import *

Help = """
Usage:
   VASP2Hirata.py Temperature[units] EIGENVAL_File Nk

Temperature must be specified and is assumed to be eV unlesss units
is specified as Ha, K or kK (e.g. 0.2 or 0.2eV or 0.1K or 0.2Ha)

EIGENVAL_File is a VASP EIGENVAL file and defaults to EIGENVAL if not specified

Nk (e.g. 3x3x3) is the k-grid and the code attempts to guess it if not specified
"""

import sys
if len(sys.argv)<2:
    print(Help)
    quit()


RawTemperature = sys.argv[1].lower()

if len(sys.argv)>=3:
    FName = sys.argv[2]
else:
    FName = 'EIGENVAL'

if len(sys.argv)>=4:
    Nk = tuple([int(x) for x in sys.argv[3].upper().split('X')])
else:
    Nk = None


eV_to_K = 11604.59
eV_to_Ha = 1/27.211

if 'kk' in RawTemperature:
    T = float(RawTemperature[:-2])*1000/eV_to_K
elif 'k' in RawTemperature:
    T = float(RawTemperature[:-1])/eV_to_K
elif 'ha' in RawTemperature:
    T = float(RawTemperature[:-2])/eV_to_Ha
elif 'ev' in RawTemperature:
    T = float(RawTemperature[:-2])
else:
    try:
        T = float(RawTemperature)
    except:
        print("Did not specify a valid temperature")
        print(Help)

EH = EIGENVALHelper(FName, Nk=Nk)
if Nk is None: Nk = EH.Nk

print("Evaluating properties from %s"%(FName))
print("Evaluating at T = %8.4f eV and Nk = %2d x %2d x %2d"\
      %(T, Nk[0], Nk[1], Nk[2]))


print("Hirata gap = %8.4f eV"%(EH.Hirata(T)))
