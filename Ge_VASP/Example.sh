for tau in 0.01 0.02 0.03 0.04 0.05l; do
    python3 VASP2Hirata.py ${tau} Outputs\EIGENVAL_HSE03_${tau} 6x6x6
done
