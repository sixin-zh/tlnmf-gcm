# Identifiability of TL-NMF
Sixin Zhang, Emmanuel Soubies, and C\'edric F\'evotte. On the Identifiability of Transform Learning for Non-negative Matrix Factorization. IEEE Signal Processing Letters. 2020.

# Install package TLNMF using Python (3.6)

* python setup.py install

# Results for the equation (9) in the paper

* python examples/tlnmf2_gcm_d4.py

The output should be a (random) matrix, e.g.
\begin{array}
   0.000 & -0.000 &   0.981 &   0.193\\
   0.345 &   0.939 & -0.000 & -0.000\\
 -0.939 &   0.345 &   0.000 &   0.000\\
 -0.000 &   0.000 &   0.193 & -0.981
\end{array}

# Results for DCT-NMF (Table 1, Figure 2, Table 2)

* sh run_isnmf.sh

This will run 5 times the DCT-NMF on independent data samples. The data are stored in ./datasets/. The results are stored in the folder ./results_dct2/

Table 1: The regression of the top 16 atoms of DCT-NMF are performed at the end of each run. 
You may check the quality of the regression by looking at, e.g. *./results_dct2/nonstationary440_sim2_win4_ws40ms_run1/isnmf_dctsci_batch_nonstationary440_sim2_K2_eps5e-07_atoms.png*.
The orange curve is a fit for the original cosine atom (from the DCT). 

Figure 2: The W and H are shown in for each run, e.g. *./results_dct2/nonstationary440_sim2_win4_ws40ms_run1/isnmf_dctsci_batch_nonstationary440_sim2_K2_eps5e-07_WH.png*.

Table 2: run the matlab code: eval_bss_mat.m

*p.s. To test other types of DCT, e.g. DCT I, change the -dct 2 to -dct 1 in the run_isnmf.sh*

# Results for TL-NMF (Table 1, Figure 2, Table 2)

* sh run_tlnmf.sh

This will run 5 times the TL-NMF on independent data samples. The results are stored in the folder ./results_pertl20/

Table 1: The regression of the top 8 atoms of TL-NMF are performed at the end of each run. 
You may check the quality of the regression by looking at, e.g.  *./results_pertl20/nonstationary440_sim1_win4_ws40ms_run1/tlnmf2_sci_batch_nonstationary440_sim1_K2_eps5e-07_atoms.png*.
The orange curve is a fit for the learnt atoms of TL-NMF. 

Figure 2: The W and H are shown in for each run, e.g. *./results_pertl20/nonstationary440_sim2_win4_ws40ms_run1/tlnmf2_sci_batch_nonstationary440_sim1_K2_eps5e-07_WH.png*.

Table 2: run the matlab code: eval_bss_mat.m

# Acknowledgement
Part of the code is adapted from: https://github.com/pierreablin/tlnmf

