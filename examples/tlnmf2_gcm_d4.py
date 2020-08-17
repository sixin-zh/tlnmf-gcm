import numpy as np
from scipy import fftpack
from tlnmf import tl_nmf_gcm_newton
from tools_draw import draw_atoms, draw_WH
import matplotlib.pyplot as plt

M = 4
N = 3
K = 2
eps_nmf = 1e-8 # taken to be 1e-16 works as well
iter_tl = 500
iter_pertl = 10
tol_tl = 1e-5

barPhit = fftpack.dct(np.eye(M), 2, norm='ortho')
barPhi = barPhit.T

barW = np.zeros((M, K))
barH = np.zeros((K, N)) # eye(K)
barW[0,0] = 1
barW[1,0] = 1
barW[2,0] = 0.1
barW[3,0] = 0.1
barW[0,1] = 0.1
barW[1,1] = 0.1
barW[2,1] = 1
barW[3,1] = 1

barH[0,0] = 1
barH[0,1] = 0
barH[0,2] = 1
barH[1,0] = 0
barH[1,1] = 1
barH[1,2] = 1

print('barW barH', barW@barH)

np.seterr(under='warn')
rng = np.random

Phi, W, H, Phi_init, infos = tl_nmf_gcm_newton(
    barPhi,barW,barH,K,verbose=True,rng=rng,\
    max_iter=iter_tl, n_iter_tl=iter_pertl,\
    tol=tol_tl, eps_nmf=eps_nmf)

A = Phi @ barPhi.T

import array_to_latex as a2l

a2l.to_ltx(A, frmt = '{:6.3f}', arraytype = 'array')

