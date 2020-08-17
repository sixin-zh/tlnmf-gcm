# Authors: Pierre Ablin <pierre.ablin@inria.fr>
#          Dylan Fagot
#
# License: MIT
import numpy as np


eps = 1e-15


def update_nmf_sparse_regh(V, W, H, V_hat, regul):
    ''' One step of NMF with L1 regularization
    The algorithm is detailed in:

        Cedric Fevotte and Jerome Idier
        "Algorithms for non-negative matrix factorization with the
        beta-divergence"
        Neural Computations, 2011

    Parameters
    ----------
    V : array, shape (M, N)
        Target spectrogram

    W : array, shape (M, K)
        Current dictionnary

    H : array, shape (K, N)
        Current activations

    V_hat : array, shape (M, N)
        Current learned spectrogram. Equals dot(W, H)

    regul : float
        Regularization level
    '''
    Ve = V + eps
    V_he = V_hat + eps
    H = H * (np.dot(W.T, Ve * V_he ** -2.) /
             (np.dot(W.T, 1. / V_he) + regul)) ** 0.5
    V_hat = np.dot(W, H)
    V_he = V_hat + eps
    W *= (np.dot(Ve * V_he ** -2., H.T) /
          (np.dot(1. / V_he, H.T) + regul * np.sum(H, axis=1)))**0.5
    # Normalize
    H = H / np.sum(H, axis=1)
    return W, H


def update_nmf_sparse(V, W, H, V_hat, regul, eps=2.2204e-16):
    ''' One step of NMF with L1 regularization
    The algorithm is detailed in:

        Cedric Fevotte and Jerome Idier
        "Algorithms for non-negative matrix factorization with the
        beta-divergence"
        Neural Computations, 2011

    Parameters
    ----------
    V : array, shape (M, N)
        Target spectrogram

    W : array, shape (M, K)
        Current dictionnary

    H : array, shape (K, N)
        Current activations

    V_hat : array, shape (M, N)
        Current learned spectrogram. Equals dot(W, H)

    regul : float
        Regularization level
    '''
    #print('eps',eps)
    Ve = V + eps
    V_he = V_hat + eps
    H = H * (np.dot(W.T, Ve * V_he ** -2.) /
             (np.dot(W.T, 1. / V_he) + regul)) ** 0.5
    V_hat = np.dot(W, H)
    V_he = V_hat + eps
    W *= (np.dot(Ve * V_he ** -2., H.T) /
          (np.dot(1. / V_he, H.T) + regul * np.sum(H, axis=1)))**0.5
    # Normalize
    C = np.sum(W, axis=0) +eps
    W = W / C
    H = H * np.expand_dims(C.transpose(),axis=1) # missing!
    return W, H

def update_nmf_smooth(V, W, H, V_hat, regul):
    ''' One step of NMF with smooth regularization
    The algorithm is detailed in:

        Cedric Fevotte
        "Majorization-minimization algorithm for smooth itakura-saito
        nonnegative matrix factorization"
        ICASSP, 2011

    Parameters
    ----------
    V : array, shape (M, N)
        Target spectrogram

    W : array, shape (M, K)
        Current dictionnary

    H : array, shape (K, N)
        Current activations

    V_hat : array, shape (M, N)
        Current learned spectrogram. Equals dot(W, H)

    regul : float
        Regularization level
    '''
    K, N = H.shape
    Ve = V + eps
    V_he = V_hat + eps
    # Minimization in H
    Gn = np.dot(W.T, Ve * V_he ** -2.)
    Gp = np.dot(W.T, 1. / V_he)
    # H1
    Ht = H.copy()
    p2 = Gp[:, 0] + regul / H[:, 1]
    p1 = - regul
    p0 = -Gn[:, 0] * H[:, 0] ** 2
    H[:, 0] = (np.sqrt(p1 ** 2 - 4. * p2 * p0) - p1) / (2 * p2)
    # Middle
    for n in range(1, N - 1):
        H[:, n] =\
            np.sqrt((Gn[:, n] * Ht[:, n] ** 2 + regul * H[:, n-1]) /
                    (Gp[:, n] + regul / H[:, n+1]))
    # HN
    p2 = Gp[:, N-1]
    p1 = regul
    p0 = - (Gn[:, N-1] * Ht[:, N-1] ** 2 + regul * H[:, N-2])
    H[:, N-1] = (np.sqrt(p1 ** 2 - 4. * p2 * p0) - p1) / (2 * p2)

    # Minimization in W
    V_hat = np.dot(W, H)
    V_he = V_hat + eps
    W = W * np.dot(Ve * V_he ** -2., H.T) / np.dot(1. / V_he, H.T)
    # Normalization
    scale = np.sum(W, axis=0)
    W = W / scale
    H = H / scale[None, :]
    return W, H

def update_NMF_onlyH_Lambda(H,V,W,K,F,N,Lambda,eps_nmf):
    V_hat = np.dot(W, H)  # Initial factorization
    Ve = V + eps_nmf
    V_he = V_hat + eps_nmf
    H = H * (np.dot(W.T, Ve * V_he ** -2.) /
             (np.dot(W.T, 1. / V_he) + Lambda)) ** 0.5
    V_hat = np.dot(W, H)  # Initial factorization
    return H
