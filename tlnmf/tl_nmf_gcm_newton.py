# mle loss in expectation, newton solver with diag approx.

import numpy as np
from scipy import fftpack

from .utils import check_random_state, unitary_projection
from .functions import new_is_div, penalty
from .nmf import update_nmf_sparse
from .transform_learning_gcm_newton import fast_transform_gcm_newton, \
    compute_gram, compute_V, compute_loss

def tl_nmf_gcm_newton(barPhi, barW, barH, K, Phi=None, W=None, H=None, regul=None, max_iter=300,
                      n_iter_tl=5, tol=1e-4, verbose=False, rng=None, eps_nmf=1e-15):
    '''Runs Transform learning NMF
    WITH SLIGHT MODIFICATION OF THE CODE TO SOLVE THE NEW OBJECTIVE FROM GCM MODEL.

    Parameters
    ----------
    Groundtruth:
    barPhi: (M,M)
    barW: (M,K)
    barH: (K,N)

    K : int
        Rank of the learned feature matrices.

    Phi : array, shape (M, M) | 'random' | 'dct' | None, optional
        Initial Transform. Should be orthogonal. If 'random', start from a
        random orthogonal matrix. If 'dct', start from the DCT coefficients.
        Random by default

    W : array, shape (M, K) | None, optional
        Initial dictionnary.

    H : array, shape (K, N) | None, optional
        Initial activations.

    regul : float | None, optional
        Level of regularization. By default, a heuristic is used.

    max_iter : int, optional
        Maximal number of iterations for the algorithm

    n_iter_tl : int, optional
        Number of iteration of Transform learning between NMF steps

    tol : float, optional
        tolerance for the stopping criterion. Iterations stop when two
        consecutive iterations of the algorithm have a relative objective
        change lower than tol.

    verbose : boolean, optional
        Wether to print or not informations about the current state

    rng : RandomState, optional
        random seed of the algorithm

    Returns
    -------
    Phi : array, shape (M, M)
        The estimated transform matrix

    W : array, shape (M, K)
        The estimated dictionnary

    H : array, shape (K, N)
        The estimated activations

    Phi_init : array, shape (M, M)
        Initial Phi

    infos_list : dict
        Contains various metrics monitoring convergence. Same as printed by
        Verbose.
    '''
    M = barPhi.shape[0]
    F = barW.shape[0]
    N = barH.shape[1]
    
    regul_type = 'sparse'
    rng = check_random_state(rng)
    # Initialization
    if regul is None:
        regul = 0 # 1e6 * float(K) / M
        assert(regul_type == 'sparse')
    else:
        assert(false) # ,'to implenet sparse regul')
    if type(Phi) is not np.ndarray:
        if Phi is None:
            Phi = 'random'
        if Phi == 'random':
            Phi = unitary_projection(rng.randn(M, M))
        elif Phi == 'dct':
            assert(false) # ,'no init with dct for cheating')
            Phi = fftpack.dct(np.eye(M), 3, norm='ortho')
    if W is None:
        W = np.abs(rng.randn(M, K)) + 1.
        W = W / np.sum(W, axis=0)
    if H is None:
        H = np.abs(rng.randn(K, N)) + 1.

    Phi_init = Phi.copy()

    V_hat = np.dot(W, H)
    obj = compute_loss(Phi,V_hat,barPhi,barW,barH,eps=eps_nmf)
    print('init loss',obj)
    
    # Monitoring
    obj_list = []
    eps_list = []
    tl_obj_list = []
    nmf_obj_list = []
    d_phi_list = []
    d_phi_i_list = []
    # Verbose
    if verbose:
        print('Running TL-NMF with %s regularization on a %d x %d '
              'problem with rank K = %d, regul=%g' % (regul_type, M, N, K, regul))
        print(' | '.join([name.center(8) for name in
                         ["iter", "obj", "eps", "NMF", "TL", "d_phi",
                          "d_phi_i"]]))

    for n in range(max_iter):
        # NMF step to update W and H
        V = compute_V(Phi,barPhi,barW,barH) # V = E(|Phi Y|**2)
        W, H = update_nmf_sparse(V, W, H, V_hat, regul, eps=eps_nmf)
        # Transform Learning       
        V_hat = np.dot(W, H)
        obj1 = compute_loss(Phi,V_hat,barPhi,barW,barH,eps=eps_nmf)
        Phi_old = Phi.copy()
        Phi = fast_transform_gcm_newton(Phi,V_hat,barPhi,barW,barH, n_iter_tl, eps=eps_nmf)
        V = compute_V(Phi,barPhi,barW,barH) # V = E(|Phi Y|**2)
        
        # Monitoring
        old_obj = obj.copy()
        obj = compute_loss(Phi,V_hat,barPhi,barW,barH,eps=eps_nmf)
        eps = (old_obj - obj) / (np.abs(obj) + np.abs(old_obj))
        eps1 = old_obj - obj1
        eps2 = obj1 - obj
        delta_phi = np.mean(np.abs(Phi - Phi_old))
        delta_phi_init = np.mean(np.abs(Phi - Phi_init))

        obj_list.append(obj)
        eps_list.append(eps)
        tl_obj_list.append(eps2)
        nmf_obj_list.append(eps1)
        d_phi_list.append(delta_phi)
        d_phi_i_list.append(delta_phi_init)
        # Terminaison
        if np.abs(eps) < tol:
            break
        if verbose:
            print(' | '.join([("%d" % (n+1)).rjust(8),
                              ("%.8e" % obj).rjust(8),
                              ("%.2e" % eps).rjust(8),
                              ("%.2e" % eps1).rjust(8),
                              ("%.2e" % eps2).rjust(8),
                              ("%.2e" % delta_phi).rjust(8),
                              ("%.2e" % delta_phi_init).rjust(8)]))
    infos = dict(obj_list=obj_list, eps_list=eps_list, tl_obj_list=tl_obj_list,
                 nmf_obj_list=nmf_obj_list, d_phi_list=d_phi_list,
                 d_phi_i_list=d_phi_i_list)
    return Phi, W, H, Phi_init, infos

