import numpy as np
from scipy.optimize import line_search
from scipy.linalg import expm


def compute_gram(Phi,barPhi):
    # return alpha_f,f' = < phi_f(t), bar phi_f'(t) >
    return Phi @ barPhi.T

def compute_V(Phi,barPhi,barW,barH):
    Alpha = compute_gram(Phi,barPhi)
    Alpha2 = Alpha * Alpha # elementwise product
    V = Alpha2 @ (barW @ barH)
    return V

def compute_loss(Phi,V_hat,barPhi,barW,barH,eps):
    V = compute_V(Phi,barPhi,barW,barH)
    V_he = V_hat + eps
    loss = (V+eps) / (V_he) + np.log(V_he)
    loss = np.sum(loss)/2
    return loss

def fast_transform_gcm_newton(Phi,V_hat,barPhi,barW,barH,
                              n_iter_optim,n_ls_tries=30,eps=2.2204e-16):
    '''Optimize w.r.t Phi
    V_hat = WH
    '''
    M, _ = Phi.shape
    obj = None
    V_he = V_hat + eps
    B = (1/V_he) @ ((barW@barH).transpose())
    for n in range(n_iter_optim):
        # Gradient
        A = compute_gram(Phi,barPhi)
        G = (A*B)@(A.transpose())
        G = 0.5 * (G - G.T)  # Project
        #print('G',G.shape)
        # Hessian
        Hs = B@((A*A).transpose())
        #print('Hs',Hs.shape)
        # Project
        H = 0.5 * (Hs + Hs.T)
        # Search direction:
        E = - G / H
        # Line-search
        transform, converged, obj = line_search_scipy_gcm_newton(
            Phi, E, G, V_hat, B,
            barPhi, barW, barH, obj, n_ls_tries, eps)
        if not converged:
            print('ls break')
            break
        Phi = np.dot(transform, Phi)
    return Phi

def line_search_scipy_gcm_newton(Phi, E, G, V_hat, B,
                                 barPhi,  barW, barH, current_loss, n_ls_tries, eps):
    M, _ = E.shape

    class function_caller(object):
        def __init__(self):
            pass

        def myf(self, E):
            self.transform = expm(E.reshape(M, M))
            new_Phi = np.matmul(self.transform, Phi)
            obj = compute_loss(new_Phi,V_hat,barPhi,barW,barH,eps)
            return obj

        def myfprime(self, E):
            self.transform = expm(E.reshape(M, M))
            new_Phi = np.matmul(self.transform, Phi)
            A = compute_gram(new_Phi,barPhi)
            G_ = (A*B)@A.transpose()
            G_ = 0.5 * (G_ - G_.T)
            return G_.ravel()

    fc = function_caller()
    xk = np.zeros(M ** 2)
    gfk = G.ravel()
    pk = E.ravel()
    old_fval = current_loss
    alpha, _, _, new_fval, _, _ = line_search(fc.myf, fc.myfprime, xk, pk, gfk,
                                              old_fval, maxiter=n_ls_tries)
    if alpha is not None:
        # alpha is the new E, return its expm
        transform = fc.transform
        obj = new_fval
        return transform, True, obj
    else:
        return 0, False, 0
