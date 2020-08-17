import numpy as np
from scipy.optimize import line_search
from scipy.linalg import expm

# new objective
def fast_transform_learning_mleb(Phi, Xs, V_hat, n_iter_optim, n_ls_tries=30, eps=2.2204e-16):
    '''Optimize w.r.t Phi
    '''
    M, _ = Phi.shape
    obj = None
    V_he = np.expand_dims(V_hat + eps, axis=0)
    V_hat_inv = 1. / V_he
    for n in range(n_iter_optim):
        # Gradient
        Xse = Xs
        Xse_sq = Xse ** 2 
        #print('Xse,Vhat,Xs', Xse.shape, V_hat_inv.shape, Xs.shape)
        Gs = np.matmul(Xse * V_hat_inv, Xs.transpose((0,2,1)))
        #print('Gs',Gs.shape)
        G = np.mean(Gs,axis=0)
        G = 0.5 * (G - G.T)  # Project
        #print('G',G.shape)
        # Hessian
        Xe_sq = np.mean(Xse_sq,axis=0)
        Hs = np.dot(V_hat_inv[0], Xe_sq.T)
        #print('Hs',Hs.shape)
        # Project
        H = 0.5 * (Hs + Hs.T)
        # Search direction:
        E = - G / H
        # Line-search
        transform, Xs_new, converged, obj =\
            line_search_scipy_mleb(Xse, E, G, V_he, obj, n_ls_tries, eps)
        if not converged:
            print('ls break')
            break
        Phi = np.dot(transform, Phi)
        Xs[:] = Xs_new
    return Phi, Xs

def line_search_scipy_mleb(Xs, E, G, V_he, current_loss, n_ls_tries, eps):
    M, _ = E.shape

    class function_caller(object):
        def __init__(self):
            pass

        def myf(self, E):
            self.transform = expm(E.reshape(M, M))
            new_Xs = np.matmul(self.transform, Xs)
            self.new_Xs = new_Xs.copy() # use this copy in the fprime to save computation time
            fs = (new_Xs ** 2 + eps) / V_he
            f = np.mean(fs,axis=0)
            return 0.5*np.sum(f + np.log(V_he[0]))

        def myfprime(self, E):
            #self.transform = expm(E.reshape(M, M))
            #new_X = np.dot(self.transform, X)
            new_Xs = self.new_Xs
            Gs = np.matmul(new_Xs / V_he, new_Xs.transpose((0,2,1)))
            G = np.mean(Gs,axis=0)
            return 0.5 * (G - G.T).ravel()

    fc = function_caller()
    xk = np.zeros(M ** 2)
    gfk = G.ravel()
    pk = E.ravel()
    old_fval = current_loss
    alpha, _, _, new_fval, _, _ = line_search(fc.myf, fc.myfprime, xk, pk, gfk,
                                              old_fval, maxiter=n_ls_tries)
    if alpha is not None:
        # alpha is the new E, return its expm
        #Enew = alpha * pk.reshape(E.shape)
        #transform = expm(Enew)
        #X_new = np.dot(transform, X)
        transform = fc.transform
        Xs_new = fc.new_Xs
        obj = new_fval
        return transform, Xs_new, True, obj
    else:
        return 0, 0, False, 0
