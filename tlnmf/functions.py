# Authors: Pierre Ablin <pierre.ablin@inria.fr>
#          Dylan Fagot
#
# License: MIT
import numpy as np


#eps = 1e-15


def is_div(A, B, eps):
    '''
    Computes the IS divergence
    '''
    M, N = A.shape
    f = (A + eps) / (B + eps)
    return np.sum(f - np.log(f)) - M * N

def grad_is_div(X,Y,V_hat,eps):
    # compute Delta(f,t)
    F = X.shape[0]
    T = Y.shape[0]
    N = X.shape[1]
    #print(Y.shape)
    #print(F,T,N)
    Delta = np.zeros((F,T))
    for f in range(F):
        for t in range(T):
            for n in range(N):
                Delta[f,t] += X[f,n]*Y[t,n]*(1/(V_hat[f,n]+eps)-1/(X[f,n]**2+eps))
    Delta = 2 * Delta
    return Delta
    
def new_is_div(A,B, eps):
    M, N = A.shape
    f = (A + eps) / (B + eps)
    return 0.5*np.sum(f + np.log(B+eps))

def grad_new_is_div(X,Y,V_hat,eps): # for new_is_div
    # compute Delta(f,t)
    F = X.shape[0]
    T = Y.shape[0]
    N = X.shape[1]
    #print(Y.shape)
    #print(F,T,N)
    Delta = np.zeros((F,T))
    for f in range(F):
        for t in range(T):
            for n in range(N):
                Delta[f,t] += X[f,n]*Y[t,n]/(V_hat[f,n]+eps)
    return Delta
    
def penalty(H, regul_type):
    if regul_type == 'sparse':
        return np.sum(H)
    else:
        _, N = H.shape
        return is_div(H[:, :N-1], H[:, 1:])
