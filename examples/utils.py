try:
    import cPickle as pickle
except ImportError:  # python 3.x
    import pickle

import scipy.signal as sig
import scipy
import soundfile as sf
import scipy.io as sio

from tlnmf import synthesis_windowing

def regress_atom(phi, alpha0=0, psi0=0, repeat=300):
    # solve MSE fit of obj = ( phi(m) - amp*cos(alpha * m + psi) )**2
#    print('regress phi', phi)
    M = phi.shape[0]
    amp0 = np.max(phi)
    arrm = np.arange(M)
#    print(arrm)
    def loss(x):
        alpha = x[0]
        psi = x[1]
        amp = x[2]
        diff = phi - amp*np.cos(alpha*arrm+psi)
        obj = np.sum(diff**2)
#        grad = np.zeros(2)
        
#        for m in range(M):
#            diff = phi[m] - amp*np.cos(alpha*m+psi)
#            obj += diff**2
#            grad[0] += 2*diff*(np.sin(alpha*m+psi))*m
#            grad[1] += 2*diff*(np.sin(alpha*m+psi))

        return obj # , grad
    
    # call a optimizer
    x0 = np.zeros(3)
    x0[0] = alpha0
    x0[1] = psi0
    x0[2] = amp0
    res = scipy.optimize.minimize(loss,x0)
    rep = 0
    bestloss = loss(x0)
    while rep < repeat:
        x0[0] = alpha0 + np.random.rand()*2*np.pi
        x0[1] = psi0 + np.random.rand()*2*np.pi
        x0[2] = amp0 * (1 + 0.1*np.random.rand())
        res = scipy.optimize.minimize(loss,x0,method='BFGS') # SLSQP')
        if res.success is True:
            rep += 1
            if loss(res.x) < bestloss:
                bestloss = loss(res.x)
                bestx = res.x
    x = bestx
    #print('best alpha',x[0])
    return x[0], x[1], x[2]

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

from scipy import fftpack
import numpy as np

def getwin(winid):
    if winid == 1:
        win = ('tukey',0.5)
    elif winid == 2:
        win = ('tukey',0.8)
    elif winid == 3:
        win = ('tukey',0.2)
    elif winid == 4:
        win = ('tukey',0.1)
    return win

def compute_dct_spectrum(Y):
    F = Y.shape[0]
    Phi = fftpack.dct(np.eye(F), 3, norm='ortho')
    X = np.dot(Phi, Y)
    V = X ** 2
    return V

def frames_to_signal(Y,signal,fs,ws):
    n_samples = len(signal)
    n_window_samples = 2 * np.floor(ws * fs / 2)
    N_box = int(np.ceil(n_samples / n_window_samples))
    n_window_samples = int(n_window_samples)
    y = synthesis_windowing(Y, n_window_samples, N_box)
    return y.reshape(n_samples)

def seperate_signal_from_WH_sci(Y,Phi,W,H,K,eps_nmf,win,fs,segsize,FOL,outfol,name):
    X1 = np.dot(Phi, Y)
    V_hat = np.dot(W, H)
    #plt.figure()
    for k in range(K):
        mask = (W[:,k:k+1] @H[k:k+1,:]+eps_nmf/K) / (V_hat+eps_nmf) # avoid underflow in @
        Zxx_k = X1 * mask
        Y_k = Phi.transpose() @ Zxx_k
        X_k = scipy.fft(Y_k, axis=0)
        iT_,y_k = sig.istft(X_k,fs,window=win,nperseg=segsize,input_onesided=False)
        y_k = np.real(y_k)
        #print('y_k',y_k.shape)
        y_k = y_k.reshape((np.size(y_k),1))
        sio.savemat(FOL + outfol + '/' + name + '_piece' + str(k) + '.mat', {'y_k':y_k, 'fs':fs, 'mask':mask})
        sf.write(FOL + outfol + '/' + name + '_piece' + str(k) + '.wav', y_k, fs)
        
        #axk=plt.subplot(1, K, k+1)
        #plt.imshow(mask)
        #plt.colorbar()
        #plt.gca().invert_yaxis()
        #plt.xlabel('n')
        #plt.ylabel('f')
        #plt.title('mask k=' + str(k+1))
    
    #plt.savefig('./results/' + outfol + '/' + name +  '_masks.png') 
