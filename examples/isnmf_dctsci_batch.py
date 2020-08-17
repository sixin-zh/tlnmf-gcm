from time import time
from os import path, mkdir

import matplotlib as mpl
mpl.use('Agg')
FONT_SIZE = 20

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

import scipy.signal as sig
import scipy

from scipy import fftpack

import soundfile as sf
from tlnmf.nmf import update_nmf_sparse
from tlnmf.functions import is_div 
from utils import getwin, seperate_signal_from_WH_sci, regress_atom

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-sn', '--signal_name', type = str, default = "stationary5")
parser.add_argument('-K', '--rank', type = int, default = 3)
parser.add_argument('-hla','--Hlambda', type = float, default = 0)
parser.add_argument('-ws', '--window_size', type = float, default = 40e-3)
parser.add_argument('-inmf', '--iter_nmf', type = int, default = 500)
parser.add_argument('-epsnmf', '--eps_nmf', type = float, default = 2.2204e-16)
parser.add_argument('-runid', '--run_id', type = int, default = 1)
parser.add_argument('-win', '--window', type = int, default = 1) # id for differnt window
parser.add_argument('-cache', '--use_cache', type = int, default = 1) # to load saved results if 1
parser.add_argument('-dct', '--dct_type', type = int, default = 2) # dct type 1,2,3,4

args = parser.parse_args()

np.seterr(under='warn')

rng = np.random # .RandomState(0)
# Read the song
sn = args.signal_name
mat_adr = 'datasets/' + sn + '.mat'
print('load mat data at:',mat_adr)

mat_dic = sio.loadmat(mat_adr)
ys = mat_dic['ys']
fs = mat_dic['fs'][0]

print('signals',ys.shape)
print('fs',fs)
L = ys.shape[0] # number of samples

ws = args.window_size # 40e-3
segsize = int(ws*fs)
for l in range(L):
    signal = ys[l,:]
    win = getwin(args.window)

    iF,iT,X = sig.stft(signal,fs,window=win,nperseg=segsize,return_onesided=False)
    Y = np.real(scipy.ifft(X,axis=0))

    if l==0:
        M = X.shape[0]
        N = X.shape[1]
        F = M
        Ys = np.zeros((L,F,N),dtype=np.complex128)
        #plt.imshow(np.abs(Y))
        #plt.title('|y_n(m)|: sample 1')
        #plt.xlabel('n')
        #plt.ylabel('m')
        #plt.show()
        
    Ys[l,:,:] = Y

# NMF under DCT basis on a batch of samples
K = args.rank
niter = args.iter_nmf
eps_nmf = args.eps_nmf

F = Y.shape[0]
N = Y.shape[1]
M = F
print('M',M)
regul = args.Hlambda * float(K) / M
dct = args.dct_type

if regul == 0:
    name = 'isnmf_dctsci_batch_' + sn + '_K' + str(K) + '_eps' + str(eps_nmf)
else:
    assert(0)

runid = args.run_id
print('name=', name)
print('runid=', runid)
outfol = sn +'_win' + str(args.window) + '_ws' + str(int(ws*1000)) + 'ms_run' + str(runid)
FOL = './results' + '_dct' + str(dct) + '/'
if not path.exists(FOL + outfol):
    mkdir(FOL + outfol)

Phit = fftpack.dct(np.eye(F), dct, norm='ortho')
Phi = Phit.T # use the right DCT type (each row of Phi is an atom of DCT)

# compute NMF decomposition of V = |Phi*Y|^2 = W*H
Xs = np.matmul(Phi, Ys)
V = np.mean(np.abs(Xs) ** 2, axis=0)  # target spectrogram
cache = args.use_cache
if not path.exists(FOL + outfol + '/' + name + '_W.npy') or cache==0:
    print('run NMF with regul',regul)
    regul_type = 'sparse'
    W = np.abs(rng.randn(F, K)) + 1.
    W = W / np.sum(W, axis=0)
    H = np.abs(rng.randn(K, N)) + 1.
    V_hat = np.dot(W, H) # + eps  # Initial factorization
    for ite in range(niter):
        W, H = update_nmf_sparse(V, W, H, V_hat, regul, eps=eps_nmf)
        V_hat = np.dot(W, H)  # Initial factorization
    
    # save W and H
    np.save(FOL + outfol + '/' + name + '_W.npy', W)
    np.save(FOL + outfol + '/' + name + '_H.npy', H)
else:
    W = np.load(FOL + outfol + '/' + name + '_W.npy')
    H = np.load(FOL + outfol + '/' + name + '_H.npy')
    V_hat = np.dot(W, H)

plt.figure()
plt.subplot(121)
cmax = np.max(np.log10(V+eps_nmf))
plt.imshow(np.log10(V+eps_nmf),vmin=cmax-4,vmax=cmax)
plt.title('log E(|Phi Y|^o2)')
plt.xlabel('n')
plt.ylabel('f')
plt.colorbar()
plt.gca().invert_yaxis()
plt.subplot(122)
plt.imshow(np.log10(V_hat+eps_nmf),vmin=cmax-4,vmax=cmax)
plt.title('log WH')
plt.colorbar()
plt.xlabel('n')
plt.ylabel('f')
plt.gca().invert_yaxis()

plt.savefig(FOL + outfol + '/' + name +  '_Vh.png')    

# sort W by energy
Xs = np.matmul(Phi, Ys)
powers = np.sum(Xs**2,axis=2)
power = np.mean(powers,axis=0)
idx_sorted = np.argsort(power)
V_hat = np.matmul(W, H) # final factorization
Ws = W[idx_sorted[::-1],:]

# plot sorted atoms
shape_to_plot = (4, 4)
n_atoms = np.prod(shape_to_plot)
idx_to_plot = idx_sorted[-n_atoms:][::-1]
Phis = Phi[idx_to_plot,:]
print('Phis',Phis.shape)

plt.figure()
f, ax = plt.subplots(*shape_to_plot)
f.set_size_inches(18, 12)

for axe, idx in zip(ax.ravel(), range(0,n_atoms)):
    alpha,psi,amp = regress_atom(Phis[idx])
    axe.plot(Phis[idx])
    fit = np.zeros(M)
    for m in range(M):
        fit[m] = amp*np.cos(alpha*m+psi)
    axe.plot(fit)
    axe.axis('off')
    opta = (alpha/np.pi) % 2 # alpha%(2*np.pi)/np.pi
    opta = min(opta, 2-opta) # use neg freq to make sure opta in 0 nad 1
    axe.title.set_text(str(opta))
    print('%g\t%d' % (opta/2*fs[0],idx+1))
    #print('%d\t%g' % (idx+1,opta/2*fs[0]))
#    print(idx+1,'\t',opta/2*fs[0])
#    axe.title.set_text(str ( (alpha/np.pi % 2 )))

plt.savefig(FOL + outfol + '/' + name + '_atoms.png',  dpi=80)

# plot the atoms in fourier domain
plt.figure()
f, ax = plt.subplots(*shape_to_plot)
f.set_size_inches(18, 12)

fftlabel = np.arange(-M/2,M/2)*2*np.pi/M
for axe, idx in zip(ax.ravel(), range(0,n_atoms)):
    fftabs = np.abs(np.fft.fft(Phis[idx,:]))
    #print(fftabs.shape)
    axe.plot(fftlabel,np.fft.fftshift(fftabs))
    axe.title.set_text(str(idx))
    #axe.axis('off')

plt.savefig(FOL + outfol + '/' + name + '_atoms_fft.png',  dpi=80)

# plot sorted W and H
plt.figure(figsize=(18, 6), dpi=80, facecolor='w', edgecolor='k')
ax = plt.subplot(121)
assert(K==2)
plt.plot(range(1,F+1),Ws[:,0],'o',markersize=12)
plt.plot(range(1,F+1),Ws[:,1],'.',markersize=12)
plt.title('w_{mk}',size=FONT_SIZE*1.5)
plt.xlabel('m',size=FONT_SIZE)
plt.grid('on')
plt.xlim(0,25)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(FONT_SIZE)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(FONT_SIZE)
    
plt.legend(['k=1','k=2'],prop={"size":FONT_SIZE})

ax = plt.subplot(122)
plt.plot(range(1,N+1),H.T[:,0],'-')
plt.plot(range(1,N+1),H.T[:,1],'--')
plt.title('h_{kn}', size=FONT_SIZE*1.5)
plt.xlabel('n', size=FONT_SIZE)
plt.xlim(0,N)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(FONT_SIZE)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(0)

plt.savefig(FOL + outfol + '/' + name +  '_WH.png')
sio.savemat(FOL + outfol + '/' + name + '_Wsorted.mat', {'Ws':Ws,'H':H})

# wiener separation
if args.window > 0:
    for l in range(L):
        Y = Ys[l,:,:]
        seperate_signal_from_WH_sci(Y,Phi,W,H,K,eps_nmf,win,fs,segsize,FOL,outfol,name + '_l' + str(l))

plt.show()
