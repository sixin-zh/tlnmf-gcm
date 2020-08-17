from time import time
from os import path, mkdir

import matplotlib as mpl
mpl.use('Agg')
FONT_SIZE = 20

import matplotlib.pyplot as plt
import numpy as np
import pickle

import soundfile as sf
from scipy import fftpack
import scipy.io as sio
import scipy.signal as sig
import scipy

from tlnmf.nmf import update_nmf_sparse
from tlnmf.functions import new_is_div, penalty
from tlnmf import tl_nmf_batch # , signal_to_frames, synthesis_windowing
from utils import getwin,  seperate_signal_from_WH_sci, regress_atom
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-sn', '--signal_name', type = str, default = "stationary5")
parser.add_argument('-K', '--rank', type = int, default = 1)
parser.add_argument('-hla','--Hlambda', type = float, default = 0)
parser.add_argument('-ws', '--window_size', type = float, default = 40e-3)
parser.add_argument('-epsnmf', '--eps_nmf', type = float, default = 2.2204e-16)
parser.add_argument('-itl', '--iter_tl', type = int, default = 500)
parser.add_argument('-toltl', '--tol_tl', type = float, default = 1e-5)
parser.add_argument('-pertl', '--iter_pertl', type = int, default = 5)
parser.add_argument('-runid', '--run_id', type = int, default = 1)
parser.add_argument('-win', '--window', type = int, default = 1) # win id for getwin
parser.add_argument('-inmf', '--iter_nmf', type = int, default = 0) # NMF for seperation
parser.add_argument('-cache', '--use_cache', type = int, default = 1) # to load saved results if 1
parser.add_argument('-phiinit','--phi_init', type = str, default = None)

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

K = args.rank
eps_nmf = args.eps_nmf
phi_init = args.phi_init

if args.Hlambda == 0:
    if phi_init == None:
        name = 'tlnmf2_sci_batch_' + sn + '_K' + str(K) + '_eps' + str(eps_nmf)
    else:
        name = 'tlnmf2_sci_batch_' + sn + '_K' + str(K) + '_eps' + str(eps_nmf) + '_phiinit' + str(phi_init)
else:
    assert(0)

runid = args.run_id
print('name=', name)
print('runid=', runid)
outfol = sn + '_win' + str(args.window) + '_ws' + str(int(ws*1000)) + 'ms_run' + str(runid)
FOL = './results_pertl' + str(args.iter_pertl) + '/'
if not path.exists(FOL + outfol):
    mkdir(FOL + outfol)

for l in range(L):
    signal = ys[l,:]

    win = getwin(args.window)
    iF,iT,X = sig.stft(signal,fs,window=win,nperseg=segsize,return_onesided=False)
    Y = np.real(scipy.ifft(X,axis=0))

    if l==0:
        M = Y.shape[0]
        N = Y.shape[1]
        F = M
        Ys = np.zeros((L,F,N))

        # plot y(t)
        plt.figure(figsize=(24, 6), dpi=80, facecolor='w', edgecolor='k')
        plt.plot(signal)
        plt.title('y(t)',size=FONT_SIZE*1.5)
        plt.xlabel('t',size=FONT_SIZE)
        ax = plt.gca()
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(FONT_SIZE*1.2)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(FONT_SIZE*1.2)
        plt.savefig(FOL + outfol + '/' + name + '_signal.eps')
        plt.show()
        
    Ys[l,:,:] = Y

print('Ys',Ys.shape)
regul = args.Hlambda * float(K) / M
print('regul',regul)

cache = args.use_cache
if not path.exists(FOL + outfol + '/' + name + '.pkl') or cache==0:
    t0 = time()
    Phi, W, H, Phi_init, infos = tl_nmf_batch(Ys, K, verbose=True, rng=rng, Phi = phi_init, \
                                              max_iter=args.iter_tl, n_iter_tl=args.iter_pertl,\
                                              tol=args.tol_tl, regul=regul, eps_nmf=eps_nmf)
    fit_time = time() - t0
    print('fit time',fit_time)
    # save Phi,W,H,Phi_init,infos into pickle
    ckpt = {}
    ckpt['Phi'] = Phi
    ckpt['W'] = W
    ckpt['H'] = H
    ckpt['Phi_init']=Phi_init
    ckpt['infos'] = infos
    ckpt['fit_time'] = fit_time
    pickle.dump(ckpt, open(FOL + outfol + '/' + name + '.pkl', "wb"))
else:
    # load saved results
    ckpt = pickle.load( open(FOL + outfol + '/' + name + '.pkl', "rb" ) )
    Phi = ckpt['Phi']
    W = ckpt['W']
    H = ckpt['H']
    Phi_init = ckpt['Phi_init']
    infos = ckpt['infos']
    fit_time = ckpt['fit_time']

# plot time objectives
plt.figure()
obj_list = infos['obj_list']
print('final obj value is', obj_list[-1])
t = np.linspace(0, fit_time, len(obj_list))
plt.plot(t, obj_list)
plt.xlabel('Time (sec.)')
plt.ylabel('Objective function')
plt.savefig(FOL + outfol + '/' + name + '_timeobj.png')

# Plot the most important atoms:
Xs = np.matmul(Phi, Ys)
powers = np.sum(Xs**2,axis=2)
power = np.mean(powers,axis=0)
shape_to_plot = (4, 2)
n_atoms = np.prod(shape_to_plot)
idx_sorted = np.argsort(power)
idx_to_plot = idx_sorted[-n_atoms:][::-1]
Phis = Phi[idx_to_plot,:]
print('Phis',Phis.shape)

plt.figure()
f, ax = plt.subplots(*shape_to_plot)
f.set_size_inches(18, 12)

for axe, idx in zip(ax.ravel(), range(0,n_atoms)):
    axe.plot(Phis[idx])
    alpha,psi,amp = regress_atom(Phis[idx])
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

plt.savefig(FOL + outfol + '/' + name + '_atoms.png',  dpi=80)

# plot the atoms in fourier domain
plt.figure()
f, ax = plt.subplots(*shape_to_plot)
f.set_size_inches(18, 12)

fftlabel0 = np.arange(0,M)*2*np.pi/M
fftlabel = np.arange(-M/2,M/2)*2*np.pi/M
for axe, idx in zip(ax.ravel(), range(0,n_atoms)):
    fftabs = np.abs(np.fft.fft(Phis[idx,:]))
    axe.plot(fftlabel,np.fft.fftshift(fftabs))
    axe.title.set_text(str(idx))
    # print out peak freq.
    idx_fft = np.argsort(fftabs)
    print('peak freq of atom', idx, 'is', fftlabel0[idx_fft[-1]])
    print('peak freq of atom', idx, 'is', fftlabel0[idx_fft[-2]])

plt.savefig(FOL + outfol + '/' + name + '_atoms_fft.png',  dpi=80)

# plot W,H
V = np.mean(np.matmul(Phi, Ys) ** 2, axis=0)  # final spectrogram
if args.iter_nmf > 0:
    niter = args.iter_nmf
    regul = args.Hlambda * float(K) / M
    print('run extra NMF with regul',regul)
    regul_type = 'sparse'
    W = np.abs(rng.randn(F, K)) + 1.
    W = W / np.sum(W, axis=0)
    H = np.abs(rng.randn(K, N)) + 1.
    V_hat = np.dot(W, H) # + eps  # Initial factorization
    for ite in range(niter):
        W, H = update_nmf_sparse(V, W, H, V_hat, regul, eps=eps_nmf)
        V_hat = np.dot(W, H)  # Initial factorization

# reorder W from energy
V_hat = np.matmul(W, H) # final factorization
Ws = W[idx_sorted[::-1],:]

#plt.figure()
plt.figure(figsize=(18, 6), dpi=80, facecolor='w', edgecolor='k')
ax = plt.subplot(121)
assert(K==2)
plt.plot(range(1,F+1),Ws[:,0],'o',markersize=12)
plt.plot(range(1,F+1),Ws[:,1],'.',markersize=12)
plt.title('w_{mk}',size=FONT_SIZE*1.5)
plt.xlabel('m',size=FONT_SIZE)
plt.grid('on')
plt.xlim(0,12)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(FONT_SIZE)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(FONT_SIZE)
    
plt.legend(['k=1','k=2'],prop={"size":FONT_SIZE})

ax = plt.subplot(122)
plt.plot(range(1,N+1),H.T[:,0],'-')
plt.plot(range(1,N+1),H.T[:,1],'--')
plt.title('h_{kn}',size=FONT_SIZE*1.5)
plt.xlabel('n',size=FONT_SIZE)
plt.xlim(0,N)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(FONT_SIZE)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(0)

plt.savefig(FOL + outfol + '/' + name +  '_WH.png')
sio.savemat(FOL + outfol + '/' + name + '_Wsorted.mat', {'Ws':Ws,'H':H,'Phis':Phis,'y':ys[0,:]})

# compute NMF decomposition of V = |Phi*Y|^2 = W*H
plt.figure()
plt.subplot(121)
cmax = np.max(np.log10(V+eps_nmf))
plt.imshow(np.log10(V+eps_nmf),vmin=cmax-4,vmax=cmax)
plt.title('log V')
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

# seperate one example into K comopnents
if args.window > 0:
    for l in range(L):
        Y = Ys[l,:,:]
        seperate_signal_from_WH_sci(Y,Phi,W,H,K,eps_nmf,win,fs,segsize,FOL,outfol,name + '_l' + str(l))

plt.show()
