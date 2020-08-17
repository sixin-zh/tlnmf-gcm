import matplotlib.pyplot as plt
import numpy as np

def draw_WH(W,H,k=0):
    # Plot the W and H of NMF
    f, ax = plt.subplots(*(1,2))
    #print(W.transpose()[0])
    ax[0].plot((W.transpose()[k]))
    #ax.hold()
    #print(wf)
    ax[0].set_title('w_fk (k=%d)' % (k))
    ax[0].set_xlabel('f')

    ax[1].plot(H[k])
    ax[1].set_title('h_kn (k=%d)' % (k))
    ax[1].set_xlabel('n')
    #ax[1].set_ylim(0,1.5*np.max(H))

def draw_atoms(Phi0,title,F=8):
    F0 = Phi0.shape[0]
    shape_to_plot = (int(F/2), 2)
    n_atoms = np.prod(shape_to_plot)
    idx_to_plot = np.arange(F0)
    #print(idx_to_plot)
    f, ax = plt.subplots(*shape_to_plot)
    f.suptitle(title)
    cmin = min(np.min(Phi0), -np.max(Phi0))
    cmax = -cmin
    for axe, idx in zip(ax.ravel(), idx_to_plot):
        axe.plot(Phi0[idx])
        axe.set_title('atom id='+str(idx))
        axe.set_ylim(ymin=cmin,ymax=cmax)
#    plt.show()
