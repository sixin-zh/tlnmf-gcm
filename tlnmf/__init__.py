"""Transform Learning - NMF"""
__version__ = '0.1'  # noqa

from .tl_nmf_batch import tl_nmf_batch # gcm model, mle loss, batch samples
from .tl_nmf_gcm_newton import tl_nmf_gcm_newton # gcm model, mle loss in expectation

from .utils import signal_to_frames, unitary_projection, synthesis_windowing  # noqa
import numpy as np

np.seterr(all='raise')
