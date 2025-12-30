import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange
from libc.math cimport sqrt,sin,cos,fabs

import os
import pickle
import pyfftw
import logging

logger = logging.getLogger(__name__)

def FFT3Dr(
    delta,
    direction='forward',
    threads=8,
    use_wisdom=False,
    fft_plan=None,
    fft_in=None,
    fft_out=None,
):
    """
    Compute a real-to-complex 3D FFT using pyFFTW.

    Parameters
    ----------
    delta : ndarray
        Real-space field to transform.
    direction : str
        Direction of FFTW (forward or backward)
    threads : int
        Number of FFTW threads.
    use_wisdom : bool
        Whether to load/save FFTW wisdom.
    fft_plan : pyfftw.FFTW or None
        Existing FFTW plan (for reuse).
    fft_in, fft_out : ndarray or None
        Pre-allocated FFT input/output arrays.

    Returns
    -------
    delta_fft: ndarray
        Fourier-transformed field.
    fft_plan : tuple
        FFTW plan (for reuse).
    fft_in, fft_out : ndarray
        FFT input/output arrays (for reuse).
    """

    if direction not in ['forward', 'backward']:
        raise Exception("FFTW direction not recognised. Select either 'forward' or 'backward'.")

    if fft_plan is None:

        nx, ny, nz = delta.shape
        
        if direction == 'forward': 
            fft_in = pyfftw.empty_aligned(delta.shape, dtype="float32")
            fft_out = pyfftw.empty_aligned((nx, ny, nz // 2 + 1),
                                           dtype="complex64")
        else:
            fft_in = pyfftw.empty_aligned(delta.shape, dtype="complex64")
            fft_out = pyfftw.empty_aligned((nx, ny, (nz - 1) * 2),
                                           dtype="float32")

        # load wisdom
        wisdom_fn = f"{{}}fft_wisdom_nx{nx}_ny{ny}_nz{nz}.txt"
        wisdom_fn = wisdom_fn.format("" if direction == "forward" else "i")
        wisdom_fn = os.path.join("wisdom", wisdom_fn)
        if use_wisdom and os.path.exists(wisdom_fn):
            logger.info(f"Importing FFT wisdom from {wisdom_fn}")
            with open(wisdom_fn, "rb") as f:
                pyfftw.import_wisdom(pickle.load(f))

        # create plan
        logger.debug("Creating FFT plan")
        fft_plan = pyfftw.FFTW(
            fft_in,
            fft_out,
            axes=(0, 1, 2),
            direction="FFTW_{}".format(direction.upper()),
            threads=threads,
            flags=("FFTW_MEASURE",) if use_wisdom else ("FFTW_ESTIMATE",),
        )

        # save wisdom
        if use_wisdom and not os.path.exists(wisdom_fn):
            logger.info(f"Saving FFT wisdom to {wisdom_fn}")
            os.makedirs("wisdom", exist_ok=True)
            with open(wisdom_fn, "wb") as f:
                pickle.dump(pyfftw.export_wisdom(), f)

    # execute FFT
    logger.debug(
        "Computing forwards FFT using {}".format(
            "MEASURE" if use_wisdom else "ESTIMATE"
        )
    )
    fft_in[:] = delta
    fft_plan()
    delta_fft = fft_out.copy()

    return delta_fft, fft_plan, fft_in, fft_out


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def tophat_smoothing(
    delta,
    radius,
    cellsize,
    threads=8,
    use_wisdom=False,
    fft_plan=None,
    ifft_plan=None,
    fft_in=None,
    fft_out=None,
    ifft_in=None,
    ifft_out=None,
):
    r"""
    Smooth density field with top-hat filter.

    Parameters
    ----------
    delta : ndarray
        Real-space field to transform.
    radius: float
        Smoothing radius of top-hat filter.
    threads : int
        Number of FFTW threads.
    use_wisdom : bool
        Whether to load/save FFTW wisdom.
    fft_plan, ifft_plan : pyfftw.FFTW or None
        Existing FFTW plans (for reuse).
    fft_in, fft_out, ifft_in, ifft_out : ndarray or None
        Pre-allocated FFT input/output arrays.

    Returns
    -------
    delta_sm : ndarray
        Density field delta smoothed by radius R.
    fft_plan, ifft_plan : pyfftw.FFTW or None
        FFTW plans (for reuse).
    fft_in, fft_out, ifft_in, ifft_out : ndarray or None
        FFT input/output arrays (for reuse).
    """

    cdef float prefact,kR,fact
    cdef int i, j, k, xdim, ydim, zdim
    cdef np.float32_t[::1] kkx, kky, kkz
    cdef np.complex64_t[:,:,::1] delta_k

    xdim, ydim, zdim = delta.shape

    # compute FFT of field
    delta_k, fft_plan, fft_in, fft_out = FFT3Dr(delta, 
                                                use_wisdom=use_wisdom, 
                                                fft_plan=fft_plan, 
                                                fft_in=fft_in, 
                                                fft_out=fft_out)

    # loop over Fourier modes
    logger.debug('Looping over fourier modes')
    kkx = np.fft.fftfreq(xdim, cellsize).astype('f4')**2
    kky = np.fft.fftfreq(ydim, cellsize).astype('f4')**2
    kkz = np.fft.rfftfreq(zdim, cellsize).astype('f4')**2

    prefact = 2.0 * np.pi * radius
    for i in prange(xdim, nogil=True):
        for j in range(ydim):
            for k in range(zdim//2 + 1):

                # skip when kx, ky and kz equal zero
                if i==0 and j==0 and k==0:
                    continue 

                # compute the value of |k|
                kR = prefact * sqrt(kkx[i] + kky[j] + kkz[k])
                if fabs(kR)<1e-5:  fact = 1.
                else:              fact = 3.0*(sin(kR) - cos(kR)*kR)/(kR*kR*kR)
                delta_k[i,j,k] =  fact * delta_k[i,j,k]

    delta_sm, ifft_plan, ifft_in, ifft_out = FFT3Dr(delta_k, 
                                                    direction='backward',
                                                    use_wisdom=use_wisdom, 
                                                    fft_plan=ifft_plan, 
                                                    fft_in=ifft_in, 
                                                    fft_out=ifft_out)

    return delta_sm, fft_plan, fft_in, fft_out, ifft_plan, ifft_in, ifft_out 

