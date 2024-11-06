import numpy as np
import pyfftw
import logging

# def setup_logging(name, level='info'):
    # logging.basicConfig(level=logging.CRITICAL,
                        # format='%(asctime)s | %(levelname)s | %(filename)s | %(funcName)s | Ln%(lineno)d | %(message)s',
                        # datefmt='%H:%M:%S')
    # logger = logging.getLogger(__name__)

    # if level == "debug":
        # level = logging.DEBUG
    # elif level == "warn":
        # level = logging.WARN
    # else:
        # level = logging.INFO

    # logger.setLevel(level)


def FFT3Dr_f(np.ndarray[np.float32_t,ndim=3] delta, int threads, bool double_prec=False):

    # align arrays
    delta_in  = pyfftw.empty_aligned(delta.shape, dtype='float{}'.format(64 if double_prec else 32))
    delta_out = pyfftw.empty_aligned((delta.shape[0], delta.shape[1], delta.shape[2]//2+1),
                                      dtype='complex{}'.format(128 if double_prec else 64))

    # plan FFTW
    fftw_plan = pyfftw.FFTW(delta_in, delta_out, axes=(0,1,2),
                            flags=('FFTW_ESTIMATE',),
                            direction='FFTW_FORWARD', threads=threads)
                            
    # put input array into delta_r and perform FFTW
    delta_in [:] = delta;  fftw_plan(delta_in, delta_out);  return delta_out


def IFFT3Dr_f(np.complex128_t[:,:,::1] delta, int threads, bool double_prec=False):

    # align arrays
    delta_in = pyfftw.empty_aligned(delta.shape, dtype='complex{}'.format(128 if double_prec else 64))
    delta_out  = pyfftw.empty_aligned([len(delta)] * 3, dtype='float{}'.format(64 if double_prec else 32))

    # plan FFTW
    fftw_plan = pyfftw.FFTW(delta_in, delta_out, axes=(0,1,2),
                            flags=('FFTW_ESTIMATE',),
                            direction='FFTW_FORWARD', threads=threads)
                            
    # put input array into delta_r and perform FFTW
    delta_in [:] = delta;  fftw_plan(delta_in, delta_out);  return delta_out
