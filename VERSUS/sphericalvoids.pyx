import os
from astropy.io import fits
# from utils cimport FFT3Dr_f, IFFT3Dr_f
import pyfftw
import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange, parallel
from libc.math cimport sqrt,pow,sin,cos,log,log10,fabs,round
import logging
cimport void_openmp_library as VOL

logger = logging.getLogger(__name__)

cdef class SphericalVoids:
    r"""
    Run spherical void-finding algorithm.

    Parameters
    ----------
    data_positions: array (N), Path
        Array of data positions (in cartesian or sky coordinates) or path to such positions.

    data_weights: array (N,3), default=None
        Array of data weights.

    random_positions: array (N,3), Path
        Array of random positions (in cartesian or sky coordinates) or path to such positions.

    random_weights: array (N), default=None
        Array of random weights.

    data_cols: list, default=None
        List of data/random position column headers. Fourth element is taken as the weights (if present). Defaults to ['RA','DEC','Z'] if randoms are provided and ['X','Y','Z'] if not.

    delta_mesh: array, Path
        If data_positions not provided, load density mesh directly from array or from path to pre-saved mesh FITS file. 

    mesh_args: dict
        Dictionary to hold cellsize, boxsize, boxcenter and box_like attributes. Must be provided if delta_mesh provided as array, else read from file.

    kwargs : dict
        Optional arguments for meshbuilder.DensityMesh object.
    """

    cdef object delta
    # cdef np.float32_t[:,:,::1] delta
    # cdef np.uint8_t[:,:,::1] survey_mask
    cdef public int[3] nmesh
    cdef public float cellsize
    cdef public float r_sep 
    cdef public float[3] boxsize
    cdef public float[3] boxcenter
    cdef public bint box_like
    cdef list data_cols
    cdef float[:] Radii
    cdef float void_delta
    cdef float void_overlap
    cdef int threads
    cdef public float[:,::1] void_position
    cdef public float[::1] void_radius
    cdef public float[:,::1] void_vsf
    
    def __init__(self, data_positions=None, data_weights=None, 
                 random_positions=None, random_weights=None, data_cols=None,
                 dtype='f4', reconstruct=None, recon_args=None, 
                 delta_mesh=None, mesh_args=None, **kwargs):

        cdef np.ndarray[np.float32_t, ndim=3] delta

        # create delta mesh
        if data_positions is not None:
            from .meshbuilder import DensityMesh
            mesh = DensityMesh(data_positions=data_positions, data_weights=data_weights,
                               random_positions=random_positions, random_weights=random_weights, data_cols=data_cols, 
                               dtype=dtype, reconstruct=reconstruct, recon_args=recon_args)
            mesh.create_mesh(**kwargs)
            self.delta = mesh.delta
            self.nmesh = mesh.nmesh
            self.cellsize = mesh.cellsize
            self.r_sep = mesh.r_sep
            self.boxsize = mesh.boxsize
            self.boxcenter = mesh.boxcenter
            self.box_like = mesh.box_like
        # load delta mesh
        elif delta_mesh is not None:
            if type(delta_mesh) is str:
                self.load_mesh(delta_mesh)
            else:
                if not all([arg in mesh_args for arg in ['cellsize', 'r_sep', 'boxsize', 'boxcenter', 'box_like']]):
                    raise Exception('cellsize, r_sep, boxsize, boxcenter and box_like must be provided in addition to delta mesh with mesh_args.')
                self.delta = delta_mesh
                self.nmesh = delta_mesh.shape
                self.cellsize = mesh_args['cellsize']
                self.r_sep = mesh_args['r_sep']
                self.boxsize = mesh_args['boxsize']
                self.boxcenter = mesh_args['boxcenter']
                self.box_like = mesh_args['box_like']
        # ensure either data or mesh is provided
        else:
            raise Exception('Either data_positions or delta_mesh must be provided')

    def load_mesh(self, mesh_fn):
        r"""
        Load pre-populated 3D mesh from FITS file.

        Parameters
        ----------
        mesh_fn: string
            Path to mesh.
        """

        f = fits.open(mesh_fn)
        # self.delta = f[0].data
        self.delta = f[0].data.byteswap().newbyteorder()
        self.nmesh = f[0].data.shape
        self.cellsize = f['cellsize'].data
        self.r_sep = f['r_sep'].data
        self.boxsize = f['boxsize'].data
        self.boxcenter = f['boxcenter'].data
        self.box_like = f['box_like'].data
        f.close()

    def FFT3Dr(self):
        logger.debug('Computing forwards FFT')

        # align arrays
        delta_in  = pyfftw.empty_aligned(self.delta.shape, dtype='float32')
        delta_out = pyfftw.empty_aligned((self.delta.shape[0], 
                                          self.delta.shape[1], 
                                          self.delta.shape[2]//2+1), dtype='complex64')

        # plan FFTW
        fftw_plan = pyfftw.FFTW(delta_in, delta_out, axes=(0,1,2),
                                flags=('FFTW_ESTIMATE',),
                                direction='FFTW_FORWARD', threads=self.threads)
                                    
        # put input array into delta_r and perform FFTW
        delta_in [:] = self.delta;  fftw_plan(delta_in, delta_out);  return delta_out

    def IFFT3Dr(self, np.complex64_t[:,:,::1] delta_k):
        logger.debug('Computing inverse FFT')

        # align arrays
        # delta_in = pyfftw.empty_aligned(self.delta.shape, dtype='complex64')
        # delta_out  = pyfftw.empty_aligned([len(self.delta)] * 3, dtype='float32')
        delta_in  = pyfftw.empty_aligned((self.delta.shape[0], 
                                          self.delta.shape[1], 
                                          self.delta.shape[2]//2+1), dtype='complex64')
        delta_out = pyfftw.empty_aligned(self.delta.shape, dtype='float32')

        # plan FFTW
        fftw_plan = pyfftw.FFTW(delta_in, delta_out, axes=(0,1,2),
                                flags=('FFTW_ESTIMATE',),
                                direction='FFTW_BACKWARD', threads=self.threads)
                                    
        # put input array into delta_r and perform FFTW
        delta_in [:] = delta_k;  fftw_plan(delta_in, delta_out);  return delta_out

    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    cdef _reset_survey_mask(self, np.float32_t[:,:,::1] delta_sm):
        r"""
        In-place operation that resets cells outside survey region to mean density.

        Parameters
        ----------
        delta_sm: array
            Smoothed overdensity mesh.
        """
        cdef int i, j, k
        cdef np.float32_t[:,:,::1] d_true = self.delta  
        
        logger.debug('Resetting survey mask')
        for i in range(self.nmesh[0]):
            for j in range(self.nmesh[1]):
                for k in range(self.nmesh[2]):
                    # if density field outside survey region then set to average density
                    if d_true[i,j,k] == 0.:
                        delta_sm[i,j,k] = 0.

        return delta_sm

    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    cdef _smoothing(self, float radius):
        r"""
        Smooth density field with top-hat filter

        Parameters
        ----------

        radius: float
            Smoothing radius of top-hat filter.
        """

        cdef float prefact,kR,fact
        cdef int i, j, k, xdim, ydim, zdim
        cdef np.float32_t[::1] kkx, kky, kkz
        cdef np.complex64_t[:,:,::1] delta_k
        
        xdim, ydim, zdim = self.nmesh
        # compute FFT of field
        delta_k = self.FFT3Dr()

        # loop over Fourier modes
        logger.debug('Looping over fourier modes')
        kkx = np.fft.fftfreq(xdim, self.cellsize).astype('f4')**2
        kky = np.fft.fftfreq(ydim, self.cellsize).astype('f4')**2
        kkz = np.fft.rfftfreq(zdim, self.cellsize).astype('f4')**2

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

        delta_sm = self.IFFT3Dr(delta_k)

        # reset survey mask
        if not self.box_like:
            delta_sm = self._reset_survey_mask(delta_sm)
            # self._reset_survey_mask(&delta_sm)

        return delta_sm


    def _sort_radii(self, float[:] radii):
        return np.sort(radii)[::-1]

    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    cdef inline long[::1] _find_underdensities(self, float[:,:,::1] delta_sm, char[:,:,::1] in_void, float[::1] delta_v, long[::1] IDs, 
                                               int xdim, int ydim, int zdim, int yzdim, long *local_voids):
        r"""
        Find underdense cells and sort them from lowest density.

        Parameters
        ----------

        in_void: array
            Array to indicate if cell belongs to a previously detected void.

        delta_v: array
            Array to store void central densities.

        IDs: array
            Array to store void ID numbers.
        """
        # cdef long local_voids=0
        cdef long[::1] indices, IDs_sort#, IDs
        cdef int i,j,k
        # cdef float[::1] delta_v#, delta_v_temp

        logger.debug(f'Looping through {delta_sm.size:d} cells to find underdensities and assigning IDs')
        for k in range(zdim):
            for j in range(ydim):
                for i in range(xdim):

                    if delta_sm[i,j,k]<self.void_delta and in_void[i,j,k]==0:
                        IDs[local_voids[0]] = yzdim*i + zdim*j + k
                        delta_v[local_voids[0]] = delta_sm[i,j,k]
                        local_voids[0] += 1

        logger.debug(f'Found {local_voids[0]} cells with delta < {self.void_delta:.2f}')

        # sort delta_v by density
        indices = np.argsort(delta_v[:local_voids[0]])
        # sort IDs by density
        IDs_sort = np.empty(local_voids[0], dtype=np.int64) 
        for i in range(local_voids[0]):
            IDs_sort[i] = IDs[indices[i]]
        for i in range(local_voids[0]):
            IDs[i] = IDs_sort[i]
        del IDs_sort
        logger.debug('Sorting of underdense cells finished.')

        return IDs

####################################################################
    # cdef inline _periodic_bounds(self, int dd, int dim, int lower, int upper):
        # r"""
        # In nearby_voids1: Wrap distances between void and candidate around box.
        # In nearby_voids2: Wrap candidate surrounding cell search around box.
        # """
        # if dd > upper:
            # dd -= dim
        # elif dd < lower:
            # dd += dim

        # return dd

    # def _calc_void_overlap(self, Rv, Rc, d):
        # r"""
        # Determine the volume overlap fraction between two spheres

        # Parameters
        # ----------

        # Rv: float
            # Radius of previously detected void.
        # Rc: float
            # Radius of void candidate.
        # d: float
            # Distance between void and candidate centres.
        # """
        # logger.debug('Calculating void volume overlap fraction')
        # # no overlap if voids are not touching
        # if d >= (Rv + Rc): return 0.

        # # expression from: https://mathworld.wolfram.com/Sphere-SphereIntersection.html
        # a = (Rv + Rc - d)**2
        # b = (d*d + 2*d*Rc - 3*Rc*Rc + 2*d*Rv + 6*Rc*Rv - 3*Rv*Rv)
        # c = 1 / (16 * d * Rc*Rc*Rc)

        # return a * b * c

    # @boundscheck(False)
    # @cython.cdivision(True)
    # @wraparound(False)
    # cdef int _nearby_voids1(self, long total_voids_found, int xdim, int ydim, int zdim,
                            # int i, int j, int k, float *void_radius, int *void_pos, float R_grid, int num_threads):
        # """
        # Identify if void candidate is a new void by computing distances to detected voids.
        # Returns 1 if another void has been detected nearby (accounting for void overlap).
        # """

        # cdef int l, dx, dy, dz, dist2, nearby_voids = 0
        # cdef float overlap_frac, tot_overlap = 0.

        # # loop over set of previously detected voids
        # for l in prange(total_voids_found, nogil=True, num_threads=num_threads):
            # # exit if void volume overlap greater than threshold 
            # if tot_overlap > self.void_overlap:
                # nearby_voids = 1
                # continue
            # # if nearby_voids > 0:
                # # continue

            # # compute distance from candidate (i,j,k) to known voids (void_pos)
            # dx = i - void_pos[3 * l]
            # if self.box_like: dx = self._periodic_bounds(dx, xdim, -xdim//2, xdim//2)

            # dy = j - void_pos[3 * l + 1]
            # if self.box_like: dy = self._periodic_bounds(dy, ydim, -ydim//2, ydim//2)

            # dz = k - void_pos[3 * l + 2]
            # if self.box_like: dz = self._periodic_bounds(dz, zdim, -zdim//2, zdim//2)

            # dist2 = dx * dx + dy * dy + dz * dz
            # dist = sqrt(dist2)

            # if distance to cell is less than combined candidate and detected void radii
            # if dist < (void_radius[l] + R_grid):
                # # count nearby voids in parallel
                # with gil:
                    # nearby_voids += 1

            # # sum amount of void overlap from nearby voids
            # overlap_frac = self._calc_void_overlap(void_radius[l], R_grid, dist)  # will be non-zero given above condition
            # with nogil:
                # tot_overlap += overlap_frac

        # return nearby_voids


    # @boundscheck(False)
    # @cython.cdivision(True)
    # @wraparound(False)
    # cdef int _nearby_voids2(self, int Ncells, int i, int j, int k, int xdim, int ydim, int zdim, int yzdim,
                               # float R_grid, float R_grid2, char *in_void, int num_threads):
        # """
        # Identify if void candidate is a new void by determining if surrounding cells belong to other voids.
        # Returns 1 if another void has been detected nearby (accounting for void overlap).
        # """

        # cdef int l, m, n, i1, j1, k1, nearby_voids = 0
        # cdef long num
        # cdef float dist2, tot_overlap = 0.

        # # loop over cells surrounding candidate void center
        # for l in prange(-Ncells, Ncells + 1, nogil=True, num_threads=num_threads):
            # # if nearby_voids > 0:
                # # continue
            # if tot_overlap > self.void_overlap:
                # nearby_voids = 1
                # continue

            # i1 = i + l
            # if self.box_like: i1 = self._periodic_bounds(i1, xdim, 0, xdim-1)

            # for m in range(-Ncells, Ncells + 1):
                # j1 = j + m
                # if self.box_like: j1 = self._periodic_bounds(j1, ydim, 0, ydim-1)

                # for n in range(-Ncells, Ncells + 1):
                    # k1 = k + n
                    # if self.box_like: k1 = self._periodic_bounds(k1, zdim, 0, zdim-1)
                    
                    # # skip if cell does not belong to another void
                    # num = yzdim*i1 + zdim*j1 + k1
                    # if in_void[num] == 0:
                        # continue
                    # # else mark as nearby void if cell is within candidate radius
                    # else:
                        # dist2 = l*l + m*m + n*n
                        # if dist2 < R_grid2:
                            # with nogil:
                                # tot_overlap += 3 / (4*np.pi*R_grid**3)  # cell fraction of candidate volume
                                # # nearby_voids += 1

        # return nearby_voids


    # @boundscheck(False)
    # @cython.cdivision(True)
    # @wraparound(False)
    # cdef void _mark_void_region(self, char *in_void, int Ncells, int xdim, int ydim, int zdim, int yzdim, 
                                # float R_grid2, int i, int j, int k, int num_threads):
        # cdef int l, m, n, i1, j1, k1
        # cdef long number
        # cdef float dist2
        
        # # add different xyz dimensions

        # # Use Cython's parallel loop (with OpenMP if configured in setup)
        # for l in prange(-Ncells, Ncells + 1, nogil=True, num_threads=num_threads):
            # i1 = i + l
            # if self.box_like: i1 = self._periodic_bounds(i1, xdim, 0, xdim-1)

            # for m in range(-Ncells, Ncells + 1):
                # j1 = j + m
                # if self.box_like: j1 = self._periodic_bounds(j1, ydim, 0, ydim-1)

                # for n in range(-Ncells, Ncells + 1):
                    # k1 = k + n
                    # if self.box_like: k1 = self._periodic_bounds(k1, zdim, 0, zdim-1)

                    # dist2 = l * l + m * m + n * n
                    # if dist2 < R_grid2:
                        # num = yzdim*i1 + zdim*j1 + k1
                        # in_void[num] = 1


####################################################################


    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    def run_voidfinding(self, list radii=[0.], float void_delta=-0.8, float void_overlap=0., int threads=0):
        r"""
        Run spherical voidfinding on density mesh.

        Parameters
        ----------

        radii: list 
            List of void radii to search for. Defaults to 4-104x cellsize.

        void_delta: float, default=-0.8
            Maximum overdensity threshold to be classified as void.

        void_overlap: float, default=0.
            Maximum allowed volume fraction of void overlap.
        """

        # cdef float[:] Radii=np.array(radii, dtype=np.float32)
        cdef np.ndarray[float, ndim=1] Radii=np.array(radii, dtype=np.float32)
        cdef float R, R_grid, R_grid2, Rmin
        cdef int bins, Ncells, nearby_voids, threads2 
        cdef long nmesh_tot=np.prod(self.nmesh)
        cdef long max_num_voids, voids_found, total_voids_found, ID
        cdef float vol_mesh, vol_void, norm
        cdef float[:,:,::1] delta_sm
        cdef char[:,:,::1] in_void
        cdef long[::1] IDs
        cdef int i, j, k, p, q, xdim, ydim, zdim, yzdim, mode
        cdef int[::1] Nvoids
        cdef double void_cell_fraction, void_volume_fraction=0.0
        cdef float[:,::1] position
        cdef int[:,::1] void_pos
        cdef float[::1] delta_v, void_rad, box_shift
        cdef float[:,::1] vsf

        cdef long local_voids
        cdef long[::1] indexes, IDs_temp
        cdef float[::1] delta_v_temp
        cdef int dims, dims2

        # set default radii if not provided
        if radii[0] == 0.:
            # ~2-10x cellsize in logarithmic spacing
            # self.Radii = 10**np.arange(1, 0.3, -0.005, dtype=np.float32) * self.cellsize  
            # self.Radii = np.logspace(1, 0.3, 28, dtype=np.float32) * self.cellsize  
            # self.Radii = np.linspace(15, 7, 25, dtype=np.float32) * self.cellsize
            # ~2-10x cellsize in reverse logarithmic spacing
            # self.Radii = (10 + 10**0.3 - np.logspace(0.3, 1, 28, dtype=np.float32)) * self.cellsize 

            # ~2-8x average galaxy separation in linear spacing
            Radii = np.linspace(8, 2, 19, dtype=np.float32) * self.r_sep
            # self.Radii = np.array(Radii)[np.array(Radii) > self.cellsize]  # ensure radii larger than cellsize
            self.Radii = Radii[Radii > self.cellsize]  # ensure radii larger than cellsize
            logger.debug(f'Radii set by default')
        else:
            # order input radii from largest to smallest
            self.Radii = self._sort_radii(Radii)
            logger.debug(f'Radii set manually')
        bins = self.Radii.size
        Rmin = np.min(self.Radii)

        # set maximum density threshold for cell to be classified as void
        self.void_delta = void_delta
        # set allowed void overlap for void classification
        self.void_overlap = void_overlap
        # set dimensions
        xdim, ydim, zdim = self.nmesh
        yzdim = ydim * zdim
        # set threads
        self.threads = os.cpu_count() if threads==0 else threads
        logger.info(f'Running spherical void-finder with {self.threads} threads')

        # check mesh resolution is equal along each axis

        # check that minimum radius is larger than grid resolution
        if Rmin<self.cellsize:
            raise Exception(f"Minimum radius {Rmin:.1f} is below cellsize {self.cellsize:.1f}")

        # determine mesh volume
        vol_mesh = self.cellsize**3 * nmesh_tot
        # determine non-overlapping volume of smallest void
        vol_void = (1 - self.void_overlap) * 4 * np.pi * Rmin**3 / 3
        # determine maximum possible number of voids
        max_num_voids = int(vol_mesh / vol_void)
        logger.debug(f"Total mesh cells = {nmesh_tot:d} ({xdim},{ydim},{zdim})")
        logger.debug(f"Maximum number of voids = {max_num_voids:d}")

        # define arrays containing void positions and radii
        void_pos    = np.zeros((max_num_voids, 3), dtype=np.int32)
        void_rad    = np.zeros(max_num_voids,      dtype=np.float32)

        # define the in_void and delta_v array
        in_void = np.zeros(self.nmesh, dtype=np.int8)
        delta_v = np.zeros(nmesh_tot,   dtype=np.float32)
        IDs     = np.zeros(nmesh_tot,   dtype=np.int64)

        # define the arrays needed to compute the VSF
        Nvoids = np.zeros(bins,   dtype=np.int32)
        vsf    = np.zeros((3, bins-1), dtype=np.float32)
        # Rmean  = np.zeros(bins-1, dtype=np.float32)

        # set function wrapping based on box-like
        if self.box_like:
            logger.debug("Using wrapped VF algorithms")
            num_voids_around1 = VOL.num_voids_around1_wrap
            num_voids_around2 = VOL.num_voids_around2_wrap
            mark_void_region  = VOL.mark_void_region_wrap
        else:
            logger.debug("Using VF algorithms with boundary conditions")
            num_voids_around1 = VOL.num_voids_around1
            num_voids_around2 = VOL.num_voids_around2
            mark_void_region  = VOL.mark_void_region

        # iterate through void radii
        total_voids_found = 0
        for q in range(bins):

            R = self.Radii[q]
            logger.debug(f'Smoothing field with top-hat filter of radius {R:.1f} Mpc/h')
            delta_sm = self._smoothing(R)  # single precision smoothing

            # check void cells are present at this radius
            if np.min(delta_sm)>self.void_delta:
                logger.info(f'No cells with delta < {self.void_delta:.2f} for R={R:.1f} Mpc/h')

                continue

            local_voids = 0
            IDs = self._find_underdensities(delta_sm, in_void, delta_v, IDs, xdim, ydim, zdim, yzdim, &local_voids)

            # determine void radius in terms of number of mesh cells
            R_grid = R/self.cellsize; Ncells = <int>R_grid + 1
            R_grid2 = R_grid * R_grid
            voids_found = 0 

            # select method to identify nearby voids based on radius
            mode = 0 if total_voids_found < (2*Ncells+1)**3 else 1
            threads2 = 1 if Ncells<12 else min(4, self.threads) #empirically this seems to be the best
            logger.debug(f'Setting threads2 = {threads2} (threads={self.threads})')
            logger.debug(f'Identifying nearby voids using mode {mode}')

            # identify nearby voids
            for p in range(local_voids):

                # find mesh coordinates of underdense cell
                ID = IDs[p]
                i,j,k = ID//yzdim, (ID%yzdim)//zdim, (ID%yzdim)%zdim

                # if cell belongs to a void continue
                if in_void[i,j,k] == 1: continue

                if mode==0:
                    # nearby_voids = self._nearby_voids1(total_voids_found, i, j, k, 
                                                       # xdim, ydim, zdim,
                                                       # &void_rad[0], &void_pos[0,0], 
                                                       # R_grid, threads2)
                    nearby_voids = num_voids_around1(self.void_overlap, total_voids_found, 
                                                     xdim, ydim, zdim, i, j, k, 
                                                     &void_rad[0], &void_pos[0,0], 
                                                     R_grid, threads2)

                else:
                    # nearby_voids = self._nearby_voids2(Ncells, i, j, k,
                                                       # xdim, ydim, zdim, yzdim,
                                                       # R_grid, R_grid2, 
                                                       # &in_void[0,0,0], threads2)
                    nearby_voids = num_voids_around2(self.void_overlap, Ncells, i, j, k, 
                                                     xdim, ydim, zdim, yzdim,
                                                     R_grid, R_grid2, 
                                                     &in_void[0,0,0], threads2)

                # if new void detected
                if nearby_voids == 0:
                    void_pos[total_voids_found, 0] = i
                    void_pos[total_voids_found, 1] = j
                    void_pos[total_voids_found, 2] = k
                    void_rad[total_voids_found] = R_grid

                    voids_found += 1; total_voids_found += 1
                    in_void[i,j,k] = 1

                    # self._mark_void_region(&in_void[0,0,0], Ncells, xdim, ydim, zdim, yzdim, 
                                           # R_grid2, i, j, k, 1)
                    
                    mark_void_region(&in_void[0,0,0], Ncells, xdim, ydim, zdim,
                                     yzdim, R_grid2, i, j, k, threads=1)

            logger.info(f'Found {voids_found} voids with radius R={R:.1f} Mpc/h')
            Nvoids[q] = voids_found 

            void_cell_fraction = np.sum(in_void, dtype=np.int64) * 1.0/nmesh_tot  # volume determined using filled cells
            void_volume_fraction += voids_found * 4.0 * np.pi * R**3 / (3.0 * vol_mesh) # volume determined using void radii
            logger.debug('Occupied void volume fraction = {:.3f} (expected {}{:.3f})'.format(void_cell_fraction, 
                                                                                             '<' if void_overlap>0. else '',
                                                                                             void_volume_fraction))

        logger.info(f'{total_voids_found} total voids found.')
        logger.info('Occupied void volume fraction = {:.3f} (expected {}{:.3f})'.format(void_cell_fraction, 
                                                                                        '<' if void_overlap>0. else '',
                                                                                        void_volume_fraction))
        # compute the void size function (dn/dlnR = # of voids/Volume/delta(lnR))
        for i in range(bins-1):
            norm = 1 / (np.prod(self.boxsize) * log(self.Radii[i] / self.Radii[i+1]))
            vsf[0,i] = sqrt(self.Radii[i] * self.Radii[i+1])  # geometric mean radius for logarithmic scale
            vsf[1,i] = Nvoids[i] * norm  # vsf
            vsf[2,i] = sqrt(Nvoids[i]) * norm  # poisson uncertainty
        

        # finish by setting the class fields
        position = np.asarray(void_pos[:total_voids_found], dtype=np.float32) + 0.5  # void positions on mesh
        box_shift = np.asarray(self.boxcenter, dtype=np.float32) - np.asarray(self.boxsize, dtype=np.float32)/2 
        # self.void_position = np.zeros_like(void_pos[:total_voids_found], dtype=np.float32)
        # for i in range(3):
            # self.void_position[i] = np.asarray(void_pos[:total_voids_found,i]) + 0.5 + self.boxcenter[i] - self.boxsize[i]/2
        self.void_position = position * np.asarray(self.cellsize, dtype=np.float32) + box_shift  # transform positions relative to data 
        # self.void_position = np.asarray(void_pos[:total_voids_found], dtype=np.float32) * self.cellsize  # relative to box not data
        self.void_radius   = np.asarray(void_rad[:total_voids_found]) * self.cellsize
        # self.Rbins         = np.asarray(Rmean)
        self.void_vsf      = np.asarray(vsf) 

