import numpy as np
import os
from astropy.io import fits
import utils
cimport numpy as np
cimport cython
from cython.parallel import prange, parallel
from libc.math cimport sqrt,pow,sin,cos,log,log10,fabs,round
import logging

logger = logging.getLogger(__name__)

cdef class SphericalVoids:
    r"""
    Run spherical void-finding algorithm.

    Parameters
    ----------
    radii: array
        Array of void radii to find.

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

    cellsize: float
        Size of mesh cell. Must be provided if delta_mesh provided as array, else read from file.

    kwargs : dict
        Optional arguments for meshbuilder.DensityMesh object.
    """
    cdef np.float32_t[:,:,::1] delta
    cdef np.uint8_t[:,:,::1] survey_mask
    cdef int[3] nmesh
    cdef int cellsize
    cdef char data_cols

    def __init__(self, data_positions=None, data_weights=None, 
                 random_positions=None, random_weights=None, data_cols=data_cols,
                 delta_mesh=None, cellsize=None, 
                 **kwargs):

        cdef int i, j, k

        # create delta mesh
        if data_positions is not None:
            from .meshbuilder import DensityMesh
            mesh = DensityMesh(data_positions=data_positions, data_weights=data_weights,
                               random_positions=random_positions, random_weights=random_weights, data_cols=data_cols)
            mesh.create_mesh(**kwargs)
            self.delta = mesh.delta
            self.nmesh = mesh.nmesh
            self.cellsize = mesh.cellsize
        # load delta mesh
        elif delta_mesh is not None:
            if type(delta_mesh) is str:
                self.load_mesh(delta_mesh)
            else:
                if cellsize is None: raise Exception('Cellsize must be provided in addition to delta mesh')
                self.delta = delta_mesh
                self.nmesh = delta_mesh.shape
                self.cellsize = cellsize
        # ensure either data or mesh is provided
        else:
            raise Exception('Either data_positions or delta_mesh must be provided')

        # determine survey mask from cells exactly equal to average density
        logger.info('Determining survey mask')
        self.survey_mask = np.zeros(self.delta.shape, dtype=np.uint8)
        for i in range(self.delta.shape[0]):
            for j in range(self.delta.shape[1]):
                for k in range(self.delta.shape[2]):
                    if self.delta[i,j,k] == 0.: continue
                    self.survey_mask[i,j,k] = 1
        print(self.survey_mask)

    def load_mesh(self, mesh_fn):
        r"""
        Load pre-populated 3D mesh from FITS file.

        Parameters
        ----------
        mesh_fn: string
            Path to mesh.
        """

        f = fits.open(mesh_fn)
        self.delta = f[0].data
        self.nmesh = f[0].data.shape
        self.cellsize = f['cellsize'].data
        f.close()


    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    cdef _smoothing(self, float radius, int threads=1, bint double_prec=False):
        r"""
        Smooth density field with top-hat filter

        Parameters
        ----------

        radius: float
            Smoothing radius of top-hat filter.

        threads: int
            Number of threads for use with multi-threading.

        double_prec: bool
            Option to use double precision density array.
        
        survey_mask: array
            Boolean array defining cells inside survey region.
        """

        cdef float prefact,kR,fact
        cdef int i, j, k, xdim, ydim, zdim
        cdef np.float32_t[::1] kkx, kky, kkz
        cdef np.complex64_t[:,:,::1] delta_k
        
        # cdef int kxx, kyy, kzz, kx, ky, kz, kx2, ky2, kz2
        # cdef int middle = self.nmesh[2]//2

        # define mesh dimension
        xdim, ydim, zdim = self.nmesh

        # compute FFT of field
        delta_k = utils.FFT3Dr_f(self.delta, threads, double_prec=double_prec) 

        # loop over Fourier modes
        kkx = np.fft.rfftfreq(self.nmesh[0], 1/self.cellsize)**2
        kky = np.fft.fftfreq(self.nmesh[1], 1/self.cellsize)**2
        kkz = np.fft.fftfreq(self.nmesh[2], 1/self.cellsize)**2
        prefact = 2.0 * np.pi * radius
        for k in prange(zdim, nogil=True):
            for j in range(ydim):
                for i in range(xdim):

                    # skip when kx, ky and kz equal zero
                    if i==0 and j==0 and k==0:
                        continue 

                    kR = prefact * sqrt(kkx[i] + kky[j] + kkz[k])
                    if fabs(kR)>1e-5:
                        delta_k[i,j,k] *= 3.0*(sin(kR) - cos(kR)*kR)/(kR*kR*kR)

        # for kxx in prange(zdim, nogil=True):
            # kx  = (kxx-zdim if (kxx>middle) else kxx)
            # kx2 = kx*kx
            
            # for kyy in range(ydim):
                # ky  = (kyy-ydim if (kyy>middle) else kyy)
                # ky2 = ky*ky
                
                # for kzz in range(middle+1): #kzz=[0,1,..,middle] --> kz>0
                    # kz  = (kzz-xdim if (kzz>middle) else kzz)
                    # kz2 = kz*kz

                    # if kxx==0 and kyy==0 and kzz==0:
                        # continue

                    # # compute the value of |k|
                    # kR = prefact*sqrt(kx2 + ky2 + kz2)
                    # if fabs(kR)<1e-5:  fact = 1.0
                    # else:              fact = 3.0*(sin(kR) - cos(kR)*kR)/(kR*kR*kR)
                    # delta_k[kxx,kyy,kzz] = delta_k[kxx,kyy,kzz]*fact

        delta_sm = utils.IFFT3Dr_f(delta_k, threads, double_prec=double_prec)

        # reset survey mask
        logging.debug('Resetting survey mask')
        delta_sm[self.survey_mask] = 0.

        return delta_sm

    def _sort_radii(self, float[:] radii):
        return np.sort(radii)[::-1]

    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    cdef inline long[::1] _find_underdensities(self, char[:,:,::1] in_void, float[::1] delta_v, long[::1] IDs, xdim, ydim, zdim, yzdim):
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
        cdef long local_voids
        cdef long[::1] indices, IDs_sort#, IDs
        cdef int i,j,k
        # cdef float[::1] delta_v#, delta_v_temp

        local_voids = 0
        for k in range(zdim):
            for j in range(ydim):
                for i in range(xdim):

                    if self.delta[i,j,k]<self.void_delta and in_void[i,j,k]==0:
                        IDs[local_voids] = yzdim*i + zdim*j + k
                        delta_v[local_voids] = self.delta[i,j,k]
                        local_voids += 1

        logger.info(f'Found {local_voids} cells with delta < {self.void_delta}')

        # sort delta_v by density
        indices = np.argsort(delta_v[:local_voids])
        # # this is just delta_v = delta_v[indexes]
        # delta_v_temp = np.empty(local_voids, dtype=np.float32)
        # for i in range(local_voids):
            # delta_v_temp[i] = delta_v[indexes[i]]
        # for i in range(local_voids):
            # delta_v[i] = delta_v_temp[i]
        # del delta_v_temp

        # sort IDs by density
        # this is just IDs = IDs[indices]
        IDs_sort = np.empty(local_voids, dtype=np.int64) 
        for i in range(local_voids):
            IDs_sort[i] = IDs[indices[i]]
        # for i in range(local_voids):
            # IDs[i] = IDs_sort[i]
        # del IDs_sort
        logger.debug('Sorting of underdense cells finished.')

        # return IDs
        return IDs_sort


    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    cdef run_voidfinding(self, float[:] radii, float void_delta=-0.8, float void_overlap=0., int threads=1):
        r"""
        Run spherical voidfinding on density mesh.

        Parameters
        ----------

        radii: list 
            List of void radii to search for.

        void_delta: float, default=-0.8
            Maximum overdensity threshold to be classified as void.

        void_overlap: float, default=0.
            Maximum allowed volume fraction of void overlap.
        """

        cdef int bins=radii.size, nmesh_tot=np.prod(self.nmesh), threads2
        cdef float Rmin=np.min(self.radii), vol_mesh, vol_void
        cdef long max_num_voids
        cdef int[:,::1] void_pos
        cdef float[::1] void_radius
        cdef char[:,:,::1] in_void
        cdef float[::1] delta_v, delta_v_temp
        cdef long[::1] IDs
        cdef long total_voids_found
        cdef int i, j, k, p, q, xdim, ydim, zdim, yzdim,

        # order input radii from largest to smallest
        self.radii = self._sort_radii(radii)
        # set maximum density threshold for cell to be classified as void
        self.void_delta = void_delta
        # set allowed void overlap for void classification
        self.void_overlap = void_overlap
        # set dimensions
        xdim, ydim, zdim = self.nmesh
        yzdim = ydim * zdim

        # check mesh resolution is equal along each axis

        # check that minimum radius is larger than grid resolution
        if Rmin<self.cellsize:
            raise Exception(f"Minimum radius {Rmin:.1f} is below cellsize {self.cellsize:.1f}")

        # determine mesh volume
        vol_mesh = self.cellsize**3 * nmesh_tot
        # determine non-overlapping volume of smallest void
        vol_void = (1 - self.void_overlap) * 4 * np.pi * Rmin**3 / 3
        # determine maximum possible number of voids
        max_num_voids = int(vol_void / vol_mesh)

        # define arrays containing void positions and radii
        void_pos    = np.zeros((max_num_voids, 3), dtype=np.int32)
        void_radius = np.zeros(max_num_voids,      dtype=np.float32)

        # define the in_void and delta_v array
        in_void = np.zeros(*self.nmesh, dtype=np.int8)
        delta_v = np.zeros(nmesh_tot,   dtype=np.float32)
        IDs     = np.zeros(nmesh_tot,   dtype=np.int64)

        # define the arrays needed to compute the VSF
        Nvoids = np.zeros(bins,   dtype=np.int32)
        vsf    = np.zeros(bins-1, dtype=np.float32)
        Rmean  = np.zeros(bins-1, dtype=np.float32)

        # iterate through void radii
        total_voids_found = 0
        for q in range(bins):

            R = self.radii[q]
            logger.info(f'Smoothing field with top-hat filter of radius {R:.2f}')
            delta_sm = self._smoothing(R, threads)  # change for double precision

            # check void cells are present at this radius
            if np.min(delta_sm)>self.void_delta:
                logger.info('No cells with delta < {self.void_delta:.2f}')
                continue

            IDs = self._find_underdensities(in_void, delta_v, IDs, xdim, ydim, zdim, yzdim)

            # determine void radius in terms of number of mesh cells
            R_grid = R/self.cellsize; Ncells = <int>R_grid + 1
            R_grid2 = R_grid * R_grid
            voids_found = 0 

            # select method to identify nearby voids based on radius
            mode = 0 if total_voids_found < (2*Ncells+1)**3 else 1
            threads2 = 1 if Ncells<12 else threads #empirically this seems to be the best
            logger.info(f'Identifying nearby voids using mode {mode}')

            # identify nearby voids
            for p in range(IDs.size):

                # find mesh coordinates of underdense cell
                ID = IDs[p]
                i,j,k = ID//yzdim, (ID%yzdim)//zdim, (ID%yzdim)%zdim

                # if cell belongs to a void continue
                if in_void[i,j,k] == 1: continue
