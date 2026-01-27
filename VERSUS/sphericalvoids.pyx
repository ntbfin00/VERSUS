import os
from astropy.io import fits
import numpy as np
cimport numpy as np
cimport cython
import logging
cimport void_library as VL
from .meshbuilder import DensityMesh
from .smoothing import tophat_smoothing
from scipy.spatial import cKDTree

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

    reconstruct: str, default=None
        Type of density field reconstruction passed to DensityMesh.run_recon(). Defaults to no reconstruction.

    recon_args: dict, default=None
        Reconstruction arguments ('f', 'bias', 'los', 'engine' and 'smoothing radius') passed to DensityMesh.run_recon().

    delta_mesh: array, Path, default=None
        If data_positions not provided, load density mesh directly from array or from path to pre-saved mesh FITS file. 

    mesh_args: dict, default=None
        Dictionary to hold cellsize, boxsize, boxcenter and box_like attributes. Must be provided if delta_mesh provided as array, else read from file.

    dtype: str, default='f4'
        Data type of mesh to generate. Defaults to Float32.

    boxsize: array (3), default=None
        Dimensions of simulation box (not to be used in the case of survey data).

    boxcenter: array (3), default=None
        Centre position of simulation box (not to be used in the case of survey data).

    use_wisdom: bool, default=False
        Whether to save and load wisdom during FFT computations. Advantageous for serial void-finding runs.

    kwargs : dict
        Optional arguments for meshbuilder.DensityMesh object.
    """

    cdef public object delta, data_tree, random_tree, data_weights, random_weights
    cdef public int[3] nmesh
    cdef public float[3] cellsize
    cdef public float r_sep 
    cdef public float[3] boxsize
    cdef public float[3] boxcenter
    cdef public bint box_like
    cdef public float volume
    cdef object box_shift
    cdef list data_cols
    cdef float[:] Radii
    cdef float void_delta
    cdef float void_overlap
    cdef int threads
    cdef bint use_wisdom 
    cdef public str vf_type
    cdef public object input_radii
    cdef public object position
    cdef public object radius
    cdef public object counts
    cdef public object size_function
    cdef object fft_plan, ifft_plan, fft_in, fft_out, ifft_in, ifft_out
    
    def __init__(self, data_positions=None, data_weights=None, 
                 random_positions=None, random_weights=None, data_cols=None,
                 reconstruct=None, recon_args=None, delta_mesh=None, mesh_args=None, 
                 dtype='f4', boxsize=None, boxcenter=None, init_sm_frac=0.45, use_wisdom=False, **kwargs):

        properties = ['r_sep', 'boxsize', 'boxcenter', 'box_like', 'volume', 'delta',
                      'data_positions', 'random_positions', 'data_weights', 'random_weights']

        # set whether to save and load FFT wisdom (useful for bulk runs)
        self.use_wisdom = use_wisdom

        # create mesh from positions
        if data_positions is not None:
            delta_mesh = DensityMesh(data_positions=data_positions, data_weights=data_weights,
                                     random_positions=random_positions, random_weights=random_weights, 
                                     data_cols=data_cols, reconstruct=reconstruct, recon_args=recon_args,
                                     dtype=dtype, boxsize=boxsize, boxcenter=boxcenter, init_sm_frac=init_sm_frac)
            delta_mesh.create_mesh(use_wisdom=self.use_wisdom, **kwargs)
            for name in properties:
                if name.endswith('_positions'):
                    setattr(self, name[:-9] + 'tree', getattr(delta_mesh, name))
                else:
                    setattr(self, name, getattr(delta_mesh, name))
        # load mesh
        elif delta_mesh is not None:
            # load mesh from DensityMesh object
            if isinstance(delta_mesh, DensityMesh):
                if not hasattr(delta_mesh, 'delta'):
                    delta_mesh.create_mesh(**kwargs)
                for name in properties:
                    if name.endswith('_positions'):
                        setattr(self, name[:-9] + 'tree', getattr(delta_mesh, name))
                    else:
                        setattr(self, name, getattr(delta_mesh, name))
            # load mesh from DensityMesh file
            elif type(delta_mesh) is str:
                self.load_mesh(delta_mesh)
            # load mesh from array
            else:
                if mesh_args is None or not all([arg in mesh_args for arg in properties[:5]]):
                    raise Exception(f'{properties[:5]} must be provided in addition to delta mesh with mesh_args.')
                logger.warning("data_tree (and random_tree) attributes must be provided along with data_weights (and random_weights) in order to mitigate discretness effects. Otherwise output will match Pylians")
                for name in properties:
                    if name in properties[:5]:
                        setattr(self, name, mesh_args[name])
                    elif name.endswith('_positions'):
                        setattr(self, name[:-9] + 'tree', None)
                    else:
                        setattr(self, name, None)
                self.delta = delta_mesh
        # ensure either data or mesh is provided
        else:
            raise Exception('Either data_positions or delta_mesh must be provided')

        self.nmesh = self.delta.shape
        self.cellsize = np.asarray(self.boxsize) / np.asarray(self.nmesh)
        self.box_shift = np.asarray(self.boxcenter, dtype=np.float32) - np.asarray(self.boxsize, dtype=np.float32)/2 
        logger.debug(f"Mesh data type: {self.delta.dtype}")

        if self.data_tree is not None:
            logger.info('Building k-d trees')
            self.data_tree = cKDTree((self.data_tree - self.box_shift) % self.boxsize, compact_nodes=False, 
                                     balanced_tree=False, boxsize=self.boxsize if self.box_like else None)
            self.random_tree = None if self.box_like else cKDTree((self.random_tree - self.box_shift) % self.boxsize, 
                                                                  compact_nodes=False, balanced_tree=False)

    def load_mesh(self, mesh_fn):
        r"""
        Load pre-populated 3D mesh from FITS file.

        Parameters
        ----------
        mesh_fn: string
            Path to mesh.
        """

        with fits.open(mesh_fn) as f:
            self.delta = f[0].data
            for name in ['r_sep', 'boxsize', 'boxcenter', 'box_like', 'volume', 
                         'data_positions', 'random_positions', 'data_weights', 'random_weights']:
                if name.endswith('_positions'):
                    setattr(self, name[:-9] + 'tree', f[name].data)
                else:
                    setattr(self, name, f[name].data)

    def rmin_spurious(self, sign):
        r"""
        Determine the detection limit for spurious voids for the given tracer sample using an empirical formula. At smaller radii, spurious voids may contaminate the output void sample.

        """

        if sign == 1:
            fact = 2.2
        else:
            fact = 1.6

        rho_mean = 3 / (4 * np.pi * self.r_sep**3)
        return (fact * sign * self.void_delta + 3.6) / rho_mean**(1/3)

    def _smoothing(self, float radius):
        r"""
        Smooth density field with a tophat filter at a given radius.

        """

        (
          delta_sm,
          self.fft_plan,
          self.fft_in,
          self.fft_out,
          self.ifft_plan,
          self.ifft_in,
          self.ifft_out
        ) = tophat_smoothing(self.delta,
                             radius,
                             self.cellsize,
                             threads=self.threads,
                             use_wisdom=self.use_wisdom,
                             fft_plan=self.fft_plan,
                             ifft_plan=self.ifft_plan,
                             fft_in=self.fft_in,
                             fft_out=self.fft_out,
                             ifft_in=self.ifft_in,
                             ifft_out=self.ifft_out)

        # reset survey mask
        if not self.box_like:
            logger.debug("Resetting survey mask")
            delta_sm[self.delta == 0.0] = 0.0

        return delta_sm


    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    def resize_voids(self, float sign):
        """
        Resize voids according to new interior densities calculated directly from the galaxy and random positions.
        """
        cdef int p, q, N_tot
        cdef float fact#, delta_enc
        cdef int[::1] Nvoids 

        cdef int[:,::1] new_position 
        cdef float[::1] new_radius 
        cdef long voids_found=0, total_voids_found=0

        logger.info("Postprocessing catalogues directly using galaxy positions.")

        # ensure correct expression for clusters
        void_delta = sign * self.void_delta

        # counts in smallest bin are underestimated due to missed upscattering from smaller radii
        self.input_radii = self.input_radii[:self.input_radii.size-1]

        # compute factor for density calculation
        data_w = np.ones(self.data_tree.n) if self.data_weights is None else self.data_weights
        if self.box_like: 
            inv_rho_mean = 3 * self.volume / (4 * np.pi * data_w.sum())
        else:
            rand_w = np.ones(self.random_tree.n) if self.random_weights is None else self.random_weights
            inv_rho_mean = rand_w.sum() / data_w.sum()

        # create grid of potential new positions around void center
        dx = self.r_sep / 2
        # offsets = np.arange(-dx, 2*dx, dx)
        offsets = np.arange(-dx, dx, dx) + dx / 2
        # offsets = np.array([0.])
        grid = np.stack(np.meshgrid(offsets, offsets, offsets, indexing="ij"), -1).reshape(-1, 3)
        pos_grid = self.position[:, None, :] + grid[None, :, :]

        # determine number of enclosed galaxies
        Rmax = self.input_radii[0]
        n_max = int(1.5 * (Rmax / self.r_sep)**3)
        particle_dist, particle_id = self.data_tree.query(
            pos_grid,
            k=n_max,
            distance_upper_bound=Rmax,
            workers=self.threads,
        )
        del self.data_tree

        new_position = np.zeros_like(self.position, dtype=np.int32) #### CHANGE
        new_radius = np.zeros_like(self.radius)
        Nvoids = np.zeros(self.input_radii.size, dtype=np.int32)

        # mask detected void cells
        available = np.ones(pos_grid.shape[0], dtype=bool)
        # available = np.ones(pos_grid.shape[:2], dtype=bool)

        # loop over input radii
        for q, R in enumerate(self.input_radii):
            mask = particle_dist < R

            weights = np.zeros_like(particle_dist, dtype=data_w.dtype)
            weights[mask] = data_w[particle_id[mask]]
            delta_enc = (weights * mask).sum(axis=2)
            sort = np.argsort(delta_enc, axis=1)
            delta_enc = np.take_along_axis(delta_enc, sort, axis=1)
            pos_grid_sorted = np.take_along_axis(pos_grid, sort[..., None], axis=1)

            if self.box_like: 
                delta_enc /= R**3
            # else:
                # rand_count = rand_w[self.random_tree.query_ball_point(pos, R, workers=self.threads)].sum() #### change
                # if rand_count == 0: break
                # delta_enc /= rand_count
            delta_enc *= inv_rho_mean 
            delta_enc -= 1

            # determine which voids pass density threshold at R
            # new_voids = sign * delta_enc.min(axis=1) < void_delta
            new_voids = sign * delta_enc[:,0] < void_delta
            new_voids &= available
            delta_min = delta_enc[new_voids].min(axis=1)
            sort = np.argsort(delta_min)
            indx = delta_enc[new_voids].argmin(axis=1)[sort]
            void_id = np.flatnonzero(new_voids)[sort]

            num_voids_around1 = VL.num_voids_around1_wrap #### CHANGE
            voids_found = 0
            # loop over voids that pass threshold cut (sorted by delta)
            # for (n, ID) in enumerate(void_id):
                # cand_pos = pos_grid[ID, indx[n]]
            for ID in void_id:
                for n in range(pos_grid.shape[1]):
                    if delta_enc[ID, n] > void_delta: break  # can make this explicit in new_voids?
                    cand_pos = pos_grid_sorted[ID, n]
                    nearby_voids = num_voids_around1(self.void_overlap, total_voids_found,
                                                     int(self.boxsize[0]), int(self.boxsize[1]), int(self.boxsize[2]), 
                                                     int(cand_pos[0]), int(cand_pos[1]), int(cand_pos[2]),
                                                     &new_radius[0], &new_position[0,0],
                                                     R, 4)

                    if nearby_voids == 0:
                        new_position[total_voids_found, 0] = cand_pos[0]
                        new_position[total_voids_found, 1] = cand_pos[1]
                        new_position[total_voids_found, 2] = cand_pos[2]
                        new_radius[total_voids_found] = R
                        voids_found += 1
                        total_voids_found += 1

                        # mark potential void positions as occupied
                        available[ID] = False
                        break 

            Nvoids[q] = voids_found

        self.position = new_position[:total_voids_found]
        self.radius   = new_radius[:total_voids_found]
        self.counts   = np.asarray(Nvoids)


    def _sort_radii(self, float[:] radii):
        return np.sort(radii)[::-1]


    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    def run_voidfinding(self, radii=[0.], float void_delta=-0.8, void_overlap=False, int threads=8):
        r"""
        Run spherical voidfinding on density mesh.

        Parameters
        ----------

        radii: list 
            List of void radii to search for. Defaults to 4-104x cellsize.

        void_delta: float, default=-0.8
            Maximum overdensity threshold to be classified as void. If value is positive, peaks will be found instead.

        void_overlap: float, default=False
            Maximum allowed volume fraction of void overlap. If False, no overlap is allowed.

        threads: int, default=8
            Number of threads used for multi-threaded processes. If set to zero, defaults to number of available CPUs.

        """
        cdef np.ndarray[np.float32_t, ndim=1] Radii=np.array(radii, dtype=np.float32)
        cdef float R, R_grid, R_grid2, Rmin, Rspurious
        cdef int bins, Ncells, nearby_voids, threads2 
        cdef long nmesh_tot=np.prod(self.nmesh)
        cdef long max_num_voids, voids_found, total_voids_found, ID
        cdef float vol_mesh, vol_void, norm
        cdef float[:,:,::1] delta_sm
        cdef long[:,:,::1] in_void
        cdef long[::1] IDs
        cdef int i, j, k, p, q, xdim, ydim, zdim, yzdim, mode
        cdef int[::1] Nvoids
        cdef double void_cell_fraction=0.0, void_volume_fraction=0.0
        cdef float[:,::1] position
        cdef int[:,::1] void_pos
        cdef float[::1] delta_v, void_rad, box_shift
        cdef float[:,::1] vsf
        cdef long local_voids
        cdef long[::1] indexes, IDs_temp

        # set maximum density threshold for cell to be classified as void
        self.void_delta = void_delta
        cellsize = np.mean(self.cellsize)

        # find peaks
        if void_delta>0: 
            sign = -1.
            void_delta *= -1
            self.vf_type, ineq = ('peak', '>')
        # find voids
        else:
            sign = 1.
            self.vf_type, ineq = ('void', '<')

        Rspurious = self.rmin_spurious(sign)
        # set default radii if not provided
        if radii[0] == 0.:
            Radii = np.arange(20, 62, 2, dtype=np.float32)[::-1]
            self.Radii = Radii[(Radii > cellsize) & (Radii > Rspurious)]  # ensure radii larger than cellsize and detection limit of spurious voids
            logger.debug(f'Radii set by default: cellsize={cellsize:.2f}, Rmin_spurious={Rspurious:.2f}.')
        else:
            # ensure extra bin for void resizing
            if self.data_tree is not None:
                Radii = np.append(Radii, Radii.min() - 2).astype(np.float32)
            # order input radii from largest to smallest
            self.Radii = self._sort_radii(Radii)
            logger.debug(f'Radii set manually')

        bins = self.Radii.size
        Rmin = np.min(self.Radii)
        if Rmin < Rspurious: logger.warning(f"Spurious {self.vf_type}s may enter sample (for Rmin < {Rspurious:.0f} Mpc/h)")

        # set allowed void overlap for void classification
        if type(void_overlap) is bool: 
            self.void_overlap = 0.
        else:
            self.void_overlap = void_overlap
            void_overlap = False
        logger.debug(f"Overlap set to {void_overlap} (value={self.void_overlap})")
        # set dimensions
        xdim, ydim, zdim = self.nmesh
        yzdim = ydim * zdim
        # set threads
        self.threads = os.cpu_count() if threads==0 else threads
        logger.info(f'Running spherical {self.vf_type}-finder with {self.threads} threads (delta {ineq} {self.void_delta:.2f})')

        # check that radii are compatible with grid resolution
        if Rmin < cellsize:
            raise Exception(f"Minimum radius {Rmin:.1f} is below cellsize {cellsize:.2f}")
        # if (abs(np.diff(self.Radii))<self.cellsize).any():
            # logger.warning(f"Radii are binned more finely than cellsize {self.cellsize:.1f}. May induce bin-to-bin correlations.")

        # determine mesh volume
        vol_mesh = np.prod(self.cellsize) * nmesh_tot
        # determine non-overlapping volume of smallest void
        vol_void = (1 - self.void_overlap) * 4 * np.pi * Rmin**3 / 3
        # determine maximum possible number of voids
        max_num_voids = int(vol_mesh / vol_void)
        logger.debug(f"Total mesh cells = {nmesh_tot:d} ({xdim},{ydim},{zdim})")
        logger.debug(f"Maximum number of voids = {max_num_voids:d}")

        # define arrays containing void positions and radii
        void_pos    = np.zeros((max_num_voids, 3), dtype=np.int32)
        void_rad    = np.zeros(max_num_voids,      dtype=np.float32)
        Nvoids = np.zeros(bins,   dtype=np.int32)

        # define the in_void and delta_v array
        in_void = np.zeros(self.nmesh, dtype=np.int64)
        delta_v = np.zeros(nmesh_tot,   dtype=np.float32)
        IDs     = np.zeros(nmesh_tot,   dtype=np.int64)

        # set function wrapping based on box-like
        if self.box_like:
            logger.debug("Using wrapped VF algorithms")
            num_voids_around1 = VL.num_voids_around1_wrap
            num_voids_around2 = VL.num_voids_around2_wrap
            mark_void_region  = VL.mark_void_region_wrap
        else:
            logger.debug("Using VF algorithms without wrapping")
            num_voids_around1 = VL.num_voids_around1
            num_voids_around2 = VL.num_voids_around2
            mark_void_region  = VL.mark_void_region

        # iterate through void radii
        total_voids_found = 0
        for q in range(bins):

            R = self.Radii[q]
            logger.debug(f'Smoothing field with top-hat filter of radius {R:.1f} Mpc/h')

            delta_sm = sign * self._smoothing(R)

            # check void cells are present at this radius
            if np.min(delta_sm)>void_delta:
                logger.info(f'No cells with delta {ineq} {self.void_delta:.2f} for R={R:.1f} Mpc/h')
                continue

            logger.debug(f'Looping through {delta_sm.size:d} cells to find underdensities and asineqing IDs')
            local_voids = 0
            for i in range(xdim):
                for j in range(ydim):
                    for k in range(zdim):

                        if delta_sm[i,j,k]<void_delta and in_void[i,j,k]==0:
                            IDs[local_voids]     = yzdim*i + zdim*j + k
                            delta_v[local_voids] = delta_sm[i,j,k]
                            local_voids += 1
            logger.debug(f'Found {local_voids} cells with delta {ineq} {self.void_delta:.2f}')

            # sort delta_v by density
            indexes = np.argsort(delta_v[:local_voids])

            # sort IDs by density
            IDs_temp = np.empty(local_voids, dtype=np.int64)
            for i in range(local_voids):
                IDs_temp[i] = IDs[indexes[i]]
            for i in range(local_voids):
                IDs[i] = IDs_temp[i]
            del IDs_temp
            logger.debug('Sorting of underdense cells finished.')

            # determine void radius in terms of number of mesh cells
            R_grid = R / cellsize; Ncells = <int>R_grid + 1
            R_grid2 = R_grid * R_grid
            voids_found = 0 

            # select method to identify nearby voids based on radius
            mode = 0 if total_voids_found < (2*Ncells+1)**3 else 1
            threads2 = 1 if Ncells<12 else min(4, self.threads)  # empirically this seems to be the best
            logger.debug(f'Setting threads2 = {threads2} (threads={self.threads})')
            if not void_overlap: logger.debug(f'Identifying nearby voids using mode {mode}')

            # identify nearby voids
            for p in range(local_voids):

                # find mesh coordinates of underdense cell
                ID = IDs[p]
                i,j,k = ID//yzdim, (ID%yzdim)//zdim, (ID%yzdim)%zdim

                # if central cell belongs to a void continue (unless using fractional overlap)
                if (self.void_overlap == 0.) and (in_void[i,j,k] > 0): continue
                
                nearby_voids = 0
                # determine amount of overlap to detect nearby voids
                if not void_overlap:
                    if mode==0:
                        # detect nearby voids using distances between centres
                        nearby_voids = num_voids_around1(self.void_overlap, total_voids_found, 
                                                         xdim, ydim, zdim, i, j, k, 
                                                         &void_rad[0], &void_pos[0,0], 
                                                         R_grid, threads2)
                    else:
                        # detect nearby voids using cell searching
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

                    mark_void_region(&in_void[0,0,0], Ncells, xdim, ydim, zdim,
                                     yzdim, R_grid2, i, j, k, threads=1)

            logger.info(f'Found {voids_found} {self.vf_type}s with radius R={R:.1f} Mpc/h')
            Nvoids[q] = voids_found 

            void_cell_fraction = np.sum(in_void, dtype=np.int64) * 1.0/nmesh_tot  # volume determined using filled cells
            void_volume_fraction += voids_found * 4.0 * np.pi * R**3 / (3.0 * vol_mesh) # volume determined using void radii
            logger.debug('Occupied void volume fraction = {:.3f} (expected {:.3f})'.format(void_cell_fraction, void_volume_fraction))

        logger.info(f'{total_voids_found} total {self.vf_type}s found.')
        logger.info(f'Occupied {self.vf_type} volume fraction = {void_cell_fraction:.3f} (expected {void_volume_fraction:.3f})')

        # finish by setting the class fields
        self.input_radii = np.asarray(self.Radii)
        pos              = np.asarray(void_pos[:total_voids_found], dtype=np.float32)  # void positions on mesh
        self.position    = pos * np.asarray(self.cellsize, dtype=np.float32)  # transform positions relative to data 
        self.radius      = np.asarray(void_rad[:total_voids_found]) * cellsize
        self.counts      = np.asarray(Nvoids)

        # post-process voids by counting enclosed galaxies (and randoms)
        if self.data_tree is None:
            logger.warning("self.data_tree (scipy.spatial.cKDTree object of positions) has not been provided to determine void sizes directly from galaxy positions. Output may be subject to discreteness effects.")
        else:
            self.resize_voids(sign)

        self.position += self.box_shift

        # compute the void size function (dn/dlnR = # of voids/Volume/delta(lnR))
        vsf = np.zeros((3, self.input_radii.size - 1), dtype=np.float32)
        for i in range(self.input_radii.size - 1):
            norm = 1 / (self.volume * np.log(self.input_radii[i] / self.input_radii[i+1]))
            vsf[0,i] = np.sqrt(self.input_radii[i] * self.input_radii[i+1])  # geometric mean radius for logarithmic scale
            vsf[1,i] = self.counts[i+1] * norm  # vsf (voids >R[i] will be detected with smoothing of R[i])
            vsf[2,i] = np.sqrt(self.counts[i+1]) * norm  # poisson uncertainty

        self.size_function = np.asarray(vsf) 




    def plot_size_function(self, dndlnr=True, poisson_err=True, log=False,
                           legend=False, grid=False, ax=None, save_fn=None, **kwargs):
        r"""
        Plot the void size function.

        Parameters
        ----------

        dndlnr: bool, default=True
            Plot number density of voids per logarithmic radius interval. If False, plot number density of voids n(R).

        poisson_err: bool, default=True
            Plot errors estimated assuming Poisson distributed void number counts. True errors should be computed from mock simulations.

        log: bool, default=False
            Plot size abundance on logarithmic scale.

        legend: bool, default=False
            Plot legend.

        grid: bool, default=False
            Plot grid.

        ax: matplotlib.axes, default=None
            Optional axes for figure.

        save_fn: str, default=None
            Path to save figure.

        kwargs:
            Optional arguments for matplotlib.pyplot.errorbar().
        """
        import matplotlib.pyplot as plt

        if self.input_radii is None:
            raise Exception("Must run SphericalVoids.run_voidfinder() first in order to use this function.")

        if ax is None: fig, ax = plt.subplots(1)

        if dndlnr:
            ylabel = rf'$d n_{{\rm {self.vf_type}}} / d \ln R_{{\rm {self.vf_type}}} \,\,[h^4{{\rm Mpc}}^{{-4}}]$'
            x, y, err = self.size_function
        else:
            ylabel = rf'$n_{{\rm {self.vf_type}}}\,\,[h^3{{\rm Mpc}}^{{-3}}]$'
            y = self.counts.astype('float32')
            x, err = self.input_radii, np.sqrt(y) / self.volume
            y /= self.volume

        ax.errorbar(x, y, err if poisson_err else None, **kwargs)
        ax.set_ylabel(ylabel, fontsize=15)
        ax.set_xlabel(r'$R_{\rm void}\, [h^{-1}{\rm Mpc}]$', fontsize=15)
        if log: ax.set_yscale('log')
        if grid: ax.grid()
        if legend: ax.legend()

        if save_fn is not None: plt.savefig(save_fn)

        return ax


    def plot_slice(self, slice_axis='Z', slice_range=(30,80), data_positions=None, 
                   legend=False, grid=False, ax=None, save_fn=None, **kwargs):
        r"""
        Plot the void size function.

        Parameters
        ----------

        slice_axis: str, default='Z'
            Axis of mesh to slice along.

        slice_range: tuple, default=(30, 80)
            Lower and upper limits of slice_axis to slice.

        data_positions: array, default=None
            Array of data positions to plot. If None, instead plot the overdensity mesh.

        legend: bool, default=False
            Plot legend.

        grid: bool, default=False
            Plot grid.

        ax: matplotlib.axes, default=None
            Optional axes for figure. If not None, only the voids/peaks are plotted (no galaxies/delta mesh).

        save_fn: str, default=None
            Path to save figure.

        kwargs:
            Optional arguments for matplotlib.patches.Circle().
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        if self.input_radii is None:
            raise Exception("Must run SphericalVoids.run_voidfinder() first in order to use this function.")

        axes_labels = ['X','Y','Z']
        axis = axes_labels.index(slice_axis.upper())
        axes_labels.pop(axis)
        axes = [0, 1, 2]
        axes.pop(axis)
        boxlims = np.c_[self.box_shift, self.boxsize + self.box_shift]

        # filter spheres
        mask_sp = (self.position[:,axis] >= slice_range[0]) & (self.position[:,axis] <= slice_range[1])
        centers = self.position[mask_sp]
        radii = self.radius[mask_sp]

        if ax is None: 
            fig, ax = plt.subplots(1, figsize=(8,8))
            # filter density grid
            if data_positions is None:
                lim = (slice_range - self.box_shift[axis]) / self.cellsize[axis]
                xy_slice = self.delta.take(indices=range(int(lim[0]), int(lim[1])), axis=axis)
                xy_slice = xy_slice.mean(axis=axis)
                extent = [*boxlims[axes[0]], *boxlims[axes[1]]]
                ax.imshow(xy_slice.T, origin='lower', extent=extent, cmap='plasma')
            # filter galaxies 
            else:
                mask_xy = (data_positions[:,axis] >= slice_range[0]) & (data_positions[:,axis] <= slice_range[1])
                xy_slice = data_positions[mask_xy]
                ax.scatter(xy_slice[:,axes[0]], xy_slice[:,axes[1]], s=0.1, color='black')

        # create plot
        label = kwargs.pop('label', None)
        color = kwargs.pop('color', 'red')
        for (i, (c, r)) in enumerate(zip(centers, radii)):
            circ = Circle((c[axes[0]], c[axes[1]]), r, fill=False, edgecolor=color, 
                          label=None if i>0 else label, **kwargs)
            ax.add_patch(circ)
        ax.set_xlabel(axes_labels[0] + r' $[h^{-1}{\rm Mpc}]$', fontsize=15)
        ax.set_ylabel(axes_labels[1] + r' $[h^{-1}{\rm Mpc}]$', fontsize=15)
        ax.set_xlim(boxlims[axes[0]])
        ax.set_ylim(boxlims[axes[1]])
        ax.set_title((rf'{slice_range[0]} $[h^{{-1}}{{\rm Mpc}}]$' 
                    + rf'$\leq {slice_axis.upper()} \leq$' 
                    + rf'{slice_range[1]} $[h^{{-1}}{{\rm Mpc}}]$'),
                    fontsize=15)
        ax.set_aspect('equal')

        if legend: ax.legend(loc='upper right')

        if save_fn is not None: plt.savefig(save_fn)

        return ax

