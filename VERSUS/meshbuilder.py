import numpy as np
import os
from astropy.io import fits
import logging
from scipy.spatial import cKDTree
from .smoothing import tophat_smoothing


logger = logging.getLogger(__name__)

# @cython.cclass
class DensityMesh:
    r"""                                                                                                                                          
    Generate density mesh from galaxy and random positions. 

    Parameters
    ----------
    data_positions: array (N), string
        Array of data positions (in cartesian or sky coordinates) or path to such positions.

    data_weights: array (N,3), default=None
        Array of data weights.

    random_positions: array (N,3), string
        Array of random positions (in cartesian or sky coordinates) or path to such positions.

    random_weights: array (N), default=None
        Array of random weights.

    data_cols: list, default=None
        List of data/random position column headers. Fourth element is taken as the weights (if present). Defaults to ['RA','DEC','Z'] if randoms are provided and ['X','Y','Z'] if not.

    dtype: string, default='f4'
        Mesh data type.

    threads: int, default=None
        Number of threads used for multi-threaded processes. If None, defaults to number of available CPUs. 

    reconstruct: string
        Type of reconstruction to run - 'disp', 'rsd' or 'disp+rsd'. Defaults to no reconstruction. Must additionally provide 'f' and 'bias' in recon_args.

    recon_args: dict
        Reconstruction arguments - 'f', 'bias', 'los' (only required for box), 'engine' and 'smoothing radius'.

    kwargs : dict
        Optional arguments.
    """

    def __init__(self, data_positions, data_weights=None, random_positions=None, random_weights=None, 
                 data_cols=None, dtype='f4', threads=None, reconstruct=None, recon_args=None, 
                 boxsize=None, boxcenter=None, **kwargs):

        # if randoms are supplied then treat as survey
        self.box_like = True if random_positions is None else False
        logger.info('Loading {}-like data'.format('box' if self.box_like else 'survey'))
        self.dtype = dtype
        self.threads = os.cpu_count() if threads is None else 8
        logger.debug(f'Data type: {self.dtype}, Threads: {self.threads}')
        # set reconstruction parameters
        self.reconstruct = reconstruct
        self.recon_args = {} if recon_args is None else recon_args
        # default options to read data column headers
        if data_cols is None:
            data_cols = ['X','Y','Z'] if self.box_like else ['RA','DEC','Z']
        # load positions from file
        if type(data_positions) is str:
            self.data_positions, self.data_weights = self._load_data(data_positions, data_cols)
        # load positions from array
        else:
            self.data_positions = data_positions
            self.data_weights = data_weights
        # automatically determine boxsize from positions
        self.boxsize = np.ceil(np.abs(self.data_positions.max(axis=0) 
                       - self.data_positions.min(axis=0))) if boxsize is None else boxsize
        self.boxcenter = (self.data_positions.max(axis=0) 
                         + self.data_positions.min(axis=0)) / 2 if boxcenter is None else boxcenter
        self.N_data = len(self.data_positions)  # total galaxies 
        self.W_data = self.N_data if data_weights is None else np.array(data_weights).sum()  # sum of galaxy weights
        # load randoms from file
        if type(random_positions) is str:
            self.random_positions, self.random_weights = self._load_data(random_positions, data_cols)
        else:
            self.random_positions = random_positions
            self.random_weights = random_weights
        self.N_random = None if random_positions is None else len(self.random_positions)  # total randoms
        self.W_random = self.N_random if random_weights is None else np.array(random_weights).sum()  # sum of random weights

    def _load_data(self, data_fn, data_cols, z_to_dist=None, **kwargs):
        r"""
        Load galaxy or random positions from FITS file

        Parameters
        ----------
        data_fn: string
            Path to data.

        data_cols: list
            Positions (cartesian 'xyz' or sky 'rdz') column headers to read. Fourth column (if included) should correspond to data weights.

        z_to_dist: callable
            Callable that provides distance as a function of redshift.

        kwargs:
            Additional arguments for pyrecon.sky_to_cartesian
        """

        weights = None

        if data_fn.endswith('.fits'):
            f = fits.open(data_fn)
            N = f[1].header['NAXIS2']
            positions = np.zeros((N, 3))
            logger.info(f'Loading positions from FITS file with column headers {data_cols}.')
            for (i,c) in enumerate(data_cols):
                # read weights if 4th column provided
                if i<3: 
                    positions[:,i] = f[1].data[c]
                else:
                    weights = f[1].data[c]
            f.close()
        elif data_fn.endswith('.npy'):
            logger.info(f'Loading positions from npy file.')
            positions = np.load(data_fn)
            # read weights if 4th column provided
            if positions.shape[1]>3: 
                weights = positions[:,3]
                positions = positions[:,:3]
        else:
            raise Exception("File type not recognised. Please provide a file ending in .fits or .npy.")

        # if x or y contained in data_cols assume positions are provided in cartesian coordinates
        # else convert sky positions to cartesian
        if not np.array([a in ''.join(data_cols).upper() for a in ('X','Y')]).all():
            logger.info('Converting sky positions to cartesian.')
            from pyrecon.utils import sky_to_cartesian
            # use LambdaCDM as default redshift to distance conversion
            if z_to_dist is None:
                from astropy.cosmology import LambdaCDM
                cosmo = LambdaCDM(H0=67.6, Om0=0.31, Ode0=0.69)
                positions[:,2] = cosmo.comoving_distance(positions[:,2])
            else:
                positions[:,2] = z_to_dist(positions[:,2])
            # convert sky positions
            positions = sky_to_cartesian(positions[:,2], positions[:,0], positions[:,1], **kwargs)

        return positions, weights


    def _set_mesh(self, engine='IterativeFFTParticleReconstruction', cellsize=4., boxpad=1.1, **kwargs):
        r"""
        Set the mesh properties and type of reconstruction algorithm

        Parameters
        ----------
        engine: string
            Reconstruction algorithm passed to pyrecon.

        cellsize: float, default=4.
            Size of mesh cell.

        boxpad: float, default=1.1
            Padding applied to mesh.
        """
        self.engine = engine

        # use galaxies (or randoms for survey) to estimate boxsize if not supplied
        if self.box_like:
            boxsize = self.boxsize
            boxcenter = self.boxcenter 
            nmesh = boxsize // cellsize // 2 * 2  # ensure boxsize is divisible by even number of cells
            cellsize = positions = None
            wrap = True
            boxpad = 1.
        else:
            boxsize = boxcenter = nmesh = None
            positions = self.random_positions
            wrap = False

        # select reconstruction algorithm
        exec(f"from pyrecon import {engine}; Recon = {engine}", globals())

        # initialise mesh
        mesh = Recon(positions=positions, cellsize=cellsize, nmesh=nmesh, 
                     boxsize=boxsize, boxcenter=boxcenter, boxpad=boxpad, 
                     wrap=wrap, dtype=self.dtype, nthreads=self.threads, **kwargs)
        logger.debug(f"Boxsize: {mesh.boxsize}, Boxcenter: {mesh.boxcenter}, Cellsize: {mesh.cellsize}")

        return mesh

    def _set_mesh_density(self, mesh, init_sm_frac=0.45, threshold_randoms=0.01):
        r"""
        Populate mesh with galaxies and randoms

        Parameters
        ----------
        mesh: pyrecon.recon.BaseReconstruction object
            Input density mesh.
            
        init_sm_frac: float, default=0.45
            Inititial spherical smoothing for galaxies and randoms on mesh.
        """

        # assign data
        mesh.assign_data(self.data_positions, self.data_weights)
        # assign randoms
        if not self.box_like: mesh.assign_randoms(self.random_positions, self.random_weights)

        r_smooth = init_sm_frac * self.r_sep
        logger.info("Applying initial smoothing of R={:.1f} Mpc/h to the density field".format(r_smooth))
        mesh.mesh_delta = mesh.mesh_data.copy()
        mesh.mesh_delta.value = tophat_smoothing(mesh.mesh_data, 
                                                 r_smooth,
                                                 mesh.cellsize[0],
                                                 threads=self.threads)[0]
        del mesh.mesh_data

        if self.box_like:
            mesh.rho_mean = mesh.mesh_delta.cmean() / mesh.cellsize.prod()
            mesh.mesh_delta /= mesh.mesh_delta.cmean()
            mesh.mesh_delta -= 1
        else:
            # if check:
                # nnonzero = self.mpicomm.allreduce(sum(np.sum(randoms > 0.) for randoms in self.mesh_randoms))
                # if nnonzero < 2: raise ValueError('Very few randoms!')

            mesh.mesh_randoms.value = tophat_smoothing(mesh.mesh_randoms, 
                                                       r_smooth,
                                                       mesh.cellsize[0],
                                                       threads=self.threads)[0]

            sum_data, sum_randoms = mesh.mesh_delta.csum(), mesh.mesh_randoms.csum()
            alpha = sum_data * 1. / sum_randoms

            for delta, randoms in zip(mesh.mesh_delta.slabs, mesh.mesh_randoms.slabs):
                delta[...] -= alpha * randoms

            threshold = threshold_randoms * sum_randoms / mesh._size_randoms
            # threshold = threshold_randoms * mesh._sumw2_randoms / sum_randoms  # test pyrecon 'noise' method

            mesh.rho_mean = np.zeros(mesh.mesh_delta.slabs.nslabs)
            for (i, (delta, randoms)) in enumerate(zip(mesh.mesh_delta.slabs, mesh.mesh_randoms.slabs)):
                mask = randoms > threshold
                delta[mask] /= (alpha * randoms)[mask]
                delta[~mask] = 0.

                mesh.rho_mean[i] = np.mean((alpha * randoms)[mask])

            del mesh.mesh_randoms
            mesh.rho_mean = np.nanmean(mesh.rho_mean) / mesh.cellsize.prod()

        print('rho_mean:', mesh.rho_mean, mesh.boxsize, mesh.nmesh)#, 'threshold', threshold)


    def run_recon(self, f=None, bias=None, los=None, engine='IterativeFFTReconstruction', 
                  boxpad=1.1, smoothing_radius=10., field='rsd', **kwargs):
        r"""
        Perform reconstruction on galaxy positions using pyrecon (https://github.com/cosmodesi/pyrecon.git)
        """

        if f is None or bias is None:
            raise Exception("Minimally 'f' and 'bias' must be provided for density field reconstruction")

        if self.box_like and los is None:
            raise Exception("'los' must be provided for density field reconstruction of box-like data")
        if not self.box_like: los = None  # survey has local line-of-sight


        logger.info(f"Running '{field}' reconstruction with {engine}")
        logger.info(f"Recon parameters: f={f:.2f}, b={bias:.2f}, los={los}, r_smooth={smoothing_radius}")
        # set and smooth mesh
        self.data_mesh = self._set_mesh(f=f, bias=bias, engine=engine, cellsize=self.cellsize, los=los,
                                        boxpad=boxpad, fft_engine='fftw', fft_plan='estimate', **kwargs)
        self.data_mesh.assign_data(self.data_positions, self.data_weights)
        if not self.box_like: self.data_mesh.assign_randoms(self.random_positions, self.random_weights)
        self.data_mesh.set_density_contrast(smoothing_radius=smoothing_radius)

        # run reconstruction
        logger.debug(f"Reconstruction running on {self.data_mesh.nmesh} mesh.")
        self.data_mesh.run()
        logger.debug("Reconstruction complete. Setting new positions.")
        # read reconstructed positions
        self.data_positions = self.data_mesh.read_shifted_positions(self.data_positions, field=field)
        if self.random_positions is not None and 'disp' in field: 
            self.random_positions = self.data_mesh.read_shifted_positions(self.random_positions, field='disp')  # RecIso
        del self.data_mesh

        
    def size_mesh(self, niterations=4, cellsize=4., **kwargs):
        r"""
        Estimate the survey volume, mean density and average galaxy separation.

        Parameters
        ----------
        niterations : int, default=4
            Number of iterations used to estimate the average galaxy separation.

        cellsize: float, default=4.
            Cellsize used to estimate the average galaxy separation.

        ran_min : float, default=0.01
            Minimum fraction of average randoms for cell to be counted as part of survey.

        kwargs : dict
            Optional arguments for pyrecon.set_density_contrast().
        """

        logger.info(f'Estimating volume and average galaxy separation (Ngal = {self.N_data})')

        # suppress logging outputs
        # logger = logging.getLogger(__name__)
        # old_level = logger.level
        # logger.setLevel(logging.DEBUG)

        # create mesh flush with survey volume
        mesh = self._set_mesh(boxpad=1., cellsize=cellsize)
        # first estimate of volume (true if box)
        self.volume = mesh.boxsize.prod()
        # first estimate of mean density
        self.rho_mean = self.W_data / self.volume
        # first estimate of average separation between galaxies
        self.r_sep = (4 * np.pi * self.rho_mean / 3)**(-1/3)  
        self.cellsize = mesh.cellsize[0]

        # iterate for better estimate of survey volume
        if not self.box_like:
            for i in range(niterations):
                logger.debug(f'Iteration {i} of volume estimation: r_sep is {self.r_sep}Mpc')
                # set cellsize to half estimated galaxy separation
                mesh = self._set_mesh(boxpad=1.0, cellsize=self.r_sep / 2, bias=1.)
                # determine survey boundary (cells outside have delta=0.)
                self._set_mesh_density(mesh)
                survey_mask = mesh.mesh_delta.value != 0.
                print('ZEROS', survey_mask.sum(), survey_mask.sum() / mesh.nmesh.prod())
                # estimate survey volume
                self.volume = survey_mask.sum() * mesh.cellsize.prod()
                # estimate average galaxy separation
                self.r_sep = (4 * np.pi * mesh.rho_mean / 3)**(-1/3)
                print(mesh.rho_mean, self.r_sep, mesh.nmesh, 'Volume', self.volume)
            self.rho_mean = mesh.rho_mean
        logger.debug(f"Volume: {self.volume:.0f}, Density: {self.rho_mean:.4f}")

        
    def create_mesh(self, boxpad=1.1, cellsize=4., init_sm_frac=0.45, save_mesh=None, **kwargs):
        r"""
        Create the density mesh

        Parameters
        ----------
        boxpad : float, default=1.1
            Padding factor for survey mesh. Simulation box has no padding.

        cellsize: float, default=4.
            Size of mesh cells. 

        init_sm_frac: float, default=0.45
            Inititial spherical smoothing for galaxies and randoms on mesh.

        ran_min : float, default=0.01
            Minimum fraction of average randoms for cell to be counted as part of survey.

        save_mesh : bool, string, default=None
            If not ``None``, path where to save the mesh in FITS format.
            If ``True``, the mesh will be saved in the default path: f'mesh/mesh_<nbins_vf>_<dtype>.fits'.

        kwargs : dict
            Optional arguments.
        """

        # estimate cellsize based on galaxy density
        if not hasattr(self, 'cellsize'): self.size_mesh(cellsize=cellsize)
        if (self.rho_mean * self.cellsize**3) > 1: 
            logger.warning("Cellsize exceeds one galaxy per cell. Mesh is no longer shot-noise dominated.")

        # run optional reconstruction
        if self.reconstruct is not None: self.run_recon(field=self.reconstruct, boxpad=boxpad, **self.recon_args)

        # generate mesh
        mesh = self._set_mesh(cellsize=self.cellsize,
                              engine='IterativeFFTParticleReconstruction', # faster when smoothing is not required
                              boxpad=boxpad, # pad if survey to better detect boundary voids
                              bias=1.,  # bias set to 1. so voids are found on galaxy (not matter) field
                              resampler='ngp')  # Do not perform 'smoothing' during particle assignment to mesh

        logger.info(f'Estimating mesh density (nmesh={mesh.nmesh})')
        logger.debug(f'Mass-Assignment-Scheme: {mesh.resampler.kind}')
        print('before final', self.rho_mean)
        self._set_mesh_density(mesh)
        print('after final', self.rho_mean)

        self.delta = mesh.mesh_delta.value
        self.nmesh = mesh.nmesh
        self.boxsize = mesh.boxsize
        self.boxcenter = mesh.boxcenter

        # save mesh to FITS file
        if save_mesh:
            # save mesh in default path
            if type(save_mesh) is bool:
                if not os.path.isdir('mesh'): os.mkdir('mesh')  # create /mesh directory
                axes = ['nx','ny','nz']
                nbins = '_'.join(axes[i] + str(n) for (i,n) in enumerate(mesh.nmesh))
                save_mesh = os.path.join('mesh', f'mesh_{nbins}_{self.dtype}')
            # save
            delta_hdu = fits.PrimaryHDU(self.delta)
            data_pos_hdu = fits.ImageHDU(data=self.data_positions, name='data_positions')
            data_weight_hdu = fits.ImageHDU(data=self.data_weights, name='data_weights')
            random_pos_hdu = fits.ImageHDU(data=self.random_positions, name='random_positions')
            random_weight_hdu = fits.ImageHDU(data=self.random_weights, name='random_weights')
            rsep_hdu = fits.ImageHDU(data=[self.r_sep], name='r_sep')
            boxsize_hdu = fits.ImageHDU(data=self.boxsize, name='boxsize')
            boxcenter_hdu = fits.ImageHDU(data=self.boxcenter, name='boxcenter')
            boxlike_hdu = fits.ImageHDU(data=[int(self.box_like)], name='box_like')
            volume_hdu = fits.ImageHDU(data=[self.volume], name='volume')
            logger.info(f'Saving density mesh to {save_mesh}.fits')
            hdul = fits.HDUList([delta_hdu, data_pos_hdu, data_weight_hdu, random_pos_hdu, random_weight_hdu,
                                 rsep_hdu, boxsize_hdu, boxcenter_hdu, boxlike_hdu, volume_hdu])
            hdul.writeto(f'{save_mesh}.fits', overwrite=True)
            hdul.close()

        np.save("delta_test_versus.npy", self.delta)

