import numpy as np
import os
from astropy.io import fits
import logging

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

    dtype: string
        Mesh data type.

    reconstruct: string
        Type of reconstruction to run - 'disp', 'rsd' or 'disp+rsd'. Defaults to no reconstruction.

    recon_args: dict
        Reconstruction arguments - 'f', 'bias', 'engine', 'los' (only required for box), 'smoothing radius' and 'recon_pad'.

    kwargs : dict
        Optional arguments.
    """

    def __init__(self, data_positions, data_weights=None, random_positions=None, random_weights=None, 
                 data_cols=None, dtype='f4', reconstruct=None, recon_args=None, **kwargs):

        # if randoms are supplied then treat as survey
        self.box_like = True if random_positions is None else False
        logger.info('Loading {}-like data'.format('box' if self.box_like else 'survey'))
        self.dtype = dtype
        logger.debug(f'Data type: {self.dtype}')
        # set reconstruction parameters
        self.reconstruct = reconstruct
        self.recon_args = {'f': 0.8, 'bias': 2.} if recon_args is None else recon_args
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
            positions = np.load(data_fn)[:,:3]
            # read weights if 4th column provided
            if len(data_cols)==4: weights = np.load(data_fn)[:,3]
        else:
            raise Exception("File type not recognised. Please provide a file ending in .fits or .npy.")

        # if x or y contained in data_cols assume positions are provided in cartesian coordinates
        # else convert sky positions to cartesian
        if not np.array([a in data_cols for a in ('X','Y','x','y')]).any():
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


    def _set_mesh(self, engine='IterativeFFTParticleReconstruction', cellsize=1., boxpad=1., **kwargs):
        r"""
        Set the mesh properties and type of reconstruction algorithm

        Parameters
        ----------
        engine: string
            Reconstruction algorithm passed to pyrecon.

        cellsize: float, default=1.
            Size of mesh cell.

        boxpad: float, default=1.1
            Padding applied to mesh.
        """
        self.engine = engine

        # use galaxies (or randoms for survey) to estimate boxsize if not supplied
        if self.box_like:
            positions = self.data_positions
            wrap = True
        else:
            positions = self.random_positions
            wrap = False

        # select reconstruction algorithm
        exec(f"from pyrecon import {engine}; Recon = {engine}", globals())

        # initialise mesh
        mesh = Recon(positions=positions, cellsize=cellsize, wrap=wrap,
                     boxpad=boxpad, dtype=self.dtype, **kwargs)

        return mesh

    def _set_mesh_density(self, mesh, smoothing_radius=0., **kwargs):
        r"""
        Populate mesh with galaxies and randoms

        Parameters
        ----------
        mesh: pyrecon.recon.BaseReconstruction object
            Input density mesh.
            
        smoothing_radius : float, default=0.
            Gaussian radius as factor of cellsize with which to smooth data and random fields before computing overdensity.
        """

        # assign data
        mesh.assign_data(self.data_positions, self.data_weights)
        # assign randoms
        if not self.box_like: mesh.assign_randoms(self.random_positions, self.random_weights)

        # save data and randoms otherwise deleted by Pyrecon
        # self.mesh_data = mesh.mesh_data
        # self.mesh_randoms = mesh.mesh_randoms

        # manually apply smoothing
        if self.engine == 'IterativeFFTParticleReconstruction' and smoothing_radius>0.:
            mesh.mesh_data.smooth_gaussian(smoothing_radius * self.cellsize)
            if not self.box_like: mesh.mesh_randoms.smooth_gaussian(smoothing_radius * self.cellsize)

        # calculate mesh overdensity
        mesh.set_density_contrast(**kwargs)


    def run_recon(self, f=0.8, bias=2, engine='IterativeFFTReconstruction', los='z', 
                  recon_pad=1.1, smoothing_radius=15., field='rsd', **kwargs):
        r"""
        Perform reconstruction on galaxy positions using pyrecon (https://github.com/cosmodesi/pyrecon.git)
        """

        if not self.box_like: los = None  # survey has local line-of-sight

        logger.info(f"Running {field} reconstruction with {engine}")
        logger.info(f"Recon parameters: f={f:.1f}, b={bias:.1f}, los={los}, r_smooth={smoothing_radius}, pad={recon_pad}")
        # set and smooth mesh
        self.data_mesh = self._set_mesh(f=f, bias=bias, engine=engine, cellsize=self.cellsize, los=los,
                                        boxpad=recon_pad, fft_engine='fftw', fft_plan='estimate', **kwargs)
        self._set_mesh_density(self.data_mesh, smoothing_radius=smoothing_radius)
        # run reconstruction
        logger.debug(f"Reconstruction running on {self.data_mesh.nmesh} mesh.")
        self.data_mesh.run()
        logger.debug("Reconstruction complete. Setting new positions.")
        # read reconstructed positions
        self.data_positions = self.data_mesh.read_shifted_positions(self.data_positions, field=field)
        if self.random_positions is not None and 'disp' in field: 
            self.random_positions = self.data_mesh.read_shifted_positions(self.random_positions, field='disp')  # RecIso
        del self.data_mesh

        
    def size_mesh(self, niterations=4, cells_per_r_sep=2, smoothing_radius=0., **kwargs):
        r"""
        Estimate the survey volume, mean density and average galaxy separation.

        Parameters
        ----------
        niterations : int, default=4
            Number of iterations used to estimate the average galaxy separation.

        cells_per_r_sep : float, default=2.
            Number of cells used in size estimation as a fraction of the average galaxy separation.

        smoothing_radius : float, default=0.
            Gaussian radius as factor of cellsize with which to smooth data and random fields before computing overdensity.

        ran_min : float, default=0.01
            Minimum fraction of average randoms for cell to be counted as part of survey.

        kwargs : dict
            Optional arguments for pyrecon.set_density_contrast().
        """

        logger.info(f'Estimating volume and average galaxy separation (Ngal = {self.N_data})')
        # create mesh flush with survey volume
        mesh = self._set_mesh(boxpad=1.)
        # first estimate of volume (true if box)
        self.volume = np.prod(mesh.boxsize)
        # first estimate of mean density
        self.rho_mean = self.W_data / self.volume
        # first estimate of average separation between galaxies
        self.r_sep = (4 * np.pi * self.rho_mean / 3)**(-1/3)  

        # iterate for better estimate of survey volume
        if not self.box_like:
            for i in range(niterations):
                logger.debug(f'Iteration {i} of volume estimation')
                # set cellsize to half estimated galaxy separation
                self.cellsize = self.r_sep/cells_per_r_sep
                mesh = self._set_mesh(boxpad=1., cellsize=self.cellsize, bias=1.)
                # determine survey boundary (cells outside have delta=0.)
                self._set_mesh_density(mesh, smoothing_radius=smoothing_radius, **kwargs)
                survey_mask = mesh.mesh_delta.value != 0.
                # estimate survey volume
                self.volume = survey_mask.sum() * mesh.cellsize.prod()
                # estimate density using cells inside survey
                self.rho_mean = (self.W_data/self.W_random) * mesh.mesh_randoms.value[survey_mask]
                self.rho_mean = self.rho_mean.mean() / mesh.cellsize.prod()  # must scale by cellsize
                # estimate average galaxy separation
                self.r_sep = (4 * np.pi * self.rho_mean / 3)**(-1/3)
        # del self.mesh_data, self.mesh_randoms
        self.cellsize = self.r_sep/cells_per_r_sep
        logger.info(f'Cellsize set to {self.cellsize:.2f} ({cells_per_r_sep:.1f} cells per average separation)') 

        
    def create_mesh(self, boxpad=1.1, cells_per_r_sep=2., smoothing_radius=0., save_mesh=None, **kwargs):
        r"""
        Create the density mesh

        Parameters
        ----------
        pad : float, default=1.1
            Padding factor for survey mesh. Simulation box has no padding.

        cells_per_r_sep: float, default=2.
            Number of mesh cells per average galaxy separation. Used to set cellsize.

        smoothing_radius : float, default=0.
            Smoothing scale for random field.

        ran_min : float, default=0.01
            Minimum fraction of average randoms for cell to be counted as part of survey.

        save_mesh : bool, string, default=None
            If not ``None``, path where to save the mesh in FITS format.
            If ``True``, the mesh will be saved in the default path: f'mesh/mesh_<nbins_vf>_<dtype>.fits'.

        kwargs : dict
            Optional arguments.
        """

        # estimate cellsize based on galaxy density
        if not hasattr(self, 'cellsize'): self.size_mesh(cells_per_r_sep=cells_per_r_sep, smoothing_radius=smoothing_radius, **kwargs)

        # run optional reconstruction
        if self.reconstruct is not None: self.run_recon(field=self.reconstruct, **self.recon_args)

        # generate mesh
        mesh = self._set_mesh(cellsize=self.cellsize,
                              boxpad=1. if self.box_like else boxpad, # pad survey to better detect boundary voids
                              bias=1.)  # bias set to 1. so voids are found on galaxy (not matter) field

        logger.info(f'Estimating mesh density (nmesh={mesh.nmesh})')
        self._set_mesh_density(mesh, smoothing_radius=smoothing_radius, **kwargs) 
        del self.data_positions
        del self.data_weights
        del self.random_positions
        del self.random_weights

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
            # delta mesh
            delta_hdu = fits.PrimaryHDU(self.delta)
            # cellsize
            cellsize_hdu = fits.ImageHDU(data=[self.cellsize], name='cellsize')
            rsep_hdu = fits.ImageHDU(data=[self.r_sep], name='r_sep')
            boxsize_hdu = fits.ImageHDU(data=self.boxsize, name='boxsize')
            boxcenter_hdu = fits.ImageHDU(data=self.boxcenter, name='boxcenter')
            boxlike_hdu = fits.ImageHDU(data=[int(self.box_like)], name='box_like')
            # save
            logger.info(f'Saving density mesh to {save_mesh}.fits')
            hdul = fits.HDUList([delta_hdu, cellsize_hdu, rsep_hdu, boxsize_hdu, boxcenter_hdu, boxlike_hdu])
            hdul.writeto(f'{save_mesh}.fits', overwrite=True)
            hdul.close()

