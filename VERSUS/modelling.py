import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from scipy.integrate import trapezoid as trapz
from scipy.spatial import cKDTree
from scipy.interpolate import PchipInterpolator as Interpolate
import logging

logger = logging.getLogger(__name__)

class SizeBias:
    r"""

    Analytically determine radius uncertainty distributions for voids detected with the spherical void-finder.

    Parameters
    ----------
    gal_pos: array (N,3)
        Array of data positions in cartesian coordinates used to compute the density profile.
    void_pos: array (N,3)
        Array of detected void positions used to compute the density profile.
    void_radii: array (N)
        Array of detected void radii used to compute the density profile.
    void_delta: float
        Density threshold parameter used in void-finding step.
    rho_mean: float
        Mean tracer density within box/survey.
    Rmax: float, default=4
        Maximum radius (in units of void radius) to fit the density profile.
    """

    def __init__(self, gal_pos, void_pos, void_radii, void_delta, rho_mean, volume, Rmax=4):
        if void_pos.shape[0] != void_radii.size: 
            raise Exception("void_pos and void_radii must have matching dimensions along the first axis.") 
        self.gal_pos = gal_pos
        indx = np.argsort(void_radii)
        self.void_pos = void_pos[indx]
        self.void_radii = void_radii[indx]

        self.void_delta = void_delta
        self.rho_mean = rho_mean
        self.volume = volume

        self.Rmax = Rmax

    def fit_profile(self, nbins=100, rbins=5, rescale=True, anchor_point='linear'):
        r"""
        Fit the rescaled (r / R) void density profile (in units of rho / rho_mean)

        Parameters
        ----------
        nbins: int, default=100
            Number of bins used to measure density profile.
        rbins: int, default=5
            Number of radius bins in which to average the density profile. Used to account for profile evolution.
        rescale: bool, default=True
            Perform rescaling of density profile to ensure density threshold is crossed at detected radius. Not required if true density profile is known.
        anchor_point: str, default='linear'
            Anchor point used for rescaling density profile if rescale is True. 'Linear' is more robust to noise as it extrapolates to r(profile = 1 + void_delta) using the slope of the void boundary. Alternatively, 'fixed' directly determines r(profile = 1+ void_delta) crossing from profile.
        """

        self.r = np.linspace(0, self.Rmax, nbins)

        boxsize = np.ceil(np.abs(self.gal_pos.max(axis=0) - self.gal_pos.min(axis=0)))
        boxcenter = (self.gal_pos.max(axis=0) + self.gal_pos.min(axis=0)) / 2
        boxshift = boxsize/2 - boxcenter
        tree = cKDTree(self.gal_pos + boxshift, compact_nodes=False, 
                       balanced_tree=False, boxsize=boxsize)

        N = np.zeros((self.r.size, self.void_radii.size))
        V = np.zeros((self.r.size, self.void_radii.size))
        for (i,r) in enumerate(self.r[1:]):
            R = r * self.void_radii
            N[i+1] = tree.query_ball_point(self.void_pos + boxshift, R,  
                                           workers=-1, return_length=True)
            V[i+1] = 4 * np.pi * R**3 / 3 

        N = N[1:] - N[:-1]
        V = V[1:] - V[:-1]

        self.r = (self.r[None, 1:] + self.r[None, :-1]) / 2  # take midpoint
        rbinning = np.exp(np.log(10) + (np.log(len(self.void_radii)) - np.log(10)) *  np.arange(rbins) / (rbins - 1))[-2::-1]
        rbinning = (self.void_radii.size - rbinning).astype(int)
        self.rsplit = np.array([radii.mean() for radii in np.array_split(self.void_radii, rbinning)])
        self.profile = np.array([rho.mean(axis=1) for rho in np.array_split((N / V), rbinning, axis=1)]) / self.rho_mean

        if rescale: 
            if anchor_point == 'linear':
                # linear interpolation of slope around (1 + delta_v/2)
                indx = -np.argmax(self.profile[:,::-1] < (1 + self.void_delta/2), axis=1)

                # interpolation points
                rows = np.arange(len(indx))
                x1 = self.r[0][indx - 1]
                x2 = self.r[0][indx]
                y1 = self.profile[rows, indx - 1]
                y2 = self.profile[rows, indx]

                # gradient
                m = (y2 - y1) / (x2 - x1)

                # offset
                c = y1 - m*x1

                # crossing
                r_c = (1 + self.void_delta - c) / m 

                self.r = self.r / r_c[:, None]
            elif anchor_point == 'fixed':
                # determine radius proxy (point of 1 + delta_v crossing)
                indx = -np.argmax(self.profile[:,::-1] < (1 + self.void_delta), axis=1)
                self.r = self.r / self.r[0][indx-1][:, None]
            else:
                raise Exception("Anchor point not recognised. Please use either 'linear' or 'fixed'.")
            
        # ensure extrapolation to rho/rho_mean = 1
        upper_lim = np.maximum(self.r[:,-1], self.Rmax)
        lims = np.c_[np.zeros(self.r.shape[0]),  upper_lim + 0.1, upper_lim + 0.2]
        r = np.insert(self.r, [0, self.r.shape[1], self.r.shape[1]], lims, axis=1)
        profile = np.insert(self.profile, [0, self.r.shape[1], self.r.shape[1]], [0, 1, 1], axis=1)
        self.splines = [
                    Interpolate(r[i if rescale else 0], profile[i], extrapolate=True) 
                    for i in range(self.profile.shape[0])
                    ]

    def interpolate(self, r, R):
        r = np.atleast_2d(r)
        R = np.atleast_1d(R)

        fit = np.zeros((r.shape[0], r.shape[1], R.size))
        rbin = np.minimum(np.searchsorted(self.rsplit, R), self.rsplit.size - 1)
        for k in np.unique(rbin):
            mask = rbin == k
            fit[:, :, mask] = self.splines[k](r[:, :, None] / R[mask])
        return fit


    def plot_profile(self, ax=None, save=None, **kwargs):
        show = False
        if ax is None: 
            fig, ax = plt.subplots(1)
            show = True
        ax.plot(self.r.T, self.profile.T, 
                label=[fr"$\langle R_v \rangle = {R:.1f}$" for R in self.rsplit])
        ax.legend(loc='lower right')
        rr = np.linspace(0, self.Rmax, 100)
        profile_interp = self.interpolate(rr[:, None] * self.rsplit, self.rsplit)
        for i in range(self.profile.shape[0]):
            ax.plot(rr, profile_interp[:,i,i], c='k', ls='--', lw=0.8)
        ax.set_ylabel(r"$\rho(r) / \bar{\rho}$", fontsize=14)
        ax.set_xlabel("$r/R_\mathrm{void}$", fontsize=14)
        ax.set_xlim(0, self.Rmax)
        ax.grid()
        plt.tight_layout()
        if save is not None: plt.savefig(save)
        if show: plt.show()


    def _F(self, r, R, Npts=1000):
        r"""
        Argument for conditional error function.
        """

        r = r[:, None]

        # fit void density profile
        rr = np.linspace(0, r, Npts)
        if not hasattr(self, 'splines'): self.fit_profile()
        rho = self.interpolate(rr[:,:,0], R)

        # calculate average enclosed density
        rho_enc = trapz(rr**2 * rho, rr, axis=0) * 3 / r**3
        A = np.sqrt(2 * np.pi * self.rho_mean * r**3 / 3 / rho_enc)
        return A * (1 + self.void_delta - rho_enc)

    def _P_detect(self, robs, R, Npts=1000):
        r"""
        
        Probability of detecting a void with true radius R radius r_obs (i.e. P(r_obs | R)).
        """

        P_survive = erfc(self._F(robs, R, Npts=Npts)) / 2
        P_survive = np.vstack([P_survive, np.ones((1, P_survive.shape[1]))])
        return (1 - P_survive[:-1]) * P_survive[1:][::-1].cumprod(axis=0)[::-1]

    def N_detect(self, robs, radii):
        r"""
        Number (and uncertainty) of voids detected in bins robs given true radii.
        """
        
        P_detect = self._P_detect(robs, radii)

        return P_detect.sum(axis=-1), np.sqrt((P_detect**2).sum(axis=-1))

    def counts_to_vsf(self, rbins, Nvoids):
        r"""
        Convert void number counts to void size function.
        """

        indx = np.argsort(rbins)[::-1]
        r = np.array(rbins)[indx]
        N = np.array(Nvoids)[indx][1:]
        vsf = np.zeros(len(r) - 1)
        for i in range(r.size - 1):
            norm = self.volume * np.log(r[i] / r[i+1])
            vsf[i] = N[i] / norm
        return vsf[::-1]

    def correct_vsf(self, robs, vsf_theory, Npts=1000, **vsf_kwargs):
        r"""
        Calculate the correction to the void size function.

        Parameters
        ----------
        robs : array
            Input radius bins used for void-finding.
        vsf_theory : function, array (2, N)
            Void size function theory of the form dn/dlnR. If passed as a function, it should take radius as an input. Other parameters can be passed using vsf_kwargs. If passed as an array, the first dimension should hold radius values and the second dimension should hold the VSF predictions.

        """

        # set integration limits
        R_lims = (1e-2, robs.max() * 1.1)

        # if vsf_theory is provided as a function of radius N(R)
        if callable(vsf_theory):
            RR = np.linspace(*R_lims, Npts)
            N_theory = self.volume * vsf_theory(RR, **vsf_kwargs) / RR
        else:
            vsf_theory = np.array(vsf_theory)
            # check if the theory has form (radii, Nvoids)
            if vsf_theory.shape[0] != 2:
                raise Exception("vsf_theory does not have shape (2, N). Provide both radii and values for theory prediction")
            RR = vsf_theory[0]
            N_theory = vsf_theory[1] * self.volume / RR
            # check if the theory has been provided over a suitable range
            if (RR.min() > R_lims[0]) or (RR.max() < R_lims[1]):
                raise Exception("Ensure that theory has been provided in the range r=[{:.1f}, {:.1f}]".format(*R_lims))
        N_corrected = trapz(N_theory * self._P_detect(robs, RR), RR, axis=-1)

        return self.counts_to_vsf(robs, N_corrected)


