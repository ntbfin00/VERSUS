import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from scipy.integrate import trapezoid as trapz
from scipy.spatial import cKDTree
from scipy.interpolate import PchipInterpolator as Interpolate

class SizeBias:
    r"""

    Analytically determine radius uncertainty distributions for voids detected with the spherical void-finder.
    """

    def __init__(self, gal_pos, void_pos, void_radii, void_delta, rho_mean):
        self.void_delta = void_delta
        self.rho_mean = rho_mean

        self.gal_pos = gal_pos
        indx = np.argsort(void_radii)
        self.void_pos = void_pos[indx]
        self.void_radii = void_radii[indx]

    def fit_profile(self, Rmax=4, nbins=100, rbins=5, rescale=True):
        r"""
        Fit the rescaled (r / R) void density profile (in units of rho / rho_mean)

        Parameters
        ----------

        Rmax: float, default=4
            Maximum radius (in units of void radius) to fit the density profile.
        nbins: int, default=100
            Number of bins used to measure density profile.
        rbins: int, default=5
            Number of radius bins in which to average the density profile. Used to account for profile evolution.
        rescale: bool, default=True
            Perform rescaling of density profile to ensure density threshold is crossed at detected radius. Not required if true density profile is known.
        """

        self.r = np.linspace(0, Rmax, nbins)

        boxsize = np.ceil(np.abs(self.gal_pos.max(axis=0) - self.gal_pos.min(axis=0)))# + 1e-5
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

        self.r = (self.r[1:] + self.r[:-1]) / 2  # take midpoint
        self.rsplit = np.array([radii[0] for radii in np.array_split(self.void_radii, rbins)])
        self.profile = np.array([rho.mean(axis=1) for rho in np.array_split(N / V, rbins, axis=1)]) / self.rho_mean

        if rescale: 
            # determine radius proxy (point of 1 + delta_v crossing)
            indx = -np.argmax(self.profile[:,::-1] < (1 + self.void_delta), axis=1) -1
            self.r = self.r[None, :] / self.r[indx][:, None]
            splines = [
                    Interpolate(np.insert(self.r[i], 0, 0.), 
                                np.insert(self.profile[i], 0, 0.)) 
                    for i in range(self.r.shape[0])
                    ]
            self.fit = lambda r : np.array([spl(r) for spl in splines])
        else:
            self.fit = Interpolate(np.insert(self.r, 0, 0.), 
                                   np.insert(self.profile, 0, 0., axis=1), 
                                   axis=1)


    def plot_profile(self, ax=None, save=None, **kwargs):
        show = False
        if ax is None: 
            fig, ax = plt.subplots(1)
            show = True
        ax.plot(self.r.T, self.profile.T, **kwargs)
        rr = np.linspace(self.r.min(), self.r.max(), 100)
        ax.plot(rr, self.fit(rr).T, c='k', ls='--')
        ax.set_ylabel(r"$\rho(r) / \bar{\rho}$", fontsize=14)
        ax.set_xlabel("$r/R_\mathrm{void}$", fontsize=14)
        ax.grid()
        plt.tight_layout()
        if save is not None: plt.savefig(save)
        if show: plt.show()

    def _F(self, r, R, Rmax=4, Npts=1000):
        r"""
        Argument for conditional error function.
        """

        # fit void density profile
        rr = np.linspace(0, r, Npts)
        if not hasattr(self, 'fit'): self.fit_profile(Rmax=Rmax)
        rbin = min(np.searchsorted(self.rsplit, R), self.rsplit.size - 1)
        rho = self.fit(rr / R)[rbin]

        # calculate average enclosed density
        rho_enc = trapz(rr**2 * rho, rr, axis=0) * 3 / r**3
        A = np.sqrt(2 * np.pi * self.rho_mean * r**3 / 3 / rho_enc)
        return A * (1 + self.void_delta - rho_enc)

    def _P_detect(self, robs, R):
        r"""
        
        Probability of detecting a void with true radius R_true at radius r_obs (i.e. P(r_obs | R_true)).
        """

        P_survive = erfc(self._F(robs, R)) / 2
        P_survive = np.append(P_survive, 1)
        return (1 - P_survive[:-1]) * P_survive[1:][::-1].cumprod()[::-1]

    def N_detect(self, robs, radii):
        r"""
        Number (and uncertainty) of voids detected in bins robs given true radii.
        """
        
        P_detect = np.array([self._P_detect(robs, R) for R in radii])

        return P_detect.sum(axis=0), np.sqrt((P_detect**2).sum(axis=0))

