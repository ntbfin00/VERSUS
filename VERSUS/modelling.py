import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from scipy.integrate import trapezoid as trapz
from scipy.spatial import cKDTree
from scipy.interpolate import PchipInterpolator #, splrep, splev #change

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

    def fit_profile(self, Rmax=4, nbins=100, rbins=5):
        r"""
        Fit the rescaled (r / R) void density profile (in units of rho / rho_mean)
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
        # self.profile = (N / V).mean(axis=1) / self.rho_mean

        self.fit = PchipInterpolator(np.insert(self.r, 0, 0.), np.insert(self.profile, 0, 0., axis=1), axis=1)
        # self.err = np.sqrt(((N / V)**2).sum(axis=1)) / self.void_radii.size / self.rho_mean
        # self.err = np.where(self.err == 0., 1e-6, self.err)
        # self.fit = splrep(np.insert(self.r, 0, 0.), np.insert(self.profile, 0, 0.), w = 1 / np.insert(self.err, 0, 1e-6))

    def plot_profile(self, ax=None, save=None, **kwargs):
        show = False
        if ax is None: 
            fig, ax = plt.subplots(1)
            show = True
        ax.plot(self.r, self.profile.T, **kwargs)
        # ax.errorbar(self.r, self.profile, self.err, **kwargs)
        rr = np.linspace(self.r.min(), self.r.max(), 100)
        ax.plot(rr, self.fit(rr).T, c='k', ls='--')
        # ax.plot(rr, splev(rr, self.fit), c='k', ls='--') #change
        ax.set_ylabel(r"$\rho(r) / \bar{\rho}$", fontsize=14)
        ax.set_xlabel("$r/R_\mathrm{void}$", fontsize=14)
        ax.grid()
        plt.tight_layout()
        if save is not None: plt.savefig(save)
        if show: plt.show()

    def _F(self, r, R, rbias=1., Rmax=4, Npts=1000):
        r"""
        Argument for conditional error function.
        """

        # fit void density profile
        rr = np.linspace(0, r, Npts)
        if not hasattr(self, 'fit'): self.fit_profile(Rmax=Rmax)
        rbin = min(np.searchsorted(self.rsplit, R), self.rsplit.size - 1)
        rho = self.fit(rr / R / rbias)[rbin]
        # rho = splev(rr / R, self.fit)  # change

        # calculate average enclosed density
        rho_enc = trapz(rr**2 * rho, rr, axis=0) * 3 / r**3
        A = np.sqrt(2 * np.pi * self.rho_mean * r**3 / 3 / rho_enc)
        return A * (1 + self.void_delta - rho_enc)

    # REMOVE!
    # def test(self, robs, R):

        # from scripts.distributions import density_profile
        # rr = np.linspace(0, robs, 1000)
        # rho_enc = trapz(rr**2 * density_profile(rr / R, vf='versus'), rr, axis=0) * 3 / robs**3
        # A = np.sqrt(2 * np.pi * self.rho_mean * robs**3 / 3 / rho_enc)
        # F = A * (1 + self.void_delta - rho_enc)

        # P_survive = erfc(F) / 2
        # P_survive = np.append(P_survive, 1)
        # return (1 - P_survive[:-1]) * P_survive[1:][::-1].cumprod()[::-1]

    def _P_detect(self, robs, R, rbias=1.):
        r"""
        
        Probability of detecting a void with true radius R_true at radius r_obs (i.e. P(r_obs | R_true)).
        """

        P_survive = erfc(self._F(robs, R, rbias=rbias)) / 2
        P_survive = np.append(P_survive, 1)
        return (1 - P_survive[:-1]) * P_survive[1:][::-1].cumprod()[::-1]

    def N_detect(self, robs, radii, rbias=1.):
        r"""
        Number (and uncertainty) of voids detected in bins robs given true radii.
        """
        
        P_detect = np.array([self._P_detect(robs, R, rbias=rbias) for R in radii])

        return P_detect.sum(axis=0), np.sqrt((P_detect**2).sum(axis=0))

