import numpy as np
from scipy.integrate import cumtrapz
from scipy.interpolate import PchipInterpolator

def Abacus_VSF(r):
    # Size distribution of voids measured by VERSUS algorithm with delta<-0.8 in Abacus c000_ph000_hod001 simulation
    # R = [30, 34, 38, 42, 46, 50, 54], Total voids: 6035
    return 5e5 * np.exp(-r / 6) - 80

def density_profile(r, vf):
    if vf == 'versus':
        return 0.9 * np.exp(200 * (10*r - 13) * np.exp(-6 * r)) + 0.1
    elif vf == 'zobov':
        return np.exp(16 * (r - 1) * np.exp(-3.5 * r))
    elif vf == 'zobov_baryc':
        return 0.9 * 0.004 ** (np.cos(3.2 * r - 1.1) * (np.exp(-2.8 * r) )) + 0.1
    elif vf == 'vide':
        return np.exp(4 / 5 * (6 * r - 5) * np.exp(- r * np.exp(r) / 2))
    elif vf == 'voxel':
        return np.exp(-120 / (np.exp(6 * r) + 20))
    elif vf == 'n12':
        return np.where(r < 1, r**12, 1)
    else:
        raise Exception("Analytic model for chosen void-finder has not been implemented")

def sample_density(R, Rmax, rho_mean, vf='VERSUS', npts=5000):

    PDF = lambda r, R: r**2 * density_profile(r / R, vf=vf)

    # compute CDF and mean counts within Rmax 
    rr = np.linspace(1e-6, Rmax, npts)
    CDF = cumtrapz(PDF(rr, R), rr, initial=0.0)
    mean_counts = 4 * np.pi * rho_mean * CDF[-1]
    CDF /= CDF[-1]

    # Poisson sample expected counts
    N = np.random.poisson(mean_counts)

    # inverse transform sample radii
    u = np.random.rand(N)
    CDF_inv = PchipInterpolator(CDF, rr)
    r_samples = CDF_inv(u)

    # randomly distribute positions within Rmax
    mu = 2 * np.random.rand(N) - 1
    phi = 2 * np.pi * np.random.rand(N)
    sint = np.sqrt(1 - mu**2)

    x = r_samples * sint * np.cos(phi)
    y = r_samples * sint * np.sin(phi)
    z = r_samples * mu

    return np.vstack([x, y, z]).T
