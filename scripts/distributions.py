import numpy as np
from scipy.integrate import cumtrapz
from scipy.interpolate import PchipInterpolator, CubicSpline

def Abacus_VSF(r):
    # Size distribution of voids measured by VERSUS algorithm with delta<-0.8 in Abacus c000_ph000_hod001 simulation
    # R = [30, 34, 38, 42, 46, 50, 54], Total voids: 6035
    return 5e5 * np.exp(-r / 6) - 80

def N_theory(r, volume=2000**3):
    # vsf computed for r = np.linspace(10, 55, 100) from CosmoBolognaLib.Cosmology.size_function
    vsf = np.load("../data/vsf_theory_interpolated.npy", allow_pickle=True)[()]
    dN = volume * np.diff(r)[0] * vsf(r) / r 
    return dN
    

def density_profile(r, vf):
    if vf == 'versus':
        delta = 0.84 * (np.exp(-5.5e4 * np.exp(-11 * r) - np.exp(-2 * r)) - 1)
    elif vf == 'vide':
        delta = 0.93 * (1 / (1 + np.exp(-6.4 * r + 7)) - 1)
    else:
        raise Exception("Analytic model for chosen void-finder has not been implemented")
    return delta + 1

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
