import numpy as np
from scipy.integrate import cumtrapz
from scipy.interpolate import PchipInterpolator

def Abacus_VSF(r):
    # Size distribution of voids measured by VERSUS algorithm with delta<-0.8 in Abacus c000_ph000_hod001 simulation
    # R = [30, 34, 38, 42, 46, 50, 54], Total voids: 6035
    return 5e5 * np.exp(-r / 6) - 80

def density_profile(r, vf='VERSUS'):
    if vf == 'VERSUS':
        # return 0.85 * np.exp(((0.3 * np.sqrt(x)) ** (1.5 * x)) - ((0.93 * x) ** (-11 * x))) + 0.11
        return 0.9 * np.exp(200 * (3*r - 4) * np.exp(-5 * r)) + 0.1
    else:
        raise Exception("Analytic model for chosen void-finder has not been implemented")

def sample_density(R, Rmax, rho_mean, vf='VERSUS', npts=5000):

    PDF = lambda r, R: r**2 * density_profile(r / R, vf=vf)

    # compute CDF and mean counts within Rmax 
    r_grid = np.linspace(1e-6, Rmax, npts)
    CDF = cumtrapz(PDF(r_grid, R), r_grid, initial=0.0)
    mean_counts = 4 * np.pi * rho_mean * CDF[-1]
    CDF /= CDF[-1]

    # Poisson sample expected counts
    N = np.random.poisson(mean_counts)

    # inverse transform sample radii
    u = np.random.rand(N)
    CDF_inv = PchipInterpolator(CDF, r_grid)
    r_samples = CDF_inv(u)

    # randomly distribute positions within Rmax
    mu = 2 * np.random.rand(N) - 1
    phi = 2 * np.pi * np.random.rand(N)
    sint = np.sqrt(1 - mu**2)

    x = r_samples * sint * np.cos(phi)
    y = r_samples * sint * np.sin(phi)
    z = r_samples * mu

    return np.vstack([x, y, z]).T#, r_samples

    

# --- Define density profile ---
# import matplotlib.pyplot as plt
# rho_mean = 0.1
# R = 35
# Rmax = 3 * R
# # particles, r_samples, Nexp, N = sample_poisson_particles(R, rho_mean, Rmax)
# particles, r_samples = sample_density(R, Rmax, rho_mean) 

# # --- Diagnostics: histogram vs analytic PDF ---
# r_test = np.linspace(0.01, Rmax, 400)
# pdf = r_test**2 * density_profile(r_test / R)
# norm = np.trapz(pdf, r_test)  # normalize
# pdf /= norm

# fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8,4))
# ax1.hist(r_samples, bins=50, density=True, histtype='step', label='sampled radii')
# ax1.plot(r_test, pdf, 'r-', lw=2, label='analytic $p(r)$')
# ax1.set_xlabel("r")
# ax1.set_ylabel("Probability density")
# ax1.legend()

# counts, bins = np.histogram(r_samples, bins=50)
# ax2.hlines(rho_mean, 0, Rmax, ls='--', color='k', alpha=0.5)
# ax2.plot(r_test, rho_mean * density_profile(r_test / R), 'r-', lw=2, label=r'analytic $\rho(r)$')
# ax2.stairs(counts  / (4/3 * np.pi * (bins[1:]**3 - bins[:-1]**3)), bins)
# plt.tight_layout()
# plt.show()
