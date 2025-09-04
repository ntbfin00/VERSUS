import numpy as np
import random
import argparse
import sys
import os
from scipy.spatial import KDTree
from distributions import Abacus_VSF, sample_density
from itertools import chain

parser = argparse.ArgumentParser(description='Compute void density profile')
parser.add_argument('-n', '--rho_mean', required=True, type=float, help="Number density of tracers")
parser.add_argument('-vf', '--voidfinder', required=True, type=str, help="Void-finder density profile to imprint")
parser.add_argument('--boxsize', type=float, default=2000)
args = parser.parse_args()

# initialise the random seeds
random.seed(a=42)
np.random.seed(42)

# inputs
Rmin, Rmax = (30, 50)
Ncand      = int(1e5) # number of candidate voids
R_extent   = 2.5  # factor of void radius to imprint void density profile

# derived quantities
boxsize = args.boxsize
Ngal = int(args.rho_mean * args.boxsize**3)
print(f'n={args.rho_mean:.4f} | N_gals={Ngal} | Ncand={Ncand} | Rmin={Rmin} | Rmax={Rmax} | Rextent={R_extent:.1f}')


# define the arrays with the positions and radii of the voids
gal_pos  = np.random.rand(Ngal, 3).astype(np.float32) * boxsize 
void_pos = np.random.rand(Ncand, 3).astype(np.float32) * boxsize 
tree = KDTree(gal_pos, boxsize=boxsize, compact_nodes=False, balanced_tree=False)

# sample Abacus-like void size distribution
population = np.linspace(Rmin, Rmax, 1000)
weights    = Abacus_VSF(population)
radii      = np.array(random.choices(population, weights, k=Ncand))
radii      = -np.sort(-radii.astype(np.float32))

# position voids in simulation
Nvoids = 0
for i in range(Ncand):

    # generate the position and radius of new void
    cand_pos = void_pos[i]
    R        = radii[i]

    if (i%(Ncand // 10))==0: print(f"Placing candidate void (R={R:.1f}): {i}, Nvoids={Nvoids}")

    # skip if a void already overlaps with candidate
    diff = np.abs(cand_pos - void_pos[:Nvoids])
    dist = np.sqrt(np.where(diff>boxsize/2, (diff-boxsize)**2, diff**2).sum(axis=1))
    if (dist < (R_extent * (R + radii[:Nvoids]))).any(): 
        continue
    else:
        void_pos[Nvoids] = cand_pos
        radii[Nvoids]    = R
        Nvoids          += 1

void_pos = void_pos[:Nvoids]
radii = radii[:Nvoids]

# remove galaxies in voids
in_R = np.zeros(gal_pos.shape[0], dtype=bool)
indx = list(chain(*tree.query_ball_point(void_pos, R_extent * radii)))
in_R[indx] = True
gal_pos = gal_pos[~in_R]

# imprint void densities
pos_in_void = []
for i in range(Nvoids):
    if (i%(Nvoids // 10))==0: print(f"Imprinting void density (R={radii[i]:.1f}): {i}")
    pos_in_void.append(sample_density(radii[i], R_extent * radii[i], 
                       args.rho_mean, vf=args.voidfinder) + void_pos[i])

pos_in_void.append(gal_pos)
gal_pos = np.concatenate(pos_in_void, axis=0)

# wrap positions
gal_pos = np.where(gal_pos>boxsize, gal_pos-boxsize, gal_pos)
gal_pos = np.where(gal_pos<0, gal_pos+boxsize, gal_pos)

print(f"Remaining galaxies: {gal_pos.shape[0]} (initial = {Ngal})")
print(f"Number of voids: {Nvoids} (initial = {Ncand})")

# save to file
save_dir = "sims/"
append_fn = f"_rho{args.rho_mean}_{args.voidfinder}.npy"
if not os.path.exists(save_dir): 
    print("making")
    os.makedirs(save_dir)
else:
    print("exists")

np.save(f"{save_dir}/gal_pos{append_fn}", gal_pos)
np.save(f"{save_dir}/void_pos{append_fn}", void_pos)
np.save(f"{save_dir}/void_rad{append_fn}", radii)
print(f"Outputs saved to file: simulated_fields/*{append_fn}")


# import matplotlib.pyplot as plt

# def compute_vsf(radii):
    # bins = np.arange(Rmin, Rmax+2, 2)
    # print(f"Histogram bins: {bins}")
    # counts = np.histogram(radii, bins=bins)[0]

    # vsf = np.zeros((3, counts.size))
    # for i in range(counts.size):
        # norm = 1 / (BoxSize**3 * np.log(bins[i+1] / bins[i]))
        # vsf[0,i] = np.sqrt(bins[i] * bins[i+1])  geometric mean radius for logarithmic scale
        # vsf[1,i] = counts[i] * norm  vsf
        # vsf[2,i] = np.sqrt(counts[i]) * norm  poisson uncertainty
    # vsf[0] = vsf[0][::-1]
    # vsf[1] = vsf[1][::-1]
    # vsf[2] = vsf[2][::-1]
    # vsf_fn = f"vsf/input_vsf_Nv{counts.sum()}_R{Rmin}-{Rmax}"
    # np.save(vsf_fn+".npy", vsf)


# plot VSF
# plt.errorbar(*vsf, color='k', marker='o', markersize=2, capsize=2)
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('R')
# plt.ylabel('dN/dlnR')
# plt.savefig(vsf_fn+".png")
# plt.show()
