import numpy as np
import random
import argparse
import sys
import os
from scipy.spatial import KDTree
from distributions import Abacus_VSF, sample_density, N_theory
from itertools import chain

parser = argparse.ArgumentParser(description='Compute void density profile')
parser.add_argument('-n', '--rho_mean', required=True, type=float, help="Number density of tracers")
parser.add_argument('-vf', '--voidfinder', required=True, type=str, help="Void-finder density profile to imprint")
parser.add_argument('--boxsize', type=float, default=2000)
parser.add_argument('--R_extent', type=float, default=2.5, help="Factor of void radius to imprint void density profile")
parser.add_argument('--save_dir', required=False, type=str, default='sims/', help="Directory to save simulations")
args = parser.parse_args()

# initialise the random seeds
random.seed(a=42)
np.random.seed(42)

# inputs
# Rmin, Rmax = (30, 50)
Rmin, Rmax = (35, 55)
Ncand      = 5000 # number of candidate voids
R_extent   = args.R_extent

# derived quantities
boxsize = args.boxsize
Ngal = int(args.rho_mean * args.boxsize**3)
print(f'n={args.rho_mean:.4f} | N_gals={Ngal} | Ncand={Ncand} | Rmin={Rmin} | Rmax={Rmax} | Rextent={R_extent:.1f}')


# define the arrays with the positions and radii of the voids
cand_pos = np.random.rand(Ncand, 3).astype(np.float32) * boxsize 
gal_pos  = np.random.rand(Ngal, 3).astype(np.float32) * boxsize 
tree = KDTree(gal_pos, boxsize=boxsize + 1e-3, compact_nodes=False, balanced_tree=False)

# sample Abacus-like void size distribution
# population = np.linspace(Rmin, Rmax, 1000)
# weights    = Abacus_VSF(population)
# radii      = np.array(random.choices(population, weights, k=Ncand))
# radii      = -np.sort(-radii.astype(np.float32))

# sample void size distribution from theory
radii = np.linspace(Rmin, Rmax, 100)
N_radii = np.rint(N_theory(radii, volume=boxsize**3)).astype(np.int32)
radii = radii[N_radii > 0]
N_radii = N_radii[N_radii > 0]
N_radii_tot = N_radii.sum()
void_pos = np.zeros((N_radii_tot, 3))
void_rad = np.zeros(N_radii_tot)

# position voids in simulation
Nvoids_tot = 0
pos_i = -1 
# for i in range(Ncand):
for (R,N_R) in zip(radii[::-1], N_radii[::-1]):
    Nvoids = 0
    while Nvoids < N_R:
        pos_i += 1

        # generate the position and radius of new void
        pos = cand_pos[pos_i]

        if (Nvoids_tot%(N_radii_tot // 10))==0: print(f"Placing candidate void (R={R:.1f}): Nvoids={Nvoids_tot}/{N_radii_tot}")

        # skip if a void already overlaps with candidate
        diff = np.abs(pos - void_pos[:Nvoids_tot])
        dist = np.sqrt(np.where(diff>boxsize/2, (diff-boxsize)**2, diff**2).sum(axis=1))
        if (dist < (R_extent * (R + void_rad[:Nvoids_tot]))).any(): 
            continue
        else:
            void_pos[Nvoids_tot] = pos
            void_rad[Nvoids_tot] = R
            Nvoids              += 1
            Nvoids_tot          += 1

# remove galaxies in voids
in_R = np.zeros(gal_pos.shape[0], dtype=bool)
indx = list(chain(*tree.query_ball_point(void_pos, R_extent * void_rad)))
in_R[indx] = True
gal_pos = gal_pos[~in_R]

# imprint void densities
pos_in_void = []
for i in range(N_radii_tot):
    if (i%(N_radii_tot // 10))==0: print(f"Imprinting void density (R={void_rad[i]:.1f}): {i} / {N_radii_tot}")
    pos_in_void.append(sample_density(void_rad[i], R_extent * void_rad[i], 
                       args.rho_mean, vf=args.voidfinder) + void_pos[i])

pos_in_void.append(gal_pos)
gal_pos = np.concatenate(pos_in_void, axis=0)

# wrap positions
gal_pos = np.where(gal_pos>boxsize, gal_pos-boxsize, gal_pos)
gal_pos = np.where(gal_pos<0, gal_pos+boxsize, gal_pos)

print(f"rho_mean = {gal_pos.shape[0] / boxsize**3}")
print(f"Remaining galaxies: {gal_pos.shape[0]} (initial = {Ngal})")
print(f"Number of voids: {Nvoids_tot}")

# save to file
append_fn = f"_rho{args.rho_mean}_{args.voidfinder}.npy"
if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)
np.save(f"{args.save_dir}gal_pos{append_fn}", gal_pos)
np.save(f"{args.save_dir}void_pos{append_fn}", void_pos)
np.save(f"{args.save_dir}void_rad{append_fn}", void_rad)
print(f"Outputs saved to file: {args.save_dir}*{append_fn}")
