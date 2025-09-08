import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.spatial import cKDTree
import argparse
from distributions import density_profile

parser = argparse.ArgumentParser(description='Compute void density profile')
parser.add_argument('-g', '--gal_pos', type=str, 
                    nargs='+', help="Path to galaxy positions")
parser.add_argument('-v', '--void_pos', type=str, 
                    nargs='+', help="Path to void positions")
parser.add_argument('-R', '--void_radii', type=str, 
                    nargs='+', help="Path to void radii")
parser.add_argument('--profile', type=str, 
                    nargs='+', default=None, help="Alternatively path to pre-save density profile")
parser.add_argument('--model', type=str, default=None, help="Type of analytic model to plot")
parser.add_argument('--Rmax', type=float, default=4, help="Maximum R limit")
parser.add_argument('--rho_mean', type=float, default=None, help="Manually enter mean density rather than measure from galaxies")
parser.add_argument('--save', type=str, default=None, help="Path to save")
parser.add_argument('--boxsize', type=float, default=2000)
parser.add_argument('--boxcenter', type=float, default=0)
parser.add_argument('--legend', type=str, default=None, 
                    nargs='+', help="Plot legend")
parser.add_argument('--color', type=str, default=None, 
                    nargs='+', help="Plot colours")
parser.add_argument('--ls', type=str, default=None, 
                    nargs='+', help="Plot linestyle")
args = parser.parse_args()

box_shift = args.boxsize/2 - args.boxcenter

if args.save is None:
    save_files = False
    save_plot = False
elif args.save.endswith('.npy'):
    args.save = args.save[:-4]
    save_files = True
    save_plot = False
elif args.save.endswith('.png') or args.save.endswith('.pdf'):
    append = args.save[-4:]
    args.save = args.save[:-4]
    save_files = False
    save_plot = True
else:
    args.save = args.save[:-4]
    append = '.png'
    save_files = True
    save_plot = True

def load(fn):
    if fn.endswith('.npy'):
        pos = np.load(fn)
    elif fn.endswith('.fits'):
        with fits.open(fn) as f:
            pos = np.array([f[1].data['X'], f[1].data['Y'], f[1].data['Z']]).T
    else:
        raise Exception(f"{fn} format not recognised.")
    return pos

def compute_profile(gal_pos, void_pos, void_radii, save=None):
    gal_pos = load(gal_pos)
    void_pos = load(void_pos)
    void_radii = load(void_radii)

    tree = cKDTree(gal_pos + box_shift, compact_nodes=False, 
                   balanced_tree=False, boxsize=args.boxsize + 0.0001)

    rr = np.arange(0, args.Rmax + 0.1, 0.1)
    N = np.zeros((rr.size, void_radii.size))
    V = np.zeros((rr.size, void_radii.size))
    for (i,r) in enumerate(rr[1:]):
        N[i+1] = tree.query_ball_point(void_pos + box_shift, r * void_radii,  
                                       workers=-1, return_length=True)
        V[i+1] = 4 * np.pi * (r * void_radii)**3 / 3

    N = N[1:] - N[:-1]
    V = V[1:] - V[:-1]
    r_mid = (rr[1:] + rr[:-1]) / 2  # take midpoint 
    rho = (N / V).mean(axis=1)
    rho_mean = gal_pos.shape[0] / args.boxsize**3 if args.rho_mean is None else args.rho_mean

    # ensure intercept at (0,0)
    r_mid = np.insert(r_mid, 0, 0.) 
    rho = np.insert(rho, 0, 0.)

    if save is not None: np.save(save, np.array([rr, rho / rho_mean]))

    return rr, rho, rho_mean

def plot_profile(rr, rho, rho_mean, save=None, ax=None, **kwargs):
    if ax is None: fig, ax = plt.subplots(1)
    ax.plot(rr, rho / rho_mean, **kwargs)
    ax.set_ylabel(r"$\rho(r) / \bar{\rho}$", fontsize=14)
    ax.set_xlabel("$r/R_\mathrm{void}$", fontsize=14)
    ax.grid()
    plt.tight_layout()
    if save is not None: plt.savefig(save)

fig, ax = plt.subplots(1)
for i in range(len(args.void_pos)):
    gal_pos = args.gal_pos[0] if len(args.gal_pos) == 1 else args.gal_pos[i]

    profile = compute_profile(gal_pos, args.void_pos[i], args.void_radii[i], 
                              save=f"{args.save}_{i}.npy" if save_files else None)

    plot_kwargs = {'lw': 2, 
                   'color': None if args.color is None else args.color[i], 
                   'ls': None if args.ls is None else args.ls[i],
                   'label': None if args.legend is None else args.legend[i]}
    plot_profile(*profile, ax=ax, **plot_kwargs)

if args.profile is not None:
    for (j,fn) in enumerate(args.profile):
        ax.plot(*np.load(fn), lw=2, 
                color=None if args.color is None else args.color[i+j+1],
                ls=None if args.ls is None else args.ls[i+j+1],
                label=None if args.legend is None else args.legend[i+j+1])

if args.model is not None: 
    ax.plot(profile[0], density_profile(profile[0], vf=args.model),
            lw=2, color=None if args.color is None else args.color[-1], 
            ls=None if args.ls is None else args.ls[-1],
            label=None if args.legend is None else args.legend[-1])

if args.legend is not None: plt.legend(loc='lower right')
if save_plot: plt.savefig(f"{args.save}{append}")
plt.show()

