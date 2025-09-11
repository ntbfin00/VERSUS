import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.spatial import cKDTree
import argparse
from distributions import density_profile
import matplotlib as mpl

parser = argparse.ArgumentParser(description='Compute void density profile')
parser.add_argument('-g', '--gal_pos', type=str, 
                    nargs='+', help="Path to galaxy positions")
parser.add_argument('-v', '--void_pos', type=str, 
                    nargs='+', help="Path to void positions")
parser.add_argument('-R', '--void_radii', type=str, 
                    nargs='+', help="Path to void radii")
parser.add_argument('--profiles', type=str, nargs='+', 
                    default=None, help="Alternatively path to pre-save density profile")
parser.add_argument('--models', type=str, nargs='+', 
                    default=None, help="Type of analytic model to plot")
parser.add_argument('--Rrange', type=float, nargs=2, default=None, help="R range to cut void data")
parser.add_argument('--Rmax', type=float, default=4, help="Maximum R limit to plot")
parser.add_argument('--step', type=float, default=0.05, help="Stepsize")
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

colors =  mpl.color_sequences['Paired']
box_shift = args.boxsize/2 - args.boxcenter

def load(fn, radius=False):
    if fn.endswith('.npy'):
        pos = np.load(fn)
    elif fn.endswith('.fits') and not radius:
        with fits.open(fn) as f:
            pos = np.array([f[1].data['X'], f[1].data['Y'], f[1].data['Z']]).T
    # read revolver output
    elif fn.endswith('.txt'):
        with open(fn) as f:
            header = f.readlines()[1].split(' ')[1:]
        pos_indx = 1
        rad_indx = 4
        if not header[pos_indx].startswith('XYZ'):
            raise Exception(f"Incorrect column ({header[pos_indx]}) read for void positions")
        if not header[rad_indx-2].startswith('R_eff'):
            raise Exception(f"Incorrect column ({header[rad_indx-2]}) read for void radius")
        f = np.loadtxt(fn)
        if radius:
            pos = f[:, rad_indx]
        else:
            pos = f[:, pos_indx:pos_indx+3]
    else:
        raise Exception(f"{fn} format not recognised.")

    return pos

def compute_profile(gal_pos, void_pos, void_radii, Rrange=None, Rmax=4, step=0.1, save=None):
    gal_pos = load(gal_pos)
    void_pos = load(void_pos)
    void_radii = load(void_radii, radius=True)

    if Rrange is not None:
        cut = (void_radii >= Rrange[0]) & (void_radii <= Rrange[1])
        void_radii = void_radii[cut]
        void_pos = void_pos[cut]

    print(f"Computing profile from {void_radii.size} voids")

    tree = cKDTree(gal_pos + box_shift, compact_nodes=False, 
                   balanced_tree=False, boxsize=args.boxsize + 0.0001)

    rr = np.arange(0, Rmax + step, step)
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
i = j = -1
if args.void_pos is not None:
    for i in range(len(args.void_pos)):
        append = f"_{i}" if len(args.void_pos) > 1 else ""
        gal_pos = args.gal_pos[i] if len(args.gal_pos) > 1 else args.gal_pos[0]

        profile = compute_profile(gal_pos, args.void_pos[i], args.void_radii[i], 
                                  Rrange=args.Rrange, Rmax=args.Rmax, step=args.step,
                                  save=f"{args.save}{append}.npy" if save_files else None)

        plot_kwargs = {'lw': 2, 
                       'color': colors[i+1] if args.color is None else args.color[i], 
                       'ls': None if args.ls is None else args.ls[i],
                       'label': None if args.legend is None else args.legend[i]}
        plot_profile(*profile, ax=ax, **plot_kwargs)

if args.profiles is not None:
    for (j,fn) in enumerate(args.profiles):
        ax.plot(*np.load(fn), lw=2, 
                color=colors[j*2+1] if args.color is None else args.color[i+j+1],
                ls=None if args.ls is None else args.ls[i+j+1],
                label=None if args.legend is None else args.legend[i+j+1])

if args.models is not None: 
    rr = np.linspace(0, args.Rmax, 100)
    for (k,model) in enumerate(args.models):
        ax.plot(rr, density_profile(rr, vf=model),
                lw=2, ls='--', color=colors[k*2] if args.color is None else args.color[i+j+k-1])
                # label=None if args.legend is None else args.legend[-1])

ax.set_ylabel(r"$\rho(r) / \bar{\rho}$", fontsize=14)
ax.set_xlabel("$r/R_\mathrm{void}$", fontsize=14)
ax.set_xlim(0, args.Rmax)
ax.set_ylim(0, 1.1)
ax.grid()
plt.tight_layout()

if args.legend is not None: plt.legend(loc='lower right')
if save_plot: plt.savefig(f"{args.save}{append}")
plt.show()

