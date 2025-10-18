import numpy as np
import argparse
from pathlib import Path
import logging
from VERSUS import setup_logging, SphericalVoids

setup_logging(level=logging.INFO)
logger = logging.getLogger("VERSUS")

def parse_args():
    parser = argparse.ArgumentParser(description="Run spherical void-finding on simulated or survey data")
    parser.add_argument('--data', help="Array or path to data positions")
    parser.add_argument('--random', help="Array or path to random positions")
    parser.add_argument('--data_weights', help="Array of weights for data positions")
    parser.add_argument('--random_weights', help="Array of weights for random positions")
    parser.add_argument('--columns', nargs='+', help="Data column headers to read positions (XYZ/rdz)")
    parser.add_argument('--mesh', default=None, 
                        help="Array or path to density mesh. If not None and data also provided, save to path provided (True for default path).")
    parser.add_argument('--cellsize', type=float, default=4., help="Size of mesh cells.")
    parser.add_argument('--reconstruct', required=False, type=str, default=None, 
                        help="Type of reconstruction ('disp', 'rsd' or 'disp+rsd'), growth rate and bias. Defaults to no reconstruction. Must additionally provide 'f' and 'bias' in recon_args.")
    parser.add_argument('--recon_args', required=False, nargs='+', default=None,
                        help="Provide dictionary of reconstruction arguments - 'f','bias','los','smoothing_radius','recon_pad','engine'.")
    parser.add_argument('--mesh_args', required=False, nargs='+', help="Provide dictionary of cellsize, r_sep, boxsize, boxcenter and box-like.")
    parser.add_argument('--radii', default=[0.], nargs='+', help="List of void radii to detect. Can be passed as dictionary of arguments to np.linspace or np.arange, e.g. \"{'start': 20, 'stop': 50, 'num (or step)': 10}\"")
    parser.add_argument('--void_delta', type=float, default=-0.8, help="Maximum overdensity to be classified as void. If value is positive, peaks will be found instead.")
    parser.add_argument('--void_overlap', default=False, help="Boolean or volume fraction of allowed void overlap. True allows overlap up to void centre while False prevents overlap.")
    parser.add_argument('--smoothing', type=float, default=0.45, help="Radius (as fraction of galaxy separation) to initally smooth density field.")
    parser.add_argument('--save_fn', type=str, default=None, help="Path to save output (void positions & radii). Defaults to 'output/'.")
    parser.add_argument('--threads', type=int, default=8, 
                        help="Number of threads used for multi-threaded processes. if set to zero, defaults to number of available CPUs.")
    parser.add_argument('--dryrun', required=False, action='store_true',help="Run script without saving outputs")

    args = parser.parse_args()
    if args.data is None and args.mesh is None:
        parser.error("Either --data or --mesh must be provided.")

    # optionally read radii as dictionary
    if str(args.radii[0]).startswith("{"):
        args.radii = ''.join(args.radii)
        exec("args.radii = " + args.radii)
        if "step" in args.radii.keys():
            args.radii = list(np.arange(**args.radii))
        else:
            args.radii = list(np.linspace(**args.radii))
    # read mesh arguments as dictionary (for case that mesh is provided as array)
    if args.mesh_args is not None: 
        args.mesh_args = ''.join(args.mesh_args)
        exec("args.mesh_args = " + args.mesh_args)
    # read reconstruction arguments as dictionary
    if args.reconstruct is not None: 
        if args.recon_args is None: parser.error("Minimally 'f' and 'bias' must be provided for density field reconstruction")
        args.recon_args = ''.join(args.recon_args)
        exec("args.recon_args = " + args.recon_args)

    if args.void_overlap in [True, 'True', 'true']:
        args.void_overlap = True
    elif args.void_overlap in [False, 'False', 'false']:
        args.void_overlap = False
    else:
        args.void_overlap = float(args.void_overlap)

    return args


def filename(save_fn):
    append = 'void_{}.npy'
    path = Path(save_fn + append)
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(path.stem[:-7])>0:
        return save_fn + '_' + append
    else:
        return save_fn + append


def main():
    args = parse_args()

    # set save_mesh argument
    if args.dryrun:
        save_mesh = False
    elif args.mesh == 'True':
        save_mesh = True
    else:
        save_mesh = args.mesh

    # initialise void finder with command-line arguments
    VF = SphericalVoids(data_positions=args.data, data_weights=args.data_weights,                                                                                           random_positions=args.random, random_weights=args.random_weights, data_cols=args.columns,
                        delta_mesh=args.mesh, mesh_args=args.mesh_args, save_mesh=save_mesh,
                        cellsize=args.cellsize, reconstruct=args.reconstruct, recon_args=args.recon_args)

    # run void finding
    VF.run_voidfinding(args.radii, void_delta=args.void_delta, void_overlap=args.void_overlap, 
                       init_sm_frac=args.smoothing, threads=args.threads)

    # save void output to file
    fn = "output/" if args.save_fn is None else args.save_fn
    logger.info(f"Saving output to {fn}*")
    if not args.dryrun:
        fn = filename(fn)
        np.save(fn.format("positions"), VF.void_position)
        np.save(fn.format("radii"), VF.void_radius)
        np.save(fn.format("vsf"), VF.void_vsf)
    else:
        logger.info(f'Output: {dict(zip(VF.radii, VF.void_count))}')


if __name__ == "__main__":
    main()
