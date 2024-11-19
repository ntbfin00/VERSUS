import numpy as np
import argparse
import logging
from VERSUS.meshbuilder import DensityMesh
from VERSUS.sphericalvoids import SphericalVoids

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Run spherical void-finding on simulated or survey data")
    parser.add_argument('--data', help="Array or path to data positions")
    parser.add_argument('--random', help="Array or path to random positions")
    parser.add_argument('--data_weights', help="Array of weights for data positions")
    parser.add_argument('--random_weights', help="Array of weights for random positions")
    parser.add_argument('--columns', help="Data column headers to read positions (XYZ/rdz)")
    parser.add_argument('--mesh', default=None, 
                        help="Array or path to density mesh. If not None and data also provided, save to path provided (True for default path).")
    parser.add_argument('--mesh_args', required=False, nargs='+', help="Provide cellsize, boxsize, boxcenter and box-like.")
    parser.add_argument('--radii', type=float, default=0., nargs='+', help="List of void radii to search for")#, required=True)
    parser.add_argument('--void_delta', type=float, default=-0.8, help="Maximum overdensity to be classified as void")
    parser.add_argument('--void_overlap', type=float, default=0., help="Volume fraction of allowed void overlap")
    parser.add_argument('--threads', type=int, default=0, 
                        help="Number of threads used for multi-threaded processes. Defaults to maximum available.")
    parser.add_argument('--save_dir', type=str, default=None, help="Path to save output (void positions & radii). Defaults to 'output/'.")

    args = parser.parse_args()
    if args.data is None and args.mesh is None:
        parser.error("Either --data or --mesh must be provided.")

    if args.mesh_args is not None: args.mesh_args = dict(zip(['cellsize','boxsize','boxcenter','box_like'], args.mesh_args))

    return args

def main():
    args = parse_args()

    VF = SphericalVoids(data_positions=args.data, data_weights=args.data_weights,                                                                                           random_positions=args.random, random_weights=args.random_weights, data_cols=args.columns,
                        delta_mesh=args.mesh, mesh_args=args.mesh_args, save_mesh=True if args.mesh == 'True' else args.mesh)

    # VF.run_voidfinding(np.array(args.radii, dtype=np.float32), void_delta=args.void_delta, void_overlap=args.void_overlap, threads=args.threads)
    VF.run_voidfinding(args.radii, void_delta=args.void_delta, void_overlap=args.void_overlap, threads=args.threads)

    if args.save_dir is None:
        from pathlib import Path
        path = "output"
        Path(path).mkdir(parents=True, exist_ok=True)
    else:
        path = args.save_dir
    import os
    np.save(os.path.join(path, "void_positions.npy"), VF.void_position)
    np.save(os.path.join(path, "void_radii.npy"), VF.void_radius)
    np.save(os.path.join(path, "void_vsf.npy"), VF.void_vsf)


if __name__ == "__main__":
    main()
