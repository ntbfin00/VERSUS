import numpy as np
import argparse
from VERSUS.meshbuilder import DensityMesh
# from VERSUS.sphericalvoids import SphericalVoids

def parse_args():
    parser = argparse.ArgumentParser(description="Run spherical void-finding on simulated or survey data")
    parser.add_argument('--data', help="Array or path to data positions")
    parser.add_argument('--randoms', help="Array or path to random positions")
    parser.add_argument('--mesh', help="Array or path to density mesh")
    parser.add_argument('--radii', type=list, help="List of void radii to search for")
    parser.add_argument('--void_delta', type=float, help="Maximum overdensity to be classified as void")
    parser.add_argument('--void_overlap', type=float, help="Volume fraction of allowed void overlap")

    return parser.parse_args()

def main():
    args = parse_args()

if __name__ == "__main__":
    main()
