# VERSUS
### Void Extraction in Real-space from SUrveys and Simulations
Void-finding with optional real-space reconstruction for use with both simulated data and survey data. The following void-types will be implemented:
- Spherical based voids (based on [Pylians3](https://github.com/franciscovillaescusa/Pylians3))
- Voxel based voids
- Zobov based voids

## Requirements

- pyrecon
- astropy
- pyfftw
- argparse

## Installation
Clone the repository:
```
git clone https://github.com/ntbfin00/VERSUS.git
```
Setup the package:
```
python setup.py build_ext --inplace
```


## Usage
Command line:
```
python main.py --data <fits file> [--random <fits file>] [--data_weights <array>] [--random_weights <array>] [--columns <list>] [--radii <list>] [--threads <int>]
```
If randoms are supplied from command line, VERSUS will run in survey (not simulation box) mode. See ```python main.py --help``` for full list of inputs.

Module:
```
from VERSUS.sphericalvoids import SphericalVoids

VF = SphericalVoids(data_positions=<path or 3D array>, data_weights=<1D array>,                                                                          random_positions=<path or 3D array>, random_weights=<1D array>,
                    data_cols=<list>, save_mesh=<bool or path>)

VF.run_voidfinding(radii, **void_finder_kwargs)

# or to supply a precomputed mesh use...
vf = SphericalVoids(delta_mesh=<path/to/mesh>, ...)

# Output:
vf.void_position  # void positions
vf.void_radius    # void radii
vf.vsf            # void size function (r, vsf, error)
```
