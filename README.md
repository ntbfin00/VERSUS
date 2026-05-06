<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/logo_w.png?raw=true" height=80/>
  <source media="(prefers-color-scheme: light)" srcset="assets/logo_b.png?raw=true" height=80/>
  <img alt="VERSUS logo">
</picture>

# Void Extraction in Real-space of Spherical UnderdensitieS
Spherical underdensity void-finding with optional real-space reconstruction for use with both simulated and survey data. Adapted from the void-finding algorithm in the [Pylians3](https://github.com/franciscovillaescusa/Pylians3) library.

<div style="text-align: center;">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/flowchart_b.png?raw=true">
    <source media="(prefers-color-scheme: light)" srcset="assets/flowchart_w.png?raw=true">
    <img src="assets/flowchart_w.png?raw=true"
         alt="VERSUS logo"
         style="width: 100%; max-width: 100%; height: auto;">
  </picture>
</div>

## Installation
To pip install:
```
pip install [-e] git+https://github.com/ntbfin00/VERSUS.git
```
The ```-e``` flag is optional and will install an editable version of the package.

## Usage
Command line:
```
python main.py --data <fits file> [--random <fits file>] [--data_weights <array>] [--random_weights <array>] [--columns <list>] [--radii <list>] [--threads <int>]
```
If randoms are supplied from command line, VERSUS will run in survey (not simulation box) mode. See ```python main.py --help``` for full list of inputs.

Module:
```
from VERSUS import SphericalVoids

VF = SphericalVoids(data_positions=<path or 3D array>, data_weights=<1D array>,                                                                          random_positions=<path or 3D array>, random_weights=<1D array>,
                    data_cols=<list>, save_mesh=<bool or path>)

VF.run_voidfinding(radii, **void_finder_kwargs)

# or to supply a precomputed 3D mesh use...
vf = SphericalVoids(delta_mesh=<path/to/mesh>, ...)

# Output:
vf.input_radii    # input smoothing radii for void-finder
vf.position  	  # void positions
vf.radius    	  # void radii
vf.counts     	  # number of voids of a given radius
vf.size_function  # void size function (r, vsf, error)
```

If making repeated measurements on a fixed mesh, enable the FFT wisdom to be saved using the ```use_wisdom=True``` flag when instantiating the ```SphericalVoids``` class. The wisdom text files will be stored in the ```wisdom``` directory.

If positive values for ```void_delta``` (default ```void_delta=-0.8```) in ```VF.run_voidfinding``` are entered, the algorithm will instead search for density peaks with enclosed density ```delta > void_delta```.

## Citation
If you use this code in a scientific publication, please cite:

```
@ARTICLE{Findlay2026,
         title={VERSUS: An excursion-set-inspired void-finder for the Stage-IV era}, 
         author={Nathan Findlay and Seshadri Nadathur},
         year={2026},
         eprint={2605.03779},
         archivePrefix={arXiv},
         primaryClass={astro-ph.CO},
         url={https://arxiv.org/abs/2605.03779}, 
}
```
