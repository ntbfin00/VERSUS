# VERSUS
### Void Extraction in Real-space from SUrveys and Simulations
Void-finding with optional real-space reconstruction for use with both simulated data and survey data. The following void-types will be implemented:
- Spherical based voids (based on [Pylians3](https://github.com/franciscovillaescusa/Pylians3))
- Voxel based voids
- Zobov based voids

```<void_type>voids.jl``` are the void-finding algorithms for voids of type <void_type>. These take a density mesh as input.

```voidparameters.jl``` is used to set the void-finding options.

```meshbuilder.jl``` utilises [Pyrecon](https://github.com/cosmodesi/pyrecon) to create a density mesh from galaxy positions.

```versus.jl``` script can be run from the command line.

```example_config.yaml``` provides all the config options for running from the command line.

## Requirements

Minimal requirements:
- Parameters.jl
- FFTW.jl
- Printf.jl

To use in-built Pyrecon mesh builder (optional):
- PyCall.jl

To run from command line (optional):
- PyCall.jl
- ArgParse.jl
- YAML.jl
- FITSIO.jl
- HDF5.jl
- DelimitedFiles.jl

## Usage
Run mesh building:
```
mesh_settings = MeshParams(; **kwargs)
input_settings = InputParams(; **kwargs)
cat = GalaxyCatalogue(<xyz galaxy positions>, [<galaxy weights>], [<xyz random positions>], [<random weights>])
delta = create_mesh(cat, mesh_settings, input_settings)
```

Run void finding:
```
input_settings = InputParams(; **kwargs)
par = VoidParams(; **kwargs)
vf = SphericalVoids.run_voidfinder(<delta mesh>, input_settings, par)

vf.type       # void type
vf.positions  # void positions
vf.radii      # void radii
vf.vsf        # void size function (r, vsf)
```

Run mesh building and void finding direct from command line:

```
/path/to/julia [-t <n_threads>] versus.jl --config <yaml file> --data <fits, hdf5 file> [--randoms <fits, hdf5 file>]
```
To supply a pre-computed mesh (instead of galaxy positions), set ```input["build_mesh"] = false```.
