# VERSUS
### Void Extraction in Real-space from SUrveys and Simulations
Void-finding with optional real-space reconstruction for use with both simulated data and survey data. The following void-types will be implemented:
- Spherical based voids (based on [Pylians3](https://github.com/franciscovillaescusa/Pylians3))
- Voxel based voids
- Zobov based voids

```example_config.yaml``` provides all the config options for running from the command line.

## Requirements

Minimal requirements:
- Parameters.jl
- FFTW.jl
- PyCall.jl
- Printf.jl

To run from command line:
- PyCall.jl
- ArgParse.jl
- YAML.jl
- FITSIO.jl
- HDF5.jl
- DelimitedFiles.jl

## Usage
Run reconstruction on galaxy positions:
```
cat = GalaxyCatalogue(<xyz galaxy positions>, [<galaxy weights>], [<xyz random positions>], [<random weights>])
mesh_settings = MeshParams(; **kwargs)
input_settings = InputParams(; **kwargs)

cat_recon = reconstruction(cat, mesh_settings, input_settings)
```

Run void finding:
```
cat = GalaxyCatalogue(<xyz galaxy positions>, [<galaxy weights>], [<xyz random positions>], [<random weights>])
mesh_settings = MeshParams(; **kwargs)
input_settings = InputParams(; **kwargs)

# e.g. Spherical voidfinder
par = SphericalVoidParams(; **kwargs)
vf = SphericalVoids.voidfinder(cat, mesh_settings, input_settings, par)

# or to supply a precomputed mesh use...
vf = SphericalVoids.voidfinder(<delta mesh>, input_settings, par)

# Output:
vf.type       # void type
vf.positions  # void positions
vf.radii      # void radii
vf.vsf        # void size function (r, vsf)
```

Run reconstruction and void finding direct from command line:
```
julia [-t <n_threads>] --project=<path/to/directory> VERSUS.jl --config <yaml file> --data <fits, hdf5 file> [--randoms <fits, hdf5 file>]
```
To supply a pre-computed mesh (instead of galaxy positions) from command line, set ```input["build_mesh"] = false```.
