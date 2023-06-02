# VERSUS
### Void Extraction in Real-space from SUrveys and Simulations
Void-finding with optional real-space reconstruction for use with both simulated data and survey data. The following void-types will be implemented:
- Spherical based voids (based on [Pylians3](https://github.com/franciscovillaescusa/Pylians3))
- Voxel based voids
- Zobov based voids

```example_config.yaml``` provides all the config options for running from the command line.

## Requirements

Minimal requirements:
- Pyrecon
- Astropy
- PyCall.jl
- FFTW.jl
- Parameters.jl
- Printf.jl

To run from command line:
- ArgParse.jl
- YAML.jl
- FITSIO.jl
- DelimitedFiles.jl

## Usage
Run reconstruction on galaxy positions:
```
cosmo = Cosmology(; **kwargs)  # LambdaCDM cosmology

# if conversion from sky to cartesian positions required
<xyz positions> = to_cartesian(cosmo, <rdz positions>)

cat = GalaxyCatalogue(<xyz galaxy positions>, [<galaxy weights>], [<xyz random positions>], [<random weights>])
recon_par = MeshParams(; **kwargs)

cat_recon = reconstruction(cosmo, cat, recon_par)
```

Run void finding:
```
cat = GalaxyCatalogue(<xyz galaxy positions>, [<galaxy weights>], [<xyz random positions>], [<random weights>])
mesh_par = MeshParams(; **kwargs)

# e.g. Spherical voidfinder
vf_par = SphericalVoidParams(; **kwargs)
vf = SphericalVoids.voidfinder(cat, mesh_par, vf_par)

# or to supply a precomputed mesh use...
vf = SphericalVoids.voidfinder(<3D delta array>, <side length>, <1D mesh centre array>, vf_par)

# Output:
vf.type       # void type
vf.positions  # void positions
vf.radii      # void radii
vf.vsf        # void size function (r, vsf)
```

Run reconstruction and void finding direct from command line:
```
julia [-t <n_threads>] --project=<path/to/directory> VERSUS.jl --config <yaml file> --data <fits file> [--randoms <fits file>]
```
To supply a pre-computed mesh (instead of galaxy positions) from command line, set ```input["build_mesh"] = false``` in config file.
