# JulVF
Void-finding algorithms written in Julia. The following void-types will be implemented:
- Zobov based voids
- Voxel based voids
- Spherical based voids

```voidparameters.jl``` is used to set the void-finding options. These are automatically set to the default values but can be changed in ```main.jl```.

```<void_type>voids.jl``` contain the void-finding routines for voids of type <void_type>.

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
- FITSIO.jl
- HDF5.jl

## Usage
Run mesh building:
```
par = VoidParams(; **kwargs)
cat = GalaxyCatalogue(<xyz galaxy positions>, <galaxy weights>, [<xyz random positions>], [<random weights>])
delta = create_mesh(par, cat, <smoothing radius>)
```

Run void finding:
```
Radii = <void radii>
par = VoidParams(; **kwargs)
vf = SphericalVoids.run_voidfinder(<delta mesh>, Radii, par)

vf.type  # void type
vf.positions  # void positions
vf.radii  # void radii
vf.vsf  # void size function
vf.rbins  # corresponding radius bins
```

Run mesh building and void finding direct from command line:

```
julia [-t <n_threads>] main.jl --data <fits, hdf5 file> [--randoms <fits, hdf5 file>] [--config <yaml file>]
```
