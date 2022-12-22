# JulVF
Void-finding codes written in Julia. The following void-types will be implemented:
- Zobov based voids
- Voxel based voids
- Spherical based voids

```voidparameters.jl``` is used to set the void-finding options. These are automatically set to the default values but can be changed in ```main.jl```.

```<void_type>voids.jl``` contain the void-finding routines for voids of type <void_type>.
