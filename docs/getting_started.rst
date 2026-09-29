Getting started
===============

Basic usage instructions for VERSUS. 

.. code-block:: python

   import numpy as np
   from VERSUS import SphericalVoids

   # load position catalogues
   data = np.load("data.npy")
   randoms = np.load("randoms.npy")  # if survey data
   
   # instantiate void-finder
   vf = SphericalVoids(data_positions=data,
                       random_positions=randoms,
                       cellsize=4)
   
   # find voids
   vf.run_voidfinding(np.arange(25, 62, 2),  # radii
                      void_delta=-0.8,
                      void_overlap=True,
                      void_merge=0.9)
   
   # output
   vf.position         # void positions
   vf.radius           # void radii
   vf.size_function    # VSF
   vf.cell_membership  # void membership   # Add usage snippet here

Peak (high-density) finding
---------------------------
