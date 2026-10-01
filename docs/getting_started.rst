Getting started
===============

Basic usage instructions for VERSUS. 

From the command-line
---------------------

VERSUS void-finding can easily be run from the command line directly: 

.. code-block:: console

   python main.py --data <filename> [--random <filename>] [--radii <list>] [--void_delta <float>]

* ``--data`` takes the path to a numpy or FITS file of (N,3) data positions.
* ``--random`` takes the path to the numpy or FITS file of (N,3) random positions (used for specifying non-trivial survey geometries).
* ``--radii`` takes the list of input radii bins in which to detect voids.
* ``--void_delta`` takes the maximum overdensity threshold to be classified as void. 

.. note::
   The random catalogue is not required for simulation boxes. Alternatively, providing the optional ``random`` argument will run void-finding in survey (not simulation box) mode. 

The resulting void catalogues will be saved to file (a specific filepath can be specified using the ``save_fn`` argument). For more details on the full list of accepted command-line arguments, run ``python main.py --help``.


From a Python script
--------------------

VERSUS can also be called modularly from a Python script, giving you full control over the void-finding settings:

.. code-block:: python

   import numpy as np
   from VERSUS import SphericalVoids, setup_logging

   # initialise logger
   setup_logging()

   # load position catalogues
   data = np.load("data.npy")
   randoms = np.load("randoms.npy")  # if survey data
   
   # instantiate void-finder
   vf = SphericalVoids(data_positions=data,
                       random_positions=randoms,
                       cellsize=4)
   
   # find voids
   radii = np.arange(25, 62, 2)
   vf.run_voidfinding(radii,
                      void_delta=-0.8,
                      void_overlap=True,
                      void_merge=0.9)
   
.. note::
   The arguments ``void_overlap`` and ``void_merge`` dictate the levels of allowed overlap and merging of void candidate spheres, respectively. Modifying their values alters the fraction of small voids that are consolidated into larger structures.

After running the void-finding step, the details of the void catalogue are specified by the following ``SphericalVoids`` class attributes:

* ``position`` holds the void centre positions.
* ``radius`` holds the void radii.
* ``counts`` holds the void number counts in each input radius bin.
* ``id`` holds the ID numbers for each void.
* ``cell_membership`` holds a 3D array matching the mesh dimensions, where each cell contains the ID of its assigned void. This is useful for plotting void shapes in 3D.
* ``size_function`` holds the derived void size function including bin centres, values, and Poissonian errorbars.

.. note::
   Saving FFT wisdom by setting ``use_wisdom=True`` can offer serious performance enhancements for serial void-finding runs in which the density mesh settings are held fixed (i.e. instantiating the ``SphericalVoids`` class only once but changing the ``run_voidfinding`` settings).


Making plots
------------

VERSUS offers some inbuilt plotting routines that can be used after running the void-finding step.

* ``SphericalVoids.plot_slice`` will plot a 2D slice through the simulation/survey with the void positions and radii marked.
* ``SphericalVoids.plot_size_function`` will plot the void size function with Poissonian errorbars.


Peak/cluster finding
---------------------------

So far we have only talked about finding voids: low-density minima of the large-scale structure. However, VERSUS can also be easily configured to detect high-density peaks known as clusters.

To do this, simply set ``void_delta`` to any value greater than 0 (zero refers to the mean density). For example, if you want to detect voids with :math:`\delta_v < -0.9`, set ``void_delta=-0.9``. However, if you want to detect clusters with :math:`\delta_v > 2.1`, set ``void_delta=2.1``.



