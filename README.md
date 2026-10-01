<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/logo_w.png?raw=true" height=80/>
  <source media="(prefers-color-scheme: light)" srcset="assets/logo_b.png?raw=true" height=80/>
  <img alt="VERSUS logo">
</picture>

# Void Extraction in Real-space of Spherical UnderdensitieS
[![Documentation Status](https://img.shields.io/readthedocs/ntbfin00-versus)](https://ntbfin00-versus.readthedocs.io/)

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

## Documentation

The documentation is hosted on Read the Docs: https://ntbfin00-versus.readthedocs.io/. Additionally, an example notebook can be found in ``basic_example.ipynb``.

## Installation
To pip install:
```
pip install [-e] git+https://github.com/ntbfin00/VERSUS.git
```
The ```-e``` flag is optional and will install an editable version of the package.

## Citation
If you use this code in a scientific publication, please cite:

```
@ARTICLE{Findlay2026,
       author = {{Findlay}, Nathan and {Nadathur}, Seshadri},
        title = "{VERSUS: an excursion-set-inspired void-finder for the Stage-IV era}",
      journal = {\mnras},
         year = 2026,
        month = sep,
       volume = {551},
       number = {2},
          eid = {stag1425},
        pages = {stag1425},
          doi = {10.1093/mnras/stag1425},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2026MNRAS.551g1425F},
}
```
