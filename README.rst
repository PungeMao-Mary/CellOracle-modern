CellOracle
==========

|GitHub Workflow Status| |PyPI| |PyPI - Python Version| |PyPI - Wheel|
|Downloads| |Docker Pulls|

CellOracle is a python library for in silico gene perturbation analyses
using single-cell omics data and Gene Regulatory Network models.

For more information, please read our paper: `Dissecting cell identity
via network inference and in silico gene
perturbation <https://www.nature.com/articles/s41586-022-05688-9>`__.

Modernized build (v0.22.2): macOS / NumPy 2 / pandas 3 / Apple GPU
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This build modernizes CellOracle for the current scientific Python stack.
See `MODERNIZATION.md <MODERNIZATION.md>`__ for implementation details and
`USAGE.md <USAGE.md>`__ for installation and copy-paste usage examples.

**Highlights**

- Runs on NumPy 2.x and pandas 2.x / 3.x with Python 3.9 - 3.13.
- No compiled ``velocyto`` dependency: the velocity / correlation kernels are
  bundled in ``celloracle.trajectory._velocity_compat``.
- Optional **Apple Metal GPU acceleration** (through MLX) for GRN signal
  propagation (``Oracle.simulate_shift``) and transition-probability
  correlations (``Oracle.estimate_transition_prob``); dense linear algebra runs
  on Apple's **Accelerate**/vecLib BLAS on macOS.
- Optional dependencies (``genomepy``/``gimmemotifs`` for motif analysis,
  ``IPython``/``ipywidgets`` for notebooks) are now truly optional.

**Bug fixes**

- ``Oracle.plot_simulation_flow_on_grid()`` and
  ``plot_simulation_flow_random_on_grid()`` no longer raise
  ``UnboundLocalError: local variable 'gridpoints_coordinates' referenced
  before assignment``.  The plotting wrapper tested the wrong attribute
  (``mass_filter`` instead of ``flow_grid``) when resolving the grid
  coordinates, and the mass-filter mask was also left unbound when
  ``calculate_mass_filter()`` had not been called.  Both are fixed: the grid is
  resolved from ``gridpoints_coordinates`` or ``flow_grid``, and a missing mass
  filter now defaults to "keep every grid point".  Plotting a plain ``Oracle``
  after ``calculate_p_mass()`` therefore works without calling
  ``calculate_mass_filter()`` first.  The same unguarded pattern in
  ``plot_pseudotime_on_grid`` / ``plot_reference_flow_on_grid`` /
  ``plot_background_on_grid`` was fixed as well.
- ``Oracle.get_markov_simulation_cell_transition_table()`` returned attributes
  that were never set.
- python-igraph 1.0 compatibility for ``Links.get_network_score()``.

Documentation, Codes, and Tutorials
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CellOracle documentation is available through the link below.

`Web
documentation <https://morris-lab.github.io/CellOracle.documentation/>`__

Questions and errors
~~~~~~~~~~~~~~~~~~~~

If you have a question, error, bug, or problem, please use the `Github
issue page <https://github.com/morris-lab/CellOracle/issues>`__.

Supported Species and reference genomes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Human: [‘hg38’, ‘hg19’]
-  Mouse: [‘mm39’, ‘mm10’, ‘mm9’]
-  S.cerevisiae: [“sacCer2”, “sacCer3”]
-  Zebrafish: [“danRer7”, “danRer10”, “danRer11”]
-  Xenopus tropicalis: [“xenTro2”, “xenTro3”]
-  Xenopus laevis: [“Xenopus_laevis_v10.1”]
-  Rat: [“rn4”, “rn5”, “rn6”]
-  Drosophila: [“dm3”, “dm6”]
-  C.elegans: [“ce6”, “ce10”]
-  Arabidopsis: [“TAIR10”]
-  Chicken: [“galGal4”, “galGal5”, “galGal6”]
-  Guinea Pig: [“Cavpor3.0”]
-  Pig: [“Sscrofa11.1”]

Changelog
~~~~~~~~~

This fork: **v0.22.2** - modernized stack (NumPy 2 / pandas 2-3, Python
3.9-3.13), built-in velocity kernels (no ``velocyto``), optional Apple Metal
GPU acceleration, plus the ``plot_simulation_flow_on_grid``
``UnboundLocalError`` fix described above.  See `MODERNIZATION.md
<MODERNIZATION.md>`__.

Upstream: please go to `this
page <https://morris-lab.github.io/CellOracle.documentation/changelog/index.html>`__.

.. |GitHub Workflow Status| image:: https://img.shields.io/github/actions/workflow/status/morris-lab/CellOracle/build_check.yml?branch=master
   :target: https://github.com/morris-lab/CellOracle/actions/workflows/build_check.yml
.. |PyPI| image:: https://img.shields.io/pypi/v/celloracle?color=blue
   :target: https://pypi.org/project/celloracle/
.. |PyPI - Python Version| image:: https://img.shields.io/pypi/pyversions/celloracle
   :target: https://pypi.org/project/celloracle/
.. |PyPI - Wheel| image:: https://img.shields.io/pypi/wheel/celloracle
   :target: https://pypi.org/project/celloracle/
.. |Downloads| image:: https://static.pepy.tech/personalized-badge/celloracle?period=total&units=international_system&left_color=grey&right_color=orange&left_text=Downloads
   :target: https://pepy.tech/project/celloracle
.. |Docker Pulls| image:: https://img.shields.io/docker/pulls/kenjikamimoto126/celloracle_ubuntu?color=red
   :target: https://hub.docker.com/r/kenjikamimoto126/celloracle_ubuntu
