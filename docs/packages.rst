.. _packages:

Packages
========

`fv3fit <https://vulcanclimatemodeling.com/docs/fv3fit>`_ is a library for machine learning workflows.

`loaders <https://vulcanclimatemodeling.com/docs/loaders>`_ provides unified APIs for accessing model output datasets.

`fv3viz <https://vulcanclimatemodeling.com/docs/fv3viz>`_ contains visualization tools.

`vcm <https://vulcanclimatemodeling.com/docs/vcm>`_ is a collection of various routines.

:doc:`synth <readme_links/synth_readme>` is a package which allows you to define data schemas and create synthetic datasets for testing.

:doc:`report <readme_links/report_readme>` handles the generation of workflow reports.

:doc:`fv3kube <readme_links/fv3kube_readme>` contains utilities to handle submitting and monitoring fv3gfs-python jobs on kubernetes.


.. rubric:: VCM packages in other repositories:

fv3gfs-python_ (`docs <https://vulcanclimatemodeling.github.io/fv3gfs-python/f12n7eq5xkoibbqp/index.html>`_) is a Python wrapper for the FV3GFS Fortran model.

fv3gfs-fortran_ is our fork of the FV3GFS fortran model, which we run using the fv3gfs-python wrapper.

fv3gfs-util_ (`docs <https://fv3gfs-util.readthedocs.io/en/latest/>`_) is a library of general-purpose Python code to use in a model script.

fv3config_ (`docs <https://fv3config.readthedocs.io/en/latest/>`_) provides routines to configure and write a FV3GFS run directory using a yaml configuration file and data stored on the cloud.

.. _fv3gfs-python: https://github.com/VulcanClimateModeling/fv3gfs-python
.. _fv3gfs-fortran: https://github.com/VulcanClimateModeling/fv3gfs-fortran
.. _fv3gfs-util: https://github.com/VulcanClimateModeling/fv3gfs-util
.. _fv3config: https://github.com/VulcanClimateModeling/fv3config

.. toctree::
    :caption: List of packages:
    :maxdepth: 1
    :glob:
    
    readme_links/fv3fit_readme
    readme_links/loaders_readme
    readme_links/fv3viz_readme
    readme_links/vcm_readme
    readme_links/synth_readme
    readme_links/report_readme
    readme_links/fv3kube_readme
