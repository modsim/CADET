.. _contents:

.. image:: _static/cadet_logo.png

|

.. image:: https://img.shields.io/github/release/modsim/cadet.svg
   :target: https://github.com/modsim/CADET/releases

.. image:: https://github.com/modsim/CADET/actions/workflows/ci.yml/badge.svg?branch=master
   :target: https://github.com/modsim/CADET/actions/workflows/ci.yml?query=branch%3Amaster

.. image:: https://anaconda.org/conda-forge/cadet/badges/downloads.svg
   :target: https://anaconda.org/conda-forge/cadet

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.8179015.svg
   :target: https://doi.org/10.5281/zenodo.8179015

|

CADET
=====
CADET is developed at the Institute of Bio- and Geosciences 1 (IBG-1) of Forschungszentrum Jülich (FZJ) under supervision of Dr. Eric von Lieres.
The heart of the CADET software is a fast and accurate solver for a comprehensive model family.
Typical applications include (but are by far not limited to) chromatography, filtration, crystallization, and fermentation.
CADET can handle arbitrary sequences and networks of unit operations, including reactors, tanks, tubes, pumps, valves, detectors, etc.
The resulting models are solved with state-of-the-art mathematical algorithms and scientific computing techniques.

- **Forum:** https://forum.cadet-web.de
- **Source:** https://github.com/modsim/cadet
- **Demo:** http://cadet-web.de
- **Newsletter:** https://cadet-web.de/newsletter/

Features
--------

* Fast and accurate solution of strongly coupled partial differential algebraic equations (PDAE)
* Computation of parameter sensitivities with algorithmic differentiation (AD)
* Shared memory parallelization using Intel TBB
* Python interface
* Support of HDF5 and XML data formats
* Flexible and extensible through modular design
* Works on Windows, Linux, and Mac OS X


Installation
------------
CADET can be installed via conda from the ``conda-forge`` channel.

``conda install -c conda-forge cadet``

This requires a working `conda installation <https://docs.anaconda.com/anaconda/install/index.html>`_.

Optionally, use `mamba <https://github.com/mamba-org/mamba>`_ which uses a faster dependency solver than ``conda``.

``mamba install -c conda-forge cadet``

For more information on how to install and build CADET, see :ref:`here <installation>`.

Ongoing Development
-------------------

We do our best to provide you with a stable API.
However, CADET is actively developed and breaking changes can sometimes be unavoidable.
For non-developers, it is recommended to upgrade from release to release instead of always working with the most recent commit.


Bugs
----

Please report any bugs that you find `here <https://github.com/modsim/cadet/issues>`_. Or, even better, fork the repository on `GitHub <https://github.com/modsim/cadet>`_ and create a pull request (PR) with the fix. 

Donations
---------

`Donations <https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=FCQ2M89558ZAG>`_ for helping to host, maintain, and further develop the CADET project are highly appreciated.


Citing
------

To cite CADET please use the following publication:

* Leweke, S.; von Lieres, E.: `Chromatography Analysis and Design Toolkit (CADET) <http://doi.org/10.1016/j.compchemeng.2018.02.025>`_, Computers and Chemical Engineering 113 (2018), 274–294.
* Püttmann, A.; Schnittert, S.; Leweke, S.; von Lieres, E.: `Utilizing algorithmic differentiation to efficiently compute chromatograms and parameter sensitivities <http://doi.org/10.1016/j.ces.2015.08.050>`_, Chemical Engineering Science, 139 (2016), 152–162.
* Püttmann, A.; Schnittert, S.; Naumann, U.; von Lieres, E.: `Fast and accurate parameter sensitivities for the general rate model of column liquid chromatography <http://doi.org/10.1016/j.compchemeng.2013.04.021>`_, Computers and Chemical Engineering 56,13 (2013), 46-57.
* von Lieres, E.; Andersson, J.: `A fast and accurate solver for the general rate model of column liquid chromatography <http://doi.org/10.1016/j.compchemeng.2010.03.008>`_, Computers and Chemical Engineering 34,8 (2010), 1180–1191.

Acknowledgments
---------------

Please refer to the `list of contributors <https://github.com/modsim/CADET/blob/master/CONTRIBUTORS.md>`_ who helped building and funding this project.


.. toctree::
    :maxdepth: 3
    :hidden:

    getting_started/index
    modelling/index
    simulation/index
    interface/index
    developer_guide/index
    .. examples/index
    CADET-Match <https://cadet.github.io/CADET-Match/master/index.html>
    license
    zbibliography
    Legal notice <https://www.fz-juelich.de/portal/EN/Service/LegalNotice/_node.html>


