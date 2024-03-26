=======
PRIMAP2
=======

.. image:: https://img.shields.io/pypi/v/primap2.svg
        :target: https://pypi.python.org/pypi/primap2
        :alt: PyPI status

.. image:: https://readthedocs.org/projects/primap2/badge/?version=main
        :target: https://primap2.readthedocs.io/en/main/?badge=main
        :alt: Documentation Status

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.4535902.svg
        :target: https://doi.org/10.5281/zenodo.4535902
        :alt: Zenodo release

PRIMAP2 is the next generation of the PRIMAP climate policy analysis suite.
PRIMAP2 is free software, you are welcome to use it in your own research.
The documentation can be found at https://primap2.readthedocs.io.

Structure
---------

PRIMAP2 is:
 * A flexible and powerful data format built on `xarray <https://xarray.pydata.org>`_.
 * A collection of functions for common tasks when wrangling climate policy
   data, like aggregation and interpolation.
 * A format for data packages built on `datalad <https://www.datalad.org>`_, providing
   metadata extraction and search on a collection of data packages.

Status
------

PRIMAP2 is in active development, and not everything promised above is built
yet.

License
-------
Copyright 2020-2022, Potsdam-Institut für Klimafolgenforschung e.V.

Copyright 2023-2024, Climate Resource Pty Ltd

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this
file except in compliance with the License. You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under
the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied. See the License for the specific language governing
permissions and limitations under the License.

PRIMAP2 incorporates parts of xarray and pint_xarray, which are available under the
Apache License, Version 2.0 as well. The full text of the xarray copyright statement is
included in the licenses directory.

Citation
--------
If you use this library and want to cite it, please cite it as:

Mika Pflüger and Johannes Gütschow. (2023-12-12).
pik-primap/primap2: PRIMAP2 Version 0.10.0.
Zenodo. https://doi.org/10.5281/zenodo.10361847
