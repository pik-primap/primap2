=========
Changelog
=========

Unreleased
----------
* Add function to convert long format dataframe to interchange format
* Add functions to help reading of different national GHG inventories

0.5.0
-----
* Add read and write functionality for interchange format
* Add csv reading capabilities in pm2io module for wide and long (tidy) CSV files.
* Add ``da.pr.coverage()`` and ``ds.pr.coverage()`` functions to summarize data
  coverage.
* Add aggregation functions ``set`` and ``count`` which use aliasing and can reduce to
  a given set of dimensions, including the entity.
* Update python packaging to use declarative style and modern setuptools.
* Support and test python 3.9 and windows.
* Add dataset attr for storing the terminology used for entities (and thus variable
  names).

0.4.0
-----
* Add the ``da.pr.set()`` and ``ds.pr.set()`` functions for overwriting / infilling /
  extending data.
* Allow for more than one source in a single Dataset / DataArray.
* Support xarray 0.17, and therefore drop support for Python 3.6.

0.3.1
-----
* Re-release 0.3.0 to trigger zenodo.

0.3.0
-----
* Add functions for downscaling and aggregation of gas baskets and categorical baskets.
* Add functions, docs, and tests for basic GWP handling.
* Add Makefile target to generate patched stub files for xarray containing the primap2
  API.
* Add development documentation detailing the structure of the repository and the tools
  used to development of PRIMAP2.
* Add selection and indexing which understands dimension names like ``area`` in addition
  to the full dimension key including the category set like ``area (ISO3)``. Works with
  ``ds.pr[key]`` and ``ds.pr.loc[selection]`` as well as ``da.pr.loc[selection]``.
* Add usage documentation for all currently included functionality.
* Access metadata easily via properties like ``ds.pr.references``.

0.2.0
-----
* Add documentation.
* Add tests.
* Add continuous integration using github actions.
* Add functions for storing to and loading from netcdf.
* Add description of the data format.
* Add function which ensures that a dataset is in this data format.
* Provide all functions using a pint extension accessor.

0.1.0
-----

* First development release.
