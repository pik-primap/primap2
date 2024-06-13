API
===
.. currentmodule:: primap2

Top-level API
-------------

.. autosummary::
    :toctree: generated/

    open_dataset
    ureg
    Not
    ProcessingStepDescription
    TimeseriesProcessingDescription

.. toctree::
    :maxdepth: 2

    pm2io

.. currentmodule:: xarray

DataArray
---------

.. _daattr:

Attributes
~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_attribute.rst

    DataArray.pr.loc
    DataArray.pr.dim_alias_translations
    DataArray.pr.gwp_context

.. _dameth:

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_method.rst

    DataArray.pr.__getitem__
    DataArray.pr.any
    DataArray.pr.convert_to_gwp
    DataArray.pr.convert_to_gwp_like
    DataArray.pr.convert_to_mass
    DataArray.pr.count
    DataArray.pr.coverage
    DataArray.pr.dequantify
    DataArray.pr.downscale_timeseries
    DataArray.pr.fill_all_na
    DataArray.pr.merge
    DataArray.pr.quantify
    DataArray.pr.set
    DataArray.pr.sum
    DataArray.pr.to_df


Dataset
-------

.. _dsattr:

Attributes
~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_attribute.rst

    Dataset.pr.comment
    Dataset.pr.contact
    Dataset.pr.dim_alias_translations
    Dataset.pr.entity_terminology
    Dataset.pr.institution
    Dataset.pr.loc
    Dataset.pr.publication_date
    Dataset.pr.references
    Dataset.pr.rights
    Dataset.pr.title

.. _dsmeth:

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_method.rst

    Dataset.pr.__getitem__
    Dataset.pr.any
    Dataset.pr.count
    Dataset.pr.coverage
    Dataset.pr.dequantify
    Dataset.pr.downscale_gas_timeseries
    Dataset.pr.downscale_timeseries
    Dataset.pr.ensure_valid
    Dataset.pr.fill_all_na
    Dataset.pr.fill_na_gas_basket_from_contents
    Dataset.pr.gas_basket_contents_sum
    Dataset.pr.has_processing_info
    Dataset.pr.merge
    Dataset.pr.quantify
    Dataset.pr.remove_processing_info
    Dataset.pr.set
    Dataset.pr.sum
    Dataset.pr.to_df
    Dataset.pr.to_interchange_format
    Dataset.pr.to_netcdf
