"""xarray extension accessors providing an API under the 'pr' namespace."""

import xarray as xr

from ._aggregate import DataArrayAggregationAccessor, DatasetAggregationAccessor
from ._alias_selection import (
    DataArrayAliasSelectionAccessor,
    DatasetAliasSelectionAccessor,
)
from ._convert import DataArrayConversionAccessor
from ._data_format import DatasetDataFormatAccessor
from ._downscale import DataArrayDownscalingAccessor, DatasetDownscalingAccessor
from ._merge import DataArrayMergeAccessor, DatasetMergeAccessor
from ._metadata import DatasetMetadataAccessor
from ._overview import DataArrayOverviewAccessor, DatasetOverviewAccessor
from ._setters import DataArraySettersAccessor, DatasetSettersAccessor
from ._units import DataArrayUnitAccessor, DatasetUnitAccessor


@xr.register_dataset_accessor("pr")
class PRIMAP2DatasetAccessor(
    DatasetAggregationAccessor,
    DatasetAliasSelectionAccessor,
    DatasetDataFormatAccessor,
    DatasetDownscalingAccessor,
    DatasetMergeAccessor,
    DatasetMetadataAccessor,
    DatasetOverviewAccessor,
    DatasetSettersAccessor,
    DatasetUnitAccessor,
):
    """Collection of methods useful for climate policy analysis."""


@xr.register_dataarray_accessor("pr")
class PRIMAP2DataArrayAccessor(
    DataArrayAggregationAccessor,
    DataArrayAliasSelectionAccessor,
    DataArrayConversionAccessor,
    DataArrayDownscalingAccessor,
    DataArrayMergeAccessor,
    DataArrayOverviewAccessor,
    DataArraySettersAccessor,
    DataArrayUnitAccessor,
):
    """Collection of methods useful for climate policy analysis."""
