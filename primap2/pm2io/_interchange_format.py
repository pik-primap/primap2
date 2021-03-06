import itertools
import re
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import strictyaml as sy
import xarray as xr
from loguru import logger
from ruamel.yaml import YAML

# entity is mandatory in the interchange format because it is transformed
# into the variables
# unit is mandatory in the interchange format because it is transformed
# into the pint units
INTERCHANGE_FORMAT_MANDATORY_COLUMNS = ["area", "source", "entity", "unit"]
INTERCHANGE_FORMAT_OPTIONAL_COLUMNS = ["category", "scenario", "provenance", "model"]
INTERCHANGE_FORMAT_COLUMN_ORDER = [
    "source",
    "scenario",
    "provenance",
    "model",
    "area",
    "entity",
    "unit",
    "category",
]

# Pretty basic schema for now; attrs could be declared more explicitly with the
# mandatory columns etc.
INTERCHANGE_FORMAT_STRICTYAML_SCHEMA = sy.Map(
    {
        sy.Optional("data_file"): sy.Str(),
        "attrs": sy.MapPattern(sy.Str(), sy.Any()),
        "dimensions": sy.MapPattern(sy.Str(), sy.Seq(sy.Str())),
        "time_format": sy.Str(),
    }
)


def dates_to_dimension(ds: xr.Dataset, time_format: str = "%Y") -> xr.DataArray:
    """
    This function converts a xr.Dataset where each time point is an individual
    data variable to a xr.DataArray with a time dimension.
    All variables which are not in the index are assumed to be time points.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with individual time points as data variables

    time_format : str
        format string for the time points. Default: %Y (for year only)

    Returns
    -------
    reduced: xr.DataArray
        xr.DataArray with the time as a dimension and time points as values
    """
    empty_vars = (x for x in ds if ds[x].count() == 0)
    da = ds.drop_vars(empty_vars).to_array("time")
    da["time"] = pd.to_datetime(da["time"].values, format=time_format, exact=False)
    return da


def metadata_for_variable(unit: str, variable: str) -> dict:
    """Convert a primap2 unit and variable key to a metadata dict.

    Derives the information needed for the data variable's attrs dict from the unit
    and the variable's key.
    Takes GWP information from the variable name in primap2 style (if present).

    Parameters
    ----------
    unit: str
        unit to be stored in the attrs dict
    variable: str
        entity to process

    Returns
    -------
    attrs: dict
        attrs for use in DataArray.attrs

    Examples
    --------

    >>> metadata_for_variable("Gg CO2 / year", "Kyoto-GHG (SARGWP100)")
    {'units': 'Gg CO2 / year', 'gwp_context': 'SARGWP100', 'entity': 'Kyoto-GHG'}

    """
    attrs = {"units": unit}

    regex_gwp = r"\s\(([A-Za-z0-9]*)\)$"
    regex_variable = r"^(.*)\s\([a-zA-z0-9]*\)$"

    gwp = re.findall(regex_gwp, variable)
    if gwp:
        attrs["gwp_context"] = gwp[0]
        entity = re.findall(regex_variable, variable)
        if not entity:
            logger.error("Can't extract entity from " + variable)
            raise ValueError("Can't extract entity from " + variable)
        attrs["entity"] = entity[0]
    else:
        attrs["entity"] = variable
    return attrs


def write_interchange_format(
    filepath: Union[str, Path], data: pd.DataFrame, attrs: Optional[dict] = None
) -> None:
    """Write dataset in interchange format to disk.

    Writes an interchange format dataset consisting of a pandas Dataframe and an
    additional meta data dict to disk. The data is stored in a csv file while the
    additional metadata is written to a yaml file.

    Parameters
    ----------
    filepath: str or pathlib.Path
        path and filename stem for the dataset. If a file ending is given it will be
        ignored and replaced by .csv for the data and .yaml for the metadata

    data: pandas.DataFrame
        DataFrame in PRIMAP2 interchange format

    attrs: dict, optional
        Interchange format meta data dict. Default: use data.attrs .
    """
    if attrs is None:
        attrs = data.attrs

    # make sure filepath is a Path object
    filepath = Path(filepath)
    data_file = filepath.parent / (filepath.stem + ".csv")
    meta_file = filepath.parent / (filepath.stem + ".yaml")

    # write the data
    data.to_csv(data_file, index=False, float_format="%g")

    attrs["data_file"] = data_file.name

    yaml = YAML()
    # settings for strictyaml compatibility: don't use flow style or aliases
    yaml.default_flow_style = False
    yaml.representer.ignore_aliases = lambda x: True
    with meta_file.open("w") as fd:
        yaml.dump(attrs, fd)


def read_interchange_format(
    filepath: Union[str, Path],
) -> pd.DataFrame:
    """Read a dataset in the interchange format from disk into memory.

    Reads an interchange format dataset from disk. The data is stored
    in a csv file while the additional metadata is stored in a yaml
    file. This function takes the yaml file as parameter, the data file is specified
    in the yaml file. If no or a wrong ending is given the function tries to load
    a file by the same name with the ending `.yaml`.

    Parameters
    ----------
    filepath: str or pathlib.Path
        path and filename for the dataset (the yaml file, not data file).

    Returns
    -------
    data: pandas.DataFrame
        DataFrame with the read data in PRIMAP2 interchange format
    """
    filepath = Path(filepath)
    if not filepath.exists():
        filepath = filepath.with_suffix(".yaml")
    with filepath.open() as meta_file:
        yaml = sy.load(meta_file.read(), schema=INTERCHANGE_FORMAT_STRICTYAML_SCHEMA)
        meta_data = yaml.data

    if "data_file" in meta_data.keys():
        data = pd.read_csv(filepath.parent / meta_data["data_file"])
    else:
        # no file information given, check for file with same name
        datafile = filepath.parent / (filepath.stem + ".csv")
        if datafile.exists():
            data = pd.read_csv(datafile)
        else:
            raise FileNotFoundError(
                f"Data file not specified in {filepath} and data file not found at "
                f"{datafile}."
            )

    data.attrs = meta_data

    return data


def from_interchange_format(
    data: pd.DataFrame,
    attrs: Optional[dict] = None,
    max_array_size: int = 1024 * 1024 * 1024,
) -> xr.Dataset:
    """Convert dataset from the interchange format to the standard PRIMAP2 format.

    Converts an interchange format DataFrame with added metadata to a PRIMAP2 xarray
    data structure. All column names and attrs are expected to be already in PRIMAP2
    format as defined for the interchange format. The attrs dict is given explicitly as
    the attrs functionality in pandas is experimental.

    Parameters
    ----------
    data: pd.DataFrame
        pandas DataFrame in PRIMAP2 interchange format.
    attrs: dict, optional
        attrs dict as defined for the PRIMAP2 interchange format. Default: use
        data.attrs.
    max_array_size: int, optional
        Maximum permitted projected array size. Larger sizes will raise an exception.
        Default: 1 G, corresponding to about 4 GB of memory usage.

    Returns
    -------
    obj: xr.Dataset
        xr dataset with the converted data
    """
    if attrs is None:
        attrs = data.attrs

    if "entity_terminology" in attrs["attrs"]:
        entity_col = f"entity ({attrs['attrs']['entity_terminology']})"
    else:
        entity_col = "entity"

    # find the time columns
    if_index_cols = set(itertools.chain(*attrs["dimensions"].values()))
    time_cols = set(data.columns.values) - if_index_cols

    # convert to xarray
    data_xr = data.to_xarray()
    index_cols = if_index_cols - {"unit", "time"}
    data_xr = data_xr.set_index({"index": list(index_cols)})
    # take the units out as they increase dimensionality and we have only one unit per
    # entity/variable
    units = data_xr["unit"]
    del data_xr["unit"]

    # build full dimensions dict from specification with default entry "*"
    entities = np.unique(data_xr[entity_col].values)
    dimensions = {}
    for entity in entities:
        if entity in attrs["dimensions"]:
            dims = attrs["dimensions"][entity]
        else:
            dims = attrs["dimensions"]["*"]
        dimensions[entity] = dims
    attrs["dimensions"] = dimensions

    # check resulting shape to estimate memory consumption
    dim_lens = {dim: len(np.unique(data_xr[dim].dropna("index"))) for dim in index_cols}
    dim_lens["time"] = len(time_cols)
    shapes = []
    for entity, dims in attrs["dimensions"].items():
        shapes.append([dim_lens[dim] for dim in dims if dim != "unit"])
    array_size = np.sum((np.product(shape) for shape in shapes))
    logger.debug(f"Expected array shapes: {shapes}, resulting in size {array_size:,}.")
    if array_size > max_array_size:
        logger.error(
            f"Set with {len(shapes)} entities and a total of {len(index_cols)} "
            f"dimensions will have a size of {array_size:,} "
            f"due to the shapes {shapes}. Aborting to avoid out-of-memory errors. To "
            f"continue, raise max_array_size (currently {max_array_size:,})."
        )
        raise ValueError(
            f"Resulting array too large: {array_size:,} > {max_array_size:,}. To "
            f"continue, raise max_array_size."
        )

    # convert time to dimension and entity to variables.
    if "time_format" in attrs:
        da = dates_to_dimension(data_xr, time_format=attrs["time_format"])
    else:
        da = dates_to_dimension(data_xr)
    data_vars = {}
    for entity, dims in attrs["dimensions"].items():
        da_entity = da.loc[{entity_col: entity}]
        # we still have a full MultiIndex, so trim it to the relevant dimensions
        da_entity["index"] = da_entity.indexes["index"].droplevel(
            list(index_cols - set(dims))
        )
        # now we can safely unstack the index
        data_vars[entity] = da_entity.unstack("index")

    data_xr = xr.Dataset(data_vars)

    # fill the entity/variable attributes
    for variable in data_xr:
        csv_units = np.unique(units.loc[{entity_col: variable}])
        if len(csv_units) > 1:
            logger.error(
                f"More than one unit for entity {variable!r}: {csv_units!r}. "
                + "There is an error in the unit harmonization."
            )
            raise ValueError(f"More than one unit for {variable!r}: {csv_units!r}.")
        data_xr[variable].attrs = metadata_for_variable(csv_units[0], variable)

    # add the dataset wide attributes
    data_xr.attrs = attrs["attrs"]

    data_xr = data_xr.pr.quantify()

    data_xr.pr.ensure_valid()
    return data_xr
