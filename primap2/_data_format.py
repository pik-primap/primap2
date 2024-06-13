import contextlib
import datetime
import pathlib
import typing
from collections.abc import Hashable, Iterable, Mapping
from typing import IO

import msgpack
import numpy as np
import pandas as pd
import pint
import xarray as xr
from attr import define
from loguru import logger

from primap2._selection import translations_from_dims

from . import _accessor_base, pm2io
from ._units import ureg


def open_dataset(
    filename_or_obj: str | pathlib.Path | IO,
    group: str | None = None,
    chunks: int | dict | None = None,
    cache: bool | None = None,
    drop_variables: str | Iterable | None = None,
    backend_kwargs: dict | None = None,
) -> xr.Dataset:
    """Open and decode a dataset from a file or file-like object.

    Parameters
    ----------
    filename_or_obj : str, Path, file-like, or DataStore
        Strings and Path objects are interpreted as a path to a netCDF file
        or an OpenDAP URL and opened with h5py. Byte-strings or file-like
        objects are also supported.
    group : str, optional
        Path to the netCDF4 group in the given file to open.
    chunks : int or dict, optional
        If chunks is provided, it is used to load the new dataset into dask
        arrays. ``chunks={}`` loads the dataset with dask using a single
        chunk for all arrays.
    cache : bool, optional
        If True, cache data loaded from the underlying datastore in memory as
        NumPy arrays when accessed to avoid reading from the underlying data-
        store multiple times. Defaults to True unless you specify the `chunks`
        argument to use dask, in which case it defaults to False. Does not
        change the behavior of coordinates corresponding to dimensions, which
        always load their data from disk into a ``pandas.Index``.
    drop_variables: str or iterable, optional
        A variable or list of variables to exclude from being parsed from the
        dataset. This may be useful to drop variables with problems or
        inconsistent values.
    backend_kwargs: dict, optional
        A dictionary of keyword arguments to pass on to the backend. This
        may be useful when backend options would improve performance or
        allow user control of dataset processing.

    Returns
    -------
    dataset : Dataset
        The newly created dataset.

    Notes
    -----
    ``open_dataset`` opens the file with read-only access. When you modify
    values of a Dataset, even one linked to files on disk, only the in-memory
    copy you are manipulating in xarray is modified: the original file on disk
    is never touched.
    """
    ds = xr.open_dataset(
        filename_or_obj=filename_or_obj,
        group=group,
        chunks=chunks,
        cache=cache,
        drop_variables=drop_variables,
        backend_kwargs=backend_kwargs,
        engine="h5netcdf",
    ).pint.quantify(unit_registry=ureg)
    if "sec_cats" in ds.attrs:
        ds.attrs["sec_cats"] = list(ds.attrs["sec_cats"])
    if "publication_date" in ds.attrs:
        ds.attrs["publication_date"] = datetime.date.fromisoformat(
            ds.attrs["publication_date"]
        )
    for entity in ds:
        if entity.startswith("Processing of "):
            ds[entity].data = np.vectorize(
                lambda x: TimeseriesProcessingDescription.deserialize(x)
            )(ds[entity].data)
    return ds


class DatasetDataFormatAccessor(_accessor_base.BaseDatasetAccessor):
    """MixIn class which provides functions for checking the data format and saving
    of Datasets."""

    def ensure_valid(self) -> None:
        """Ensure this is a valid primap2 data set.

        Logs any deviations or non-standard properties. If the dataset violates any
        hard requirements of primap2 data sets, an exception is raised, otherwise the
        function simply returns.
        """
        if not isinstance(self._ds, xr.Dataset):
            logger.error("object is not an xarray Dataset.")
            raise ValueError("ds is not an xr.Dataset")

        ensure_valid_dimensions(self._ds)
        ensure_no_dimension_without_coordinates(self._ds)
        ensure_valid_coordinates(self._ds)
        ensure_valid_coordinate_values(self._ds)
        ensure_valid_data_variables(self._ds)
        ensure_valid_attributes(self._ds)

    def to_interchange_format(self, time_format: str = "%Y") -> pd.DataFrame:
        """Convert dataset to the interchange format.

        The interchange format consists of a pandas DataFrame in wide format with the
        meta data dictionary stored in the DataFrame's attrs dict.

        Parameters
        ----------
        time_format: str, optional
          string for strftime formatting of the "time" axis.

        Returns
        -------
        df: pd.DataFrame
        """
        dsd = self._ds.pr.dequantify()

        dsd["time"] = dsd["time"].dt.strftime(time_format)
        time_cols = list(dsd["time"].values)

        if "entity_terminology" in self._ds.attrs:
            entity_col = f"entity ({self._ds.attrs['entity_terminology']})"
        else:
            entity_col = "entity"

        dfs = []
        entities = []
        for x in dsd:
            if isinstance(x, str) and x.startswith("Processing of "):
                continue
            entities.append(x)
            df = (
                dsd[x]
                .to_dataset("time")
                .to_dataframe()
                .dropna(
                    how="all",
                    subset=time_cols,
                )
                .reset_index()
            )
            df[entity_col] = x
            df["unit"] = dsd[x].attrs.get("units", "no unit")
            dfs.append(df)

        df = pd.concat(dfs, ignore_index=True)

        add_coords = list(set(self._ds.coords) - set(self._ds.dims))
        df, dims = pm2io._data_reading.sort_columns_and_rows(
            df,
            dimensions=[dim for dim in dsd.dims if dim != "time"]
            + [entity_col, "unit"]
            + add_coords,
        )

        dimensions = {}
        all_dimensions = [str(dim) for dim in dsd.dims] + [entity_col, "unit"]
        for entity in entities:
            dims = [str(dim) for dim in dsd[entity].dims] + [entity_col, "unit"]
            if set(dims) == set(all_dimensions):
                # use default option for entities which use all the dataset's dimensions
                if "*" not in dimensions:
                    dimensions["*"] = all_dimensions
            else:
                dimensions[entity] = dims

        # all non-float dtypes are specified
        dtypes = {
            entity: str(dsd[entity].dtype)
            for entity in entities
            if dsd[entity].dtype != float
        }

        # add attrs for additional coords
        additional_coordinates = {}
        for coord in add_coords:
            coords_current = set(self._ds.coords[coord].coords) - {coord}
            if len(coords_current) != 1:
                logger.error(
                    f"Additional coordinate {coord!r} has more than one dimension, "
                    f"which is not supported."
                )
                raise ValueError(
                    f"Additional coordinate {coord!r} has more than one dimension"
                )

            additional_coordinates[coord] = next(iter(coords_current))

        df.attrs = {
            "attrs": self._ds.attrs,
            "time_format": time_format,
            "dimensions": dimensions,
        }
        if additional_coordinates:
            df.attrs["additional_coordinates"] = additional_coordinates
        if dtypes:
            df.attrs["dtypes"] = dtypes

        return df

    def to_netcdf(
        self,
        path: pathlib.Path | str,
        mode: str = "w",
        group: str | None = None,
        encoding: Mapping | None = None,
    ) -> bytes | None:
        """Write dataset contents to a netCDF file.

        Parameters
        ----------
        path : str or Path
            File path to which to save this dataset.
        mode : {"w", "a"}, default: "w"
            Write ('w') or append ('a') mode. If mode='w', any existing file at
            this location will be overwritten. If mode='a', existing variables
            will be overwritten.
        group : str, optional
            Path to the netCDF4 group in the given file to open. The group(s)
            will be created if necessary.
        encoding : dict, optional
            Nested dictionary with variable names as keys and dictionaries of
            variable specific encodings as values, e.g.,
            ``{"my_variable": {"dtype": "int16", "scale_factor": 0.1,
            "zlib": True}, ...}``

            This supports both the NetCDF4-style compression
            encoding parameters ``{"zlib": True, "complevel": 9}`` and the h5py
            ones ``{"compression": "gzip", "compression_opts": 9}``.
            This allows using any compression plugin installed in the HDF5
            library, e.g. LZF.
        """
        ds = self._ds.pint.dequantify()
        if "publication_date" in ds.attrs:
            ds.attrs["publication_date"] = ds.attrs["publication_date"].isoformat()
        for entity in ds:
            if (
                isinstance(entity, str)
                and entity.startswith("Processing of ")
                and ds[entity].data.dtype == object
            ):
                ds[entity].data = np.vectorize(lambda x: x.serialize())(ds[entity].data)
        return ds.to_netcdf(
            path=path,
            mode=mode,
            group=group,
            encoding=encoding,
            engine="h5netcdf",
            format="NETCDF4",
        )

    def remove_processing_info(self) -> xr.Dataset:
        """Return dataset with all variables with processing information removed."""
        return self._ds.drop_vars(
            [
                var
                for var in self._ds
                if isinstance(var, str) and var.startswith("Processing of ")
            ]
        )

    def has_processing_info(self) -> bool:
        """True if the dataset has processing information for at least one entity."""
        return any(
            isinstance(var, str) and var.startswith("Processing of ")
            for var in self._ds
        )

    def expand_dims(
        self,
        dim: str,
        coord_value: str,
        terminology: str | None = None,
    ) -> xr.Dataset:
        """
        Add a dimension and coordinate to dataset and if a terminology is given also to attrs.
        Dimension has length 1. primap2 equivalent of xarray's `expand_dims`

        To add a dimension to a PRIMAP2 DataArray use the native xarray method
        `expand_dims`. As PRIMAP2 DataArrays don't have the coordinate information
        in the attributes, no additional code is needed.

        Parameters
        ----------
        dim
            dimension to add. A coordinate with the same name will be added as well
        coord_value
            value for the
        terminology

        Returns
        -------
            dataset with added dimension and coordinate
        """

        ds_out = self._ds.copy(
            deep=True
        )  # deep=True necessary to keep attrs of original ds
        if terminology is not None:
            # get the translation for the attrs (use the shortest key to get the right key for the attrs)
            dim_to_add = f"{dim} ({terminology})"
            translations = translations_from_dims([dim_to_add])
            shortest_key = min(list(translations.keys()), key=len)
            # store in attributes
            ds_out.attrs[shortest_key] = translations[shortest_key]
        else:
            dim_to_add = dim

        ds_out = ds_out.expand_dims(dim={dim_to_add: [coord_value]})
        return ds_out


def split_dim_name(dim_name: str) -> tuple[str, str]:
    """Split a dimension name composed of the dimension, and the category set in
    parentheses into its parts."""
    try:
        dim, category_set = dim_name.split("(", 1)
    except ValueError:
        logger.error(f"{dim_name!r} not in the format 'dim (category_set)'.")
        raise ValueError(
            f"{dim_name!r} not in the format 'dim (category_set)'"
        ) from None
    return dim[:-1], category_set[:-1]


def split_var_name(var_name: str) -> tuple[str, str]:
    """Split a variable name composed of the entity and the gwp_context in parentheses
    into its parts."""
    try:
        entity, gwp_context = var_name.split("(", maxsplit=1)
    except ValueError:
        raise ValueError(
            f"Tried to split variable name {var_name!r}, but it does not specify a "
            f"gwp_context in parentheses."
        ) from None
    return entity[:-1], gwp_context[:-1]


def ensure_no_dimension_without_coordinates(ds: xr.Dataset):
    for dim in ds.dims:
        if dim not in ds.coords:
            logger.error(f"No coord found for dimension {dim!r}.")
            raise ValueError(f"dim {dim!r} has no coord")


def ensure_valid_coordinates(ds: xr.Dataset):
    additional_coords = set(ds.coords) - set(ds.dims)
    for coord in ds.coords:
        if not isinstance(coord, str):
            # Note: can not be tested properly because xarray deals poorly with
            # coordinate names which are not sortable with strings so the test never
            # gets here anyway
            logger.error(
                f"Coordinate {coord!r} is of type {type(coord)}, but "
                f"only strings are allowed."
            )
            raise ValueError(f"Coord key {coord!r} is not a string")
        elif coord in additional_coords:
            if " " in coord:
                logger.error(
                    f"Additional coordinate name {coord!r} contains a space, "
                    f"replace it with an underscore."
                )
                raise ValueError(f"Coord key {coord!r} contains a space")


def ensure_valid_attributes(ds: xr.Dataset):
    with contextlib.suppress(KeyError):
        reference = ds.attrs["references"]
        if not reference.startswith("doi:"):
            logger.info(f"Reference information is not a DOI: {reference!r}")
    with contextlib.suppress(KeyError):
        contact = ds.attrs["contact"]
        if "@" not in contact:
            logger.info(f"Contact information is not an email address: {contact!r}.")
    with contextlib.suppress(KeyError):
        publication_date = ds.attrs["publication_date"]
        if not isinstance(publication_date, datetime.date):
            logger.error(
                f"Publication date is not a datetime.date object: {publication_date!r}."
            )
            raise ValueError("Publication date is not a date object.")
    valid_attr_keys = {
        "references",
        "rights",
        "contact",
        "title",
        "comment",
        "institution",
        "area",
        "cat",
        "sec_cats",
        "scen",
        "entity_terminology",
        "publication_date",
    }
    unknown_attr_keys = set(ds.attrs.keys()) - valid_attr_keys
    if unknown_attr_keys:
        logger.warning(f"Unknown metadata in attrs: {unknown_attr_keys!r}, typo?")


def ensure_valid_data_variables(ds: xr.Dataset):
    ds = ensure_units_exist(ds)

    for key in ds:
        da: xr.DataArray = ds[key]  # type: ignore
        ensure_entity_and_units_exist(key, da)
        ensure_entity_and_units_valid(key, da)
        ensure_variable_name(str(key), da)

        if "gwp_context" in da.attrs:
            ensure_gwp_context_valid(str(key), da)
        else:
            ensure_not_gwp(key, da)

        if "described_variable" in da.attrs or (
            isinstance(key, str) and key.startswith("Processing of ")
        ):
            ensure_processing_variable_name(str(key), da)


def ensure_processing_variable_name(name: str, da: xr.DataArray) -> None:
    if "described_variable" not in da.attrs:
        logger.error(
            f"variable name {name!r} implies metadata variable, but 'described_variable'"
            f" is not defined in attrs."
        )
        raise ValueError(f"'described_variable' attr missing for {name!r}")
    if name != f"Processing of {da.attrs['described_variable']}":
        logger.error(
            f"variable name {name!r} is inconsistent with described_variable "
            f"{da.attrs['described_variable']!r}"
        )
        raise ValueError(
            f"variable name {name!r} inconsistent with described_variable"
            f" {da.attrs['described_variable']!r}"
        )


def ensure_entity_and_units_exist(key: Hashable, da: xr.DataArray):
    if "entity" not in da.attrs:
        logger.error(f"{key!r} has no entity declared in attributes.")
        raise ValueError(f"entity missing for {key!r}")

    if da.pint.units is None and da.dtype == float:
        logger.error(f"{key!r} is numerical (float) data, but has no units.")
        raise ValueError(f"units missing for {key!r}")


def ensure_units_exist(ds: xr.Dataset) -> xr.Dataset:
    try:
        # ensure that units are attached
        return ds.pint.quantify(unit_registry=ureg)
    except ValueError:  # raised by pint when units attached and in attrs
        logger.error("'units' in variable attrs, but data is quantified already.")
        raise


def ensure_entity_and_units_valid(key: Hashable, da: xr.DataArray):
    entity = da.attrs["entity"]
    units = da.pint.units
    with contextlib.suppress(pint.UndefinedUnitError):
        # if the entity is a gas and it is not converted to a gwp, the dimensions
        # should be compatible with an emission rate
        unit_entity = ureg(entity)
        if "gwp_context" not in da.attrs and not units.is_compatible_with(
            unit_entity * ureg.Gg / ureg.year
        ):
            logger.warning(
                f"{key!r} has a unit of {units}, which is not "
                f"compatible with an emission rate."
            )


def ensure_gwp_context_valid(key: str, da: xr.DataArray):
    units = da.pint.units
    gwp_context = da.attrs["gwp_context"]

    if units.dimensionality != {"[carbon]": 1, "[mass]": 1, "[time]": -1}:
        logger.error(
            f"{key!r} is a global warming potential, but the dimension is not "
            f"[CO2 * mass / time]."
        )
        raise ValueError(f"{key} has wrong dimensionality for gwp_context.")

    try:
        with ureg.context(gwp_context):
            pass
    except KeyError:
        logger.error(f"gwp_context {gwp_context!r} for {key!r} is not valid.")
        raise ValueError(f"Invalid gwp_context {gwp_context!r} for {key!r}") from None

    if "(" not in key or not key.endswith(")"):
        logger.warning(f"{key!r} has a gwp_context in attrs, but not in its " f"name.")


def ensure_not_gwp(key: Hashable, da: xr.DataArray):
    if (
        da.pint.units is not None
        and da.pint.units.dimensionality == {"[carbon]": 1, "[mass]": 1, "[time]": -1}
        and da.attrs["entity"] != "CO2"
    ):
        logger.warning(
            f"{key!r} has the dimension [CO2 * mass / time], but is "
            f"not CO2. gwp_context missing?"
        )


def ensure_variable_name(key: str, da: xr.DataArray):
    if "gwp_context" in da.attrs:
        common_name = f"{da.attrs['entity']} ({da.attrs['gwp_context']})"
    else:
        common_name = da.attrs["entity"]

    if common_name != key:
        logger.info(f"The name {key!r} is not in standard format {common_name!r}.")


def ensure_valid_coordinate_values(ds: xr.Dataset):
    if "provenance" in ds.dims:
        wrong_values = set(ds["provenance"].data) - {"projected", "measured", "derived"}
        if wrong_values:
            logger.error(f"provenance contains invalid values: {wrong_values!r}")
            raise ValueError(f"Invalid provenance: {wrong_values!r}")


def ensure_valid_dimensions(ds: xr.Dataset):
    required_direct_dims = {"time", "source"}
    required_indirect_dims = {"area"}
    optional_direct_dims = {"provenance", "model"}
    optional_indirect_dims = {"cat", "scen"}  # sec_cats is special

    for req_dim in required_direct_dims:
        if req_dim not in ds.dims:
            logger.error(f"{req_dim!r} not found in dims, but is required.")
            raise ValueError(f"{req_dim!r} not in dims")

        for var in ds:
            if (
                isinstance(var, str)
                and var.startswith("Processing of ")
                and req_dim == "time"
            ):
                if req_dim in ds[var].dims:
                    logger.error(
                        f"{var!r} is a metadata variable, but 'time' is a dimension."
                    )
                    raise ValueError(
                        f"{var!r} contains metadata, but carries 'time' dimension"
                    )
            else:
                if req_dim not in ds[var].dims:
                    logger.error(
                        f"{req_dim!r} not found in dims for variable {var!r}, but is required."
                    )
                    raise ValueError(f"{req_dim!r} not in dims for variable {var!r}")

    required_indirect_dims_long = []
    for req_dim in required_indirect_dims:
        if req_dim not in ds.attrs:
            logger.error(
                f"{req_dim!r} not found in attrs, required dimension is therefore"
                f" undefined."
            )
            raise ValueError(f"{req_dim!r} not in attrs")

        if ds.attrs[req_dim] not in ds.dims:
            logger.error(
                f"{ds.attrs[req_dim]!r} defined as {req_dim!r} dimension, but not found"
                f" in dims."
            )
            raise ValueError(f"{req_dim!r} dimension not in dims")

        required_indirect_dims_long.append(ds.attrs[req_dim])

    included_optional_dims = []
    for opt_dim in optional_indirect_dims:
        if opt_dim in ds.attrs:
            long_name = ds.attrs[opt_dim]
            included_optional_dims.append(long_name)
            if long_name not in ds.dims:
                logger.error(
                    f"{long_name!r} defined as {opt_dim}, but not found in dims."
                )
                raise ValueError(f"{opt_dim!r} not in dims")

    if "sec_cats" in ds.attrs:
        for sec_cat in ds.attrs["sec_cats"]:
            included_optional_dims.append(sec_cat)
            if sec_cat not in ds.dims:
                logger.error(
                    f"Secondary category {sec_cat!r} defined, but not found in dims: "
                    f"{ds.dims}."
                )
                raise ValueError(f"Secondary category {sec_cat!r} not in dims")

    if "sec_cats" in ds.attrs and "cat" not in ds.attrs:
        logger.warning(
            "Secondary category defined, but no primary category defined, weird."
        )

    all_dims = set(ds.dims.keys())
    unknown_dims = (
        all_dims
        - required_direct_dims
        - set(required_indirect_dims_long)
        - optional_direct_dims
        - set(included_optional_dims)
    )

    if unknown_dims:
        logger.warning(
            f"Dimension(s) {unknown_dims} unknown, likely a typo or missing in"
            f" sec_cats."
        )

    for dim in required_indirect_dims.union(optional_indirect_dims):
        if dim in ds.attrs:
            split_dim_name(ds.attrs[dim])


@define(frozen=True, kw_only=True)
class ProcessingStepDescription:
    """Structured description of a processing step done on a timeseries.

    Attributes
    ----------
    time
        Time points for which data was changed during the processing step. Use
        "all" if all time points were changed or it is not specified which time
        points were changed.
    function
        The name of the function which did the processing.
    description
        Human-readable description of the processing step.
    source
        Optional: a short identifier for the source of the data which was used for the
        processing.
    """

    time: np.ndarray[np.datetime64] | typing.Literal["all"]
    function: str
    description: str
    source: str | None = None

    def __str__(self) -> str:
        if self.source is None:
            return (
                f"Using function={self.function} for times={self.time}:"
                f" {self.description}"
            )
        else:
            return (
                f"Using function={self.function} with source={self.source} for "
                f"times={self.time}: {self.description}"
            )

    def unstructure(self) -> dict[str, typing.Any]:
        """Convert into basic python types."""
        return {
            "time": "all"
            if isinstance(self.time, str) and self.time == "all"
            else list(
                np.datetime_as_string(
                    self.time,
                    unit="Y",
                )
            ),
            "description": self.description,
            "function": self.function,
            "source": self.source,
        }

    @classmethod
    def structure(cls, u: dict[str, typing.Any]) -> "ProcessingStepDescription":
        """Initialize from basic python types as created by "unstructure"."""
        time = u.pop("time")
        return cls(
            time="all" if time == "all" else np.array(time, dtype=np.datetime64), **u
        )


@define(frozen=True)
class TimeseriesProcessingDescription:
    """Structured description of all processing steps done on a timeseries.

    Attributes
    ----------
    steps
        Steps that were performed during processing, in order from first to last.
    """

    steps: list[ProcessingStepDescription]

    def __str__(self) -> str:
        return "\n".join(str(step) for step in self.steps)

    def serialize(self) -> bytes:
        """Convert into binary data, e.g. for saving to disk."""
        return msgpack.packb(
            {"steps": [x.unstructure() for x in self.steps]}, use_bin_type=True
        )

    @classmethod
    def deserialize(cls, b: bytes) -> "TimeseriesProcessingDescription":
        """Parse from binary data as produced by "serialize".

        Parameters
        ----------
        b
            Binary data representing a TimeseriesProcessingDescription.
        """
        ust = msgpack.unpackb(b, raw=False, use_list=False)
        return cls(steps=[ProcessingStepDescription.structure(x) for x in ust["steps"]])
