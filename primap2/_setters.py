import typing

import numpy as np
import xarray as xr

from . import _accessor_base


class DataArraySettersAccessor(_accessor_base.BaseDataArrayAccessor):
    def set(
        self,
        dim: typing.Hashable,
        key: typing.Any,
        value: typing.Union[xr.DataArray, np.ndarray],
        *,
        value_dims: typing.Optional[typing.List[typing.Hashable]] = None,
        existing: str = "error",
    ) -> xr.DataArray:
        """Set values, expanding the given dimension as necessary.

        The handling of already existing key values can be selected using the
        ``existing`` parameter.

        Parameters
        ----------
        dim: str
            Dimension along which values should be set.
        key: scalar or list of scalars
            Keys in the dimension which should be set. Key values which are missing
            in the dimension are inserted. The handling of key values which already
            exist in the dimension is determined by the ``existing`` parameter.
        value: xr.DataArray or np.ndarray
            Values that will be inserted at the positions specified by ``key``.
            ``value`` needs to be broadcastable to ``da[{dim: key}]``.
        value_dims: list of str, optional
            Specifies the dimensions of ``value``. If ``value`` is not a DataArray
            and ``da[{dim: key}]`` is higher-dimensional, it is necessary to specify
            the value dimensions.
        existing: "error", "overwrite", or "fillna", optional
            How to handle existing keys. If ``existing="error"`` (default), a ValueError
            is raised if any key already exists. If ``existing="overwrite"``, new values
            overwrite current values for existing keys. If ``existing="fillna"``, the
            new values only overwrite NaN values for existing keys.

        Examples
        --------
        >>> import pandas as pd
        >>> import xarray as xr
        >>> import numpy as np
        >>> da = xr.DataArray(
        ...     [[0.0, 1.0, 2.0, 3.0], [2.0, 3.0, 4.0, 5.0]],
        ...     coords=[
        ...         ("area (ISO3)", ["COL", "MEX"]),
        ...         ("time", pd.date_range("2000", "2003", freq="AS")),
        ...     ],
        ... )
        >>> da
        <xarray.DataArray (area (ISO3): 2, time: 4)>
        array([[0., 1., 2., 3.],
               [2., 3., 4., 5.]])
        Coordinates:
          * area (ISO3)  (area (ISO3)) <U3 'COL' 'MEX'
          * time         (time) datetime64[ns] 2000-01-01 2001-01-01 ... 2003-01-01

        Setting an existing value

        >>> da.pr.set("area", "COL", np.array([0.5, 0.6, 0.7, 0.8]))
        Traceback (most recent call last):
        ...
        ValueError: Values {'COL'} for 'area (ISO3)' already exist. Use existing='ove...
        >>> da.pr.set(
        ...     "area", "COL", np.array([0.5, 0.6, 0.7, 0.8]), existing="overwrite"
        ... )
        <xarray.DataArray (area (ISO3): 2, time: 4)>
        array([[0.5, 0.6, 0.7, 0.8],
               [2. , 3. , 4. , 5. ]])
        Coordinates:
          * area (ISO3)  (area (ISO3)) object 'COL' 'MEX'
          * time         (time) datetime64[ns] 2000-01-01 2001-01-01 ... 2003-01-01

        Introducing a new value uses the same syntax

        >>> da.pr.set("area", "ARG", np.array([0.5, 0.6, 0.7, 0.8]))
        <xarray.DataArray (area (ISO3): 3, time: 4)>
        array([[0.5, 0.6, 0.7, 0.8],
               [0. , 1. , 2. , 3. ],
               [2. , 3. , 4. , 5. ]])
        Coordinates:
          * area (ISO3)  (area (ISO3)) object 'ARG' 'COL' 'MEX'
          * time         (time) datetime64[ns] 2000-01-01 2001-01-01 ... 2003-01-01

        You can also mix existing and new values

        >>> da.pr.set(
        ...     "area",
        ...     ["COL", "ARG"],
        ...     np.array([[0.5, 0.6, 0.7, 0.8], [5, 6, 7, 8]]),
        ...     existing="overwrite",
        ... )
        <xarray.DataArray (area (ISO3): 3, time: 4)>
        array([[5. , 6. , 7. , 8. ],
               [0.5, 0.6, 0.7, 0.8],
               [2. , 3. , 4. , 5. ]])
        Coordinates:
          * area (ISO3)  (area (ISO3)) object 'ARG' 'COL' 'MEX'
          * time         (time) datetime64[ns] 2000-01-01 2001-01-01 ... 2003-01-01

        If you want to use broadcasting or have more dimensions, the dimensions of your
        input can't be determined automatically anymore. Use the value_dims parameter
        to supply this information.

        >>> da.pr.set(
        ...     "area",
        ...     ["COL", "ARG"],
        ...     np.array([0.5, 0.6, 0.7, 0.8]),
        ...     existing="overwrite",
        ... )
        Traceback (most recent call last):
        ...
        ValueError: Could not automatically determine value dimensions, please use th...
        >>> da.pr.set(
        ...     "area",
        ...     ["COL", "ARG"],
        ...     np.array([0.5, 0.6, 0.7, 0.8]),
        ...     value_dims=["time"],
        ...     existing="overwrite",
        ... )
        <xarray.DataArray (area (ISO3): 3, time: 4)>
        array([[0.5, 0.6, 0.7, 0.8],
               [0.5, 0.6, 0.7, 0.8],
               [2. , 3. , 4. , 5. ]])
        Coordinates:
          * area (ISO3)  (area (ISO3)) object 'ARG' 'COL' 'MEX'
          * time         (time) datetime64[ns] 2000-01-01 2001-01-01 ... 2003-01-01

        Instead of overwriting existing values, you can also choose to only fill missing
        values.

        >>> da.pr.loc[{"area": "COL", "time": "2001"}] = np.nan
        >>> da
        <xarray.DataArray (area (ISO3): 2, time: 4)>
        array([[ 0., nan,  2.,  3.],
               [ 2.,  3.,  4.,  5.]])
        Coordinates:
          * area (ISO3)  (area (ISO3)) <U3 'COL' 'MEX'
          * time         (time) datetime64[ns] 2000-01-01 2001-01-01 ... 2003-01-01
        >>> da.pr.set(
        ...     "area",
        ...     ["COL", "ARG"],
        ...     np.array([0.5, 0.6, 0.7, 0.8]),
        ...     value_dims=["time"],
        ...     existing="fillna",
        ... )
        <xarray.DataArray (area (ISO3): 3, time: 4)>
        array([[0.5, 0.6, 0.7, 0.8],
               [0. , 0.6, 2. , 3. ],
               [2. , 3. , 4. , 5. ]])
        Coordinates:
          * area (ISO3)  (area (ISO3)) object 'ARG' 'COL' 'MEX'
          * time         (time) datetime64[ns] 2000-01-01 2001-01-01 ... 2003-01-01

        Because you can also supply a DataArray as a value, it is easy to define values
        from existing values using arithmetic

        >>> da.pr.set("area", "ARG", da.pr.loc[{"area": "COL"}] * 2)
        <xarray.DataArray (area (ISO3): 3, time: 4)>
        array([[ 0., nan,  4.,  6.],
               [ 0., nan,  2.,  3.],
               [ 2.,  3.,  4.,  5.]])
        Coordinates:
          * area (ISO3)  (area (ISO3)) object 'ARG' 'COL' 'MEX'
          * time         (time) datetime64[ns] 2000-01-01 2001-01-01 ... 2003-01-01

        Returns
        -------
        da : xr.DataArray
            modified DataArray
        """
        dim = self._da.pr.dim_alias_translations.get(dim, dim)
        if dim not in self._da.dims:
            raise ValueError(f"Dimension {dim!r} does not exist.")

        if value_dims is not None:
            value_dims = [
                self._da.pr.dim_alias_translations.get(x, x) for x in value_dims
            ]

        if np.ndim(key) == 0:  # scalar
            key = [key]
        else:
            key = key

        if existing == "error":
            already_existing = set(self._da[dim].values).intersection(set(key))
            if already_existing:
                raise ValueError(
                    f"Values {already_existing!r} for {dim!r} already exist."
                    " Use existing='overwrite' or 'fillna' to avoid this error."
                )
            # without conflicting keys, the new keys are filled
            existing = "fillna"

        if isinstance(value, xr.DataArray):
            if value_dims is not None:
                raise ValueError("value_dims given, but value is already a DataArray.")

            # conform value to given dim: key
            if dim not in value.dims:
                if dim in value.coords:
                    value = value.reset_coords(dim, drop=True)
                value = value.expand_dims({dim: key})  # type: ignore
            else:
                value = value.loc[{dim: key}]
            value, expanded = xr.broadcast(value, self._da)
            value.attrs = self._da.attrs
            value.name = self._da.name

        else:
            new_index = list(self._da[dim].values)
            for item in key:
                if item not in new_index:
                    new_index.append(item)
            new_index = np.array(new_index, dtype=self._da[dim].dtype)
            expanded = self._da.reindex({dim: new_index}, copy=False)
            sel = expanded.loc[{dim: key}]

            if value_dims is None:
                value_dims = []
                for i, dim in enumerate(sel.dims):
                    if sel.shape[i] > 1:
                        value_dims.append(dim)
                if len(value_dims) != len(value.shape):
                    raise ValueError(
                        "Could not automatically determine value dimensions, please"
                        " use the value_dims parameter."
                    )
            value = xr.DataArray(
                value,
                coords=[(dim, sel[dim]) for dim in value_dims],
                name=self._da.name,
                attrs=self._da.attrs,
            ).broadcast_like(sel)

        if existing == "overwrite":
            return value.combine_first(expanded)
        elif existing == "fillna":
            return expanded.combine_first(value)
        else:
            raise ValueError(
                "If given, 'existing' must specify one of 'error', 'overwrite', or"
                f" 'fillna', not {existing!r}."
            )


class DatasetSettersAccessor(_accessor_base.BaseDatasetAccessor):
    @staticmethod
    def _set_apply(
        da: xr.DataArray,
        dim: typing.Hashable,
        key: typing.Any,
        value: xr.Dataset,
        existing: str,
    ) -> xr.DataArray:
        if dim not in da.dims:
            return da
        return da.pr.set(dim=dim, key=key, value=value[da.name], existing=existing)

    def set(
        self,
        dim: typing.Hashable,
        key: typing.Any,
        value: xr.Dataset,
        *,
        existing: str = "error",
    ) -> xr.Dataset:
        """Set values, expanding the given dimension as necessary.

        All data variables which have the given dimension are modified.
        The affected data variables are mutated using
        ``DataArray.pr.set(dim, key, value[name], existing=existing)``.

        Parameters
        ----------
        dim: str
            Dimension along which values should be set. Only data variables which have
            this dimension are mutated.
        key: scalar or list of scalars
            Keys in the dimension which should be set. Key values which are missing
            in the dimension are inserted. The handling of key values which already
            exist in the dimension is determined by the ``existing`` parameter.
        value: xr.Dataset
            Values that will be inserted at the positions specified by ``key``.
            ``value`` needs to contain all data variables which have the dimension.
            ``value`` has to be broadcastable to ``ds.pr.loc[{dim: key}]``.
        existing: "error", "overwrite", or "fillna", optional
            How to handle existing keys. If ``existing="error"`` (default), a ValueError
            is raised if any key already exists. If ``existing="overwrite"``, new values
            overwrite current values for existing keys. If ``existing="fillna"``, the
            new values only overwrite NaN values for existing keys.

        Examples
        --------
        >>> import pandas as pd
        >>> import xarray as xr
        >>> import numpy as np
        >>> area = ("area (ISO3)", ["COL", "MEX"])
        >>> time = ("time", pd.date_range("2000", "2003", freq="AS"))
        >>> ds = xr.Dataset(
        ...     {
        ...         "CO2": xr.DataArray(
        ...             [[0.0, 1.0, 2.0, 3.0], [2.0, 3.0, 4.0, 5.0]],
        ...             coords=[area, time],
        ...         ),
        ...         "SF4": xr.DataArray(
        ...             [[0.5, 1.5, 2.5, 3.5], [2.5, 3.5, np.nan, 5.5]],
        ...             coords=[area, time],
        ...         ),
        ...     },
        ...     attrs={"area": "area (ISO3)"},
        ... )
        >>> ds
        <xarray.Dataset>
        Dimensions:      (area (ISO3): 2, time: 4)
        Coordinates:
          * area (ISO3)  (area (ISO3)) <U3 'COL' 'MEX'
          * time         (time) datetime64[ns] 2000-01-01 2001-01-01 ... 2003-01-01
        Data variables:
            CO2          (area (ISO3), time) float64 0.0 1.0 2.0 3.0 2.0 3.0 4.0 5.0
            SF4          (area (ISO3), time) float64 0.5 1.5 2.5 3.5 2.5 3.5 nan 5.5
        Attributes:
            area:     area (ISO3)

        Setting an existing value

        >>> ds.pr.set("area", "MEX", ds.pr.loc[{"area": "COL"}] * 20)
        Traceback (most recent call last):
        ...
        ValueError: Values {'MEX'} for 'area (ISO3)' already exist. Use existing='ove...
        >>> ds.pr.set(
        ...     "area", "MEX", ds.pr.loc[{"area": "COL"}] * 20, existing="overwrite"
        ... )
        <xarray.Dataset>
        Dimensions:      (area (ISO3): 2, time: 4)
        Coordinates:
          * area (ISO3)  (area (ISO3)) object 'COL' 'MEX'
          * time         (time) datetime64[ns] 2000-01-01 2001-01-01 ... 2003-01-01
        Data variables:
            CO2          (area (ISO3), time) float64 0.0 1.0 2.0 3.0 0.0 20.0 40.0 60.0
            SF4          (area (ISO3), time) float64 0.5 1.5 2.5 3.5 10.0 30.0 50.0 70.0
        Attributes:
            area:     area (ISO3)

        Instead of overwriting existing values, you can also choose to only fill
        missing values

        >>> ds.pr.set("area", "MEX", ds.pr.loc[{"area": "COL"}] * 20, existing="fillna")
        <xarray.Dataset>
        Dimensions:      (area (ISO3): 2, time: 4)
        Coordinates:
          * area (ISO3)  (area (ISO3)) object 'COL' 'MEX'
          * time         (time) datetime64[ns] 2000-01-01 2001-01-01 ... 2003-01-01
        Data variables:
            CO2          (area (ISO3), time) float64 0.0 1.0 2.0 3.0 2.0 3.0 4.0 5.0
            SF4          (area (ISO3), time) float64 0.5 1.5 2.5 3.5 2.5 3.5 50.0 5.5
        Attributes:
            area:     area (ISO3)

        Introducing a new value uses the same syntax

        >>> ds.pr.set("area", "BOL", ds.pr.loc[{"area": "COL"}] * 20)
        <xarray.Dataset>
        Dimensions:      (area (ISO3): 3, time: 4)
        Coordinates:
          * area (ISO3)  (area (ISO3)) object 'BOL' 'COL' 'MEX'
          * time         (time) datetime64[ns] 2000-01-01 2001-01-01 ... 2003-01-01
        Data variables:
            CO2          (area (ISO3), time) float64 0.0 20.0 40.0 60.0 ... 3.0 4.0 5.0
            SF4          (area (ISO3), time) float64 10.0 30.0 50.0 70.0 ... 3.5 nan 5.5
        Attributes:
            area:     area (ISO3)

        Note that data variables which do not contain the specified dimension are
        ignored

        >>> ds["population"] = xr.DataArray([1e6, 1.2e6, 1.3e6, 1.4e6], coords=(time,))
        >>> ds
        <xarray.Dataset>
        Dimensions:      (area (ISO3): 2, time: 4)
        Coordinates:
          * area (ISO3)  (area (ISO3)) <U3 'COL' 'MEX'
          * time         (time) datetime64[ns] 2000-01-01 2001-01-01 ... 2003-01-01
        Data variables:
            CO2          (area (ISO3), time) float64 0.0 1.0 2.0 3.0 2.0 3.0 4.0 5.0
            SF4          (area (ISO3), time) float64 0.5 1.5 2.5 3.5 2.5 3.5 nan 5.5
            population   (time) float64 1e+06 1.2e+06 1.3e+06 1.4e+06
        Attributes:
            area:     area (ISO3)
        >>> ds.pr.set("area", "BOL", ds.pr.loc[{"area": "COL"}] * 20)
        <xarray.Dataset>
        Dimensions:      (area (ISO3): 3, time: 4)
        Coordinates:
          * area (ISO3)  (area (ISO3)) object 'BOL' 'COL' 'MEX'
          * time         (time) datetime64[ns] 2000-01-01 2001-01-01 ... 2003-01-01
        Data variables:
            CO2          (area (ISO3), time) float64 0.0 20.0 40.0 60.0 ... 3.0 4.0 5.0
            SF4          (area (ISO3), time) float64 10.0 30.0 50.0 70.0 ... 3.5 nan 5.5
            population   (time) float64 1e+06 1.2e+06 1.3e+06 1.4e+06
        Attributes:
            area:     area (ISO3)

        Returns
        -------
        ds : xr.Dataset
            modified Dataset
        """
        dim = self._ds.pr.dim_alias_translations.get(dim, dim)
        if dim not in self._ds.dims:
            raise ValueError(f"Dimension {dim!r} does not exist.")

        if not isinstance(value, xr.Dataset):
            raise TypeError(f"value must be a Dataset, not {type(value)}")

        return self._ds.map(
            self._set_apply,
            keep_attrs=True,
            dim=dim,
            key=key,
            value=value,
            existing=existing,
        )