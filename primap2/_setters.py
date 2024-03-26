import typing

import numpy as np
import xarray as xr

from . import _accessor_base
from ._alias_selection import alias_dims


class DataArraySettersAccessor(_accessor_base.BaseDataArrayAccessor):
    @staticmethod
    def _sel_error(
        da: xr.DataArray, dim: typing.Hashable, key: typing.Iterable
    ) -> xr.DataArray:
        try:
            return da.loc[{dim: key}]
        except KeyError:
            missing = set(key).difference(set(da[dim].values))
            raise KeyError(
                f"Values {missing!r} not in {dim!r}, use new='extend' to "
                f"automatically insert new values into dim."
            ) from None

    @alias_dims(["dim", "value_dims"])
    def set(
        self,
        dim: typing.Hashable,
        key: typing.Any,
        value: xr.DataArray | np.ndarray,
        *,
        value_dims: list[typing.Hashable] | None = None,
        existing: str = "fillna_empty",
        new: str = "extend",
    ) -> xr.DataArray:
        """Set values, optionally expanding the given dimension as necessary.

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
        existing: "fillna_empty", "error", "overwrite", or "fillna", optional
            How to handle existing keys. If ``existing="fillna_empty"`` (default), new
            values overwrite existing values only if all existing values are NaN.
            If ``existing="error"``, a ValueError is raised if any key already exists
            in the index. If ``existing="overwrite"``, new values overwrite current
            values for existing keys. If ``existing="fillna"``, the new values only
            overwrite NaN values for existing keys.
        new: "extend", or "error", optional
            How to handle new keys. If ``new="extend"`` (default), keys which do not
            exist so far are automatically inserted by extending the dimension.
            If ``new="error"``, a KeyError is raised if any key is not yet in the
            dimension.

        Examples
        --------
        >>> import pandas as pd
        >>> import xarray as xr
        >>> import numpy as np
        >>> da = xr.DataArray(
        ...     [[0.0, 1.0, 2.0, 3.0], [2.0, 3.0, 4.0, 5.0]],
        ...     coords=[
        ...         ("area (ISO3)", ["COL", "MEX"]),
        ...         ("time", pd.date_range("2000", "2003", freq="YS")),
        ...     ],
        ... )
        >>> da
        <xarray.DataArray (area (ISO3): 2, time: 4)> Size: 64B
        array([[0., 1., 2., 3.],
               [2., 3., 4., 5.]])
        Coordinates:
          * area (ISO3)  (area (ISO3)) <U3 24B 'COL' 'MEX'
          * time         (time) datetime64[ns] 32B 2000-01-01 2001-01-01 ... 2003-01-01

        Setting an existing value

        >>> da.pr.set("area", "COL", np.array([0.5, 0.6, 0.7, 0.8]))
        Traceback (most recent call last):
        ...
        ValueError: Values {'COL'} for 'area (ISO3)' already exist and contain data. ...
        >>> da.pr.set(
        ...     "area", "COL", np.array([0.5, 0.6, 0.7, 0.8]), existing="overwrite"
        ... )
        <xarray.DataArray (area (ISO3): 2, time: 4)> Size: 64B
        array([[0.5, 0.6, 0.7, 0.8],
               [2. , 3. , 4. , 5. ]])
        Coordinates:
          * time         (time) datetime64[ns] 32B 2000-01-01 2001-01-01 ... 2003-01-01
          * area (ISO3)  (area (ISO3)) <U3 24B 'COL' 'MEX'

        By default, existing values are only overwritten if all existing values are
        NaN

        >>> da_partly_empty = da.copy(deep=True)
        >>> da_partly_empty.pr.loc[{"area": "COL"}] = np.nan
        >>> da_partly_empty
        <xarray.DataArray (area (ISO3): 2, time: 4)> Size: 64B
        array([[nan, nan, nan, nan],
               [ 2.,  3.,  4.,  5.]])
        Coordinates:
          * area (ISO3)  (area (ISO3)) <U3 24B 'COL' 'MEX'
          * time         (time) datetime64[ns] 32B 2000-01-01 2001-01-01 ... 2003-01-01
        >>> da_partly_empty.pr.set("area", "COL", np.array([0.5, 0.6, 0.7, 0.8]))
        <xarray.DataArray (area (ISO3): 2, time: 4)> Size: 64B
        array([[0.5, 0.6, 0.7, 0.8],
               [2. , 3. , 4. , 5. ]])
        Coordinates:
          * area (ISO3)  (area (ISO3)) <U3 24B 'COL' 'MEX'
          * time         (time) datetime64[ns] 32B 2000-01-01 2001-01-01 ... 2003-01-01
        >>> # if even one value contains data, the default is to raise an Error
        >>> da_partly_empty.pr.loc[{"area": "COL", "time": "2001"}] = 0.6
        >>> da_partly_empty
        <xarray.DataArray (area (ISO3): 2, time: 4)> Size: 64B
        array([[nan, 0.6, nan, nan],
               [2. , 3. , 4. , 5. ]])
        Coordinates:
          * area (ISO3)  (area (ISO3)) <U3 24B 'COL' 'MEX'
          * time         (time) datetime64[ns] 32B 2000-01-01 2001-01-01 ... 2003-01-01
        >>> da_partly_empty.pr.set("area", "COL", np.array([0.5, 0.6, 0.7, 0.8]))
        Traceback (most recent call last):
        ...
        ValueError: Values {'COL'} for 'area (ISO3)' already exist and contain data. ...

        Introducing a new value uses the same syntax as modifying existing values

        >>> da.pr.set("area", "ARG", np.array([0.5, 0.6, 0.7, 0.8]))
        <xarray.DataArray (area (ISO3): 3, time: 4)> Size: 96B
        array([[0.5, 0.6, 0.7, 0.8],
               [0. , 1. , 2. , 3. ],
               [2. , 3. , 4. , 5. ]])
        Coordinates:
          * area (ISO3)  (area (ISO3)) <U3 36B 'ARG' 'COL' 'MEX'
          * time         (time) datetime64[ns] 32B 2000-01-01 2001-01-01 ... 2003-01-01

        You can also mix existing and new values

        >>> da.pr.set(
        ...     "area",
        ...     ["COL", "ARG"],
        ...     np.array([[0.5, 0.6, 0.7, 0.8], [5, 6, 7, 8]]),
        ...     existing="overwrite",
        ... )
        <xarray.DataArray (area (ISO3): 3, time: 4)> Size: 96B
        array([[5. , 6. , 7. , 8. ],
               [0.5, 0.6, 0.7, 0.8],
               [2. , 3. , 4. , 5. ]])
        Coordinates:
          * area (ISO3)  (area (ISO3)) <U3 36B 'ARG' 'COL' 'MEX'
          * time         (time) datetime64[ns] 32B 2000-01-01 2001-01-01 ... 2003-01-01

        If you don't want to automatically extend the dimensions with new values, you
        can request checking that all keys already exist using ``new="error"``:

        >>> da.pr.set("area", "ARG", np.array([0.5, 0.6, 0.7, 0.8]), new="error")
        Traceback (most recent call last):
        ...
        KeyError: "Values {'ARG'} not in 'area (ISO3)', use new='extend' to automatic...

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
        <xarray.DataArray (area (ISO3): 3, time: 4)> Size: 96B
        array([[0.5, 0.6, 0.7, 0.8],
               [0.5, 0.6, 0.7, 0.8],
               [2. , 3. , 4. , 5. ]])
        Coordinates:
          * time         (time) datetime64[ns] 32B 2000-01-01 2001-01-01 ... 2003-01-01
          * area (ISO3)  (area (ISO3)) <U3 36B 'ARG' 'COL' 'MEX'

        Instead of overwriting existing values, you can also choose to only fill missing
        values.

        >>> da.pr.loc[{"area": "COL", "time": "2001"}] = np.nan
        >>> da
        <xarray.DataArray (area (ISO3): 2, time: 4)> Size: 64B
        array([[ 0., nan,  2.,  3.],
               [ 2.,  3.,  4.,  5.]])
        Coordinates:
          * area (ISO3)  (area (ISO3)) <U3 24B 'COL' 'MEX'
          * time         (time) datetime64[ns] 32B 2000-01-01 2001-01-01 ... 2003-01-01
        >>> da.pr.set(
        ...     "area",
        ...     ["COL", "ARG"],
        ...     np.array([0.5, 0.6, 0.7, 0.8]),
        ...     value_dims=["time"],
        ...     existing="fillna",
        ... )
        <xarray.DataArray (area (ISO3): 3, time: 4)> Size: 96B
        array([[0.5, 0.6, 0.7, 0.8],
               [0. , 0.6, 2. , 3. ],
               [2. , 3. , 4. , 5. ]])
        Coordinates:
          * area (ISO3)  (area (ISO3)) <U3 36B 'ARG' 'COL' 'MEX'
          * time         (time) datetime64[ns] 32B 2000-01-01 2001-01-01 ... 2003-01-01

        Because you can also supply a DataArray as a value, it is easy to define values
        from existing values using arithmetic

        >>> da.pr.set("area", "ARG", da.pr.loc[{"area": "COL"}] * 2)
        <xarray.DataArray (area (ISO3): 3, time: 4)> Size: 96B
        array([[ 0., nan,  4.,  6.],
               [ 0., nan,  2.,  3.],
               [ 2.,  3.,  4.,  5.]])
        Coordinates:
          * area (ISO3)  (area (ISO3)) object 24B 'ARG' 'COL' 'MEX'
          * time         (time) datetime64[ns] 32B 2000-01-01 2001-01-01 ... 2003-01-01

        Returns
        -------
        da : xr.DataArray
            modified DataArray
        """
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

        if new not in ("error", "extend"):
            raise ValueError(
                f"If given, 'new' must specify one of 'error' or 'extend', not {new!r}."
            )

        if isinstance(value, xr.DataArray):
            if value_dims is not None:
                raise ValueError("value_dims given, but value is already a DataArray.")

            # conform value to given dim: key
            if dim not in value.dims:
                if dim in value.coords:
                    value = value.reset_coords([dim], drop=True)
                value = value.expand_dims({dim: key})  # type: ignore
            else:
                value = value.loc[{dim: key}]

            # we broadcast value to full self._da, a possible optimization would be
            # to broadcast value only to sel, but would need more careful handling
            # later.
            if new == "extend":
                expanded, value = xr.broadcast(self._da, value)
            else:
                expanded = self._da
                value = value.broadcast_like(expanded)

            sel = self._sel_error(expanded, dim, key)

            value.attrs = self._da.attrs
            value.name = self._da.name

        else:
            # convert value to DataArray

            if new == "extend":
                new_index = list(self._da[dim].values)
                for item in key:
                    if item not in new_index:
                        new_index.append(item)
                new_index = np.array(new_index, dtype=self._da[dim].dtype)
                expanded = self._da.reindex({dim: new_index}, copy=False)
            else:
                expanded = self._da

            sel = self._sel_error(expanded, dim, key)

            if value_dims is None:
                value_dims = []
                for i, idim in enumerate(sel.dims):
                    if sel.shape[i] > 1:
                        value_dims.append(idim)
                if len(value_dims) != len(value.shape):
                    raise ValueError(
                        "Could not automatically determine value dimensions, please"
                        " use the value_dims parameter."
                    )
            value = xr.DataArray(
                value,
                coords=[(idim, sel[idim].data) for idim in value_dims],
                name=self._da.name,
                attrs=self._da.attrs,
            ).broadcast_like(sel)

        if existing == "fillna_empty":
            if sel.count().item() > 0:
                already_existing = set(self._da[dim].values).intersection(set(key))
                raise ValueError(
                    f"Values {already_existing!r} for {dim!r} already exist and contain"
                    f" data. Use existing='overwrite' or 'fillna' to avoid this error."
                )
            existing = "fillna"

        if existing == "overwrite":
            return value.combine_first(expanded)
        elif existing == "fillna":
            return expanded.combine_first(value)
        else:
            raise ValueError(
                "If given, 'existing' must specify one of 'error', 'overwrite', "
                f"'fillna_empty', or 'fillna', not {existing!r}."
            )


class DatasetSettersAccessor(_accessor_base.BaseDatasetAccessor):
    @staticmethod
    def _set_apply(
        da: xr.DataArray,
        dim: typing.Hashable,
        key: typing.Any,
        value: xr.Dataset,
        existing: str,
        new: str,
    ) -> xr.DataArray:
        if dim not in da.dims:
            return da
        return da.pr.set(
            dim=dim, key=key, value=value[da.name], existing=existing, new=new
        )

    @alias_dims(["dim"])
    def set(
        self,
        dim: typing.Hashable,
        key: typing.Any,
        value: xr.Dataset,
        *,
        existing: str = "fillna_empty",
        new: str = "extend",
    ) -> xr.Dataset:
        """Set values, optionally expanding the given dimension as necessary.

        All data variables which have the given dimension are modified.
        The affected data variables are mutated using
        ``DataArray.pr.set(dim, key, value[name], existing=existing, new=new)``.

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
        existing: "fillna_empty", "error", "overwrite", or "fillna", optional
            How to handle existing keys. If ``existing="fillna_empty"`` (default), new
            values overwrite existing values only if all existing values are NaN.
            If ``existing="error"``, a ValueError is raised if any key already exists
            in the index. If ``existing="overwrite"``, new values overwrite current
            values for existing keys. If ``existing="fillna"``, the new values only
            overwrite NaN values for existing keys.
        new: "extend", or "error", optional
            How to handle new keys. If ``new="extend"`` (default), keys which do not
            exist so far are automatically inserted by extending the dimension.
            If ``new="error"``, a KeyError is raised if any key is not yet in the
            dimension.

        Examples
        --------
        >>> import pandas as pd
        >>> import xarray as xr
        >>> import numpy as np
        >>> area = ("area (ISO3)", ["COL", "MEX"])
        >>> time = ("time", pd.date_range("2000", "2003", freq="YS"))
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
        <xarray.Dataset> Size: ...
        Dimensions:      (area (ISO3): 2, time: 4)
        Coordinates:
          * area (ISO3)  (area (ISO3)) <U3 24B 'COL' 'MEX'
          * time         (time) datetime64[ns] 32B 2000-01-01 2001-01-01 ... 2003-01-01
        Data variables:
            CO2          (area (ISO3), time) float64 64B 0.0 1.0 2.0 3.0 2.0 3.0 4.0 5.0
            SF4          (area (ISO3), time) float64 64B 0.5 1.5 2.5 3.5 2.5 3.5 nan 5.5
        Attributes:
            area:     area (ISO3)

        Setting an existing value

        >>> ds.pr.set("area", "MEX", ds.pr.loc[{"area": "COL"}] * 20)
        Traceback (most recent call last):
        ...
        ValueError: Values {'MEX'} for 'area (ISO3)' already exist and contain data. ...
        >>> ds.pr.set(
        ...     "area", "MEX", ds.pr.loc[{"area": "COL"}] * 20, existing="overwrite"
        ... )
        <xarray.Dataset> Size: ...
        Dimensions:      (area (ISO3): 2, time: 4)
        Coordinates:
          * area (ISO3)  (area (ISO3)) object 16B 'COL' 'MEX'
          * time         (time) datetime64[ns] 32B 2000-01-01 2001-01-01 ... 2003-01-01
        Data variables:
            CO2          (area (ISO3), time) float64 64B 0.0 1.0 2.0 ... 20.0 40.0 60.0
            SF4          (area (ISO3), time) float64 64B 0.5 1.5 2.5 ... 30.0 50.0 70.0
        Attributes:
            area:     area (ISO3)

        Instead of overwriting existing values, you can also choose to only fill
        missing values

        >>> ds.pr.set("area", "MEX", ds.pr.loc[{"area": "COL"}] * 20, existing="fillna")
        <xarray.Dataset> Size: ...
        Dimensions:      (area (ISO3): 2, time: 4)
        Coordinates:
          * area (ISO3)  (area (ISO3)) object 16B 'COL' 'MEX'
          * time         (time) datetime64[ns] 32B 2000-01-01 2001-01-01 ... 2003-01-01
        Data variables:
            CO2          (area (ISO3), time) float64 64B 0.0 1.0 2.0 3.0 2.0 3.0 4.0 5.0
            SF4          (area (ISO3), time) float64 64B 0.5 1.5 2.5 ... 3.5 50.0 5.5
        Attributes:
            area:     area (ISO3)

        By default, existing values are only filled if all existing values are
        missing in all data variables

        >>> ds_partly_empty = ds.copy(deep=True)
        >>> ds_partly_empty["CO2"].pr.loc[{"area": "COL"}] = np.nan
        >>> ds_partly_empty["SF4"].pr.loc[{"area": "COL"}] = np.nan
        >>> ds_partly_empty
        <xarray.Dataset> Size: ...
        Dimensions:      (area (ISO3): 2, time: 4)
        Coordinates:
          * area (ISO3)  (area (ISO3)) <U3 24B 'COL' 'MEX'
          * time         (time) datetime64[ns] 32B 2000-01-01 2001-01-01 ... 2003-01-01
        Data variables:
            CO2          (area (ISO3), time) float64 64B nan nan nan nan 2.0 3.0 4.0 5.0
            SF4          (area (ISO3), time) float64 64B nan nan nan nan 2.5 3.5 nan 5.5
        Attributes:
            area:     area (ISO3)
        >>> ds_partly_empty.pr.set(
        ...     "area", "COL", ds_partly_empty.pr.loc[{"area": "MEX"}] * 10
        ... )
        <xarray.Dataset> Size: ...
        Dimensions:      (area (ISO3): 2, time: 4)
        Coordinates:
          * area (ISO3)  (area (ISO3)) object 16B 'COL' 'MEX'
          * time         (time) datetime64[ns] 32B 2000-01-01 2001-01-01 ... 2003-01-01
        Data variables:
            CO2          (area (ISO3), time) float64 64B 20.0 30.0 40.0 ... 3.0 4.0 5.0
            SF4          (area (ISO3), time) float64 64B 25.0 35.0 nan ... 3.5 nan 5.5
        Attributes:
            area:     area (ISO3)
        >>> # if even one value is non-nan, this fails by default
        >>> ds_partly_empty["SF4"].pr.loc[{"area": "COL", "time": "2001"}] = 2
        >>> ds_partly_empty.pr.set(
        ...     "area", "COL", ds_partly_empty.pr.loc[{"area": "MEX"}] * 10
        ... )
        Traceback (most recent call last):
        ...
        ValueError: Values {'COL'} for 'area (ISO3)' already exist and contain data. ...

        Introducing a new value uses the same syntax

        >>> ds.pr.set("area", "BOL", ds.pr.loc[{"area": "COL"}] * 20)
        <xarray.Dataset> Size: ...
        Dimensions:      (area (ISO3): 3, time: 4)
        Coordinates:
          * area (ISO3)  (area (ISO3)) object 24B 'BOL' 'COL' 'MEX'
          * time         (time) datetime64[ns] 32B 2000-01-01 2001-01-01 ... 2003-01-01
        Data variables:
            CO2          (area (ISO3), time) float64 96B 0.0 20.0 40.0 ... 3.0 4.0 5.0
            SF4          (area (ISO3), time) float64 96B 10.0 30.0 50.0 ... 3.5 nan 5.5
        Attributes:
            area:     area (ISO3)

        If you don't want to automatically extend the dimensions with new values, you
        can request checking that all keys already exist using ``new="error"``:

        >>> ds.pr.set("area", "BOL", ds.pr.loc[{"area": "COL"}] * 20, new="error")
        Traceback (most recent call last):
        ...
        KeyError: "Values {'BOL'} not in 'area (ISO3)', use new='extend' to automatic...

        Note that data variables which do not contain the specified dimension are
        ignored

        >>> ds["population"] = xr.DataArray([1e6, 1.2e6, 1.3e6, 1.4e6], coords=(time,))
        >>> ds
        <xarray.Dataset> Size: ...
        Dimensions:      (area (ISO3): 2, time: 4)
        Coordinates:
          * area (ISO3)  (area (ISO3)) <U3 24B 'COL' 'MEX'
          * time         (time) datetime64[ns] 32B 2000-01-01 2001-01-01 ... 2003-01-01
        Data variables:
            CO2          (area (ISO3), time) float64 64B 0.0 1.0 2.0 3.0 2.0 3.0 4.0 5.0
            SF4          (area (ISO3), time) float64 64B 0.5 1.5 2.5 3.5 2.5 3.5 nan 5.5
            population   (time) float64 32B 1e+06 1.2e+06 1.3e+06 1.4e+06
        Attributes:
            area:     area (ISO3)
        >>> ds.pr.set("area", "BOL", ds.pr.loc[{"area": "COL"}] * 20)
        <xarray.Dataset> Size: ...
        Dimensions:      (area (ISO3): 3, time: 4)
        Coordinates:
          * area (ISO3)  (area (ISO3)) object 24B 'BOL' 'COL' 'MEX'
          * time         (time) datetime64[ns] 32B 2000-01-01 2001-01-01 ... 2003-01-01
        Data variables:
            CO2          (area (ISO3), time) float64 96B 0.0 20.0 40.0 ... 3.0 4.0 5.0
            SF4          (area (ISO3), time) float64 96B 10.0 30.0 50.0 ... 3.5 nan 5.5
            population   (time) float64 32B 1e+06 1.2e+06 1.3e+06 1.4e+06
        Attributes:
            area:     area (ISO3)

        Returns
        -------
        ds : xr.Dataset
            modified Dataset
        """
        if not isinstance(value, xr.Dataset):
            raise TypeError(f"value must be a Dataset, not {type(value)}")

        return self._ds.map(
            self._set_apply,
            keep_attrs=True,
            dim=dim,
            key=key,
            value=value,
            existing=existing,
            new=new,
        )
