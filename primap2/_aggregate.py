from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import xarray as xr

from ._accessor_base import BaseDataArrayAccessor, BaseDatasetAccessor
from ._alias_selection import alias_dims
from ._types import DatasetOrDataArray, DimOrDimsT


def dim_names(obj: DatasetOrDataArray):
    if isinstance(obj.dims, tuple):
        return obj.dims
    else:
        return obj.dims.keys()  # type: ignore


def select_no_scalar_dimension(
    obj: DatasetOrDataArray, sel: Mapping[Hashable, Any] | None
) -> DatasetOrDataArray:
    """Select but raise an error if the selection produces a new scalar dimensions.

    This can be used to guard against later broadcasting of the selection.

    Parameters
    ----------
    obj: xr.Dataset or xr.DataArray
    sel: selection dictionary

    Returns
    -------
    selection : same type as obj
    """
    if sel is None:
        return obj
    selection: DatasetOrDataArray = obj.loc[sel]
    if dim_names(obj) != dim_names(selection):
        raise ValueError(
            "The dimension of the selection doesn't match the dimension of the "
            "orginal dataset. Likely you used a selection casting to a scalar "
            "dimension, like sel={'axis': 'value'}. Please use "
            "sel={'axis': ['value']} instead."
        )
    return selection


class DataArrayAggregationAccessor(BaseDataArrayAccessor):
    def _reduce_dim(
        self, dim: DimOrDimsT | None, reduce_to_dim: DimOrDimsT | None
    ) -> DimOrDimsT | None:
        if dim is not None and reduce_to_dim is not None:
            raise ValueError(
                "Only one of 'dim' and 'reduce_to_dim' may be supplied, not both."
            )

        if dim is None and reduce_to_dim is not None:
            if isinstance(reduce_to_dim, str):
                reduce_to_dim = [reduce_to_dim]
            dim = set(self._da.dims) - set(reduce_to_dim)

        return dim

    @alias_dims(["dim"], wraps=xr.DataArray.any)
    def any(self, *args, **kwargs):
        return self._da.any(*args, **kwargs)

    @alias_dims(["dim", "reduce_to_dim"])
    def count(
        self,
        dim: DimOrDimsT | None = None,
        *,
        reduce_to_dim: DimOrDimsT | None = None,
        keep_attrs: bool = True,
        **kwargs,
    ) -> xr.DataArray:
        """Reduce this array by counting along some dimension(s).

        By default, works like da.count(), but with some additional features:

        1. Dimension aliases can be used instead of full dimension names everywhere.
        2. Instead of specifying the dimension(s) to reduce via ``dim``, you can specify
           the dimensions that the result should have via ``reduce_to_dim``. Then,
           `count` will be applied along all other dimensions.

        Parameters
        ----------
        dim: str or list of str, optional
          Dimension(s) over which to apply `count`. Only one of ``dim`` and
          ``reduce_to_dim`` arguments can be supplied. If neither is supplied, then
          the count is calculated over all dimensions.
        reduce_to_dim: str or list of str, optional
          Dimension(s) of the result. Only one of ``dim`` and ``reduce_to_dim``
          arguments can be supplied. Supplying ``reduce_to_dim="dim_1"`` is therefore
          equivalent to giving ``dim=set(da.dims) - {"dim_1"}``, but more legible.
        keep_attrs: bool, optional
          Keep the attr metadata (default True).
        **kwargs: dict
          Additional keyword arguments are passed directly to xarray's da.count().

        Returns
        -------
        counted : xr.DataArray
        """
        dim = self._reduce_dim(dim, reduce_to_dim)
        return self._da.count(dim=dim, keep_attrs=keep_attrs, **kwargs)

    @alias_dims(["dim", "reduce_to_dim", "skipna_evaluation_dims"])
    def sum(
        self,
        dim: DimOrDimsT | None = None,
        *,
        reduce_to_dim: DimOrDimsT | None = None,
        skipna: bool | None = None,
        skipna_evaluation_dims: DimOrDimsT | None = None,
        keep_attrs: bool = True,
        min_count: int | None = None,
    ) -> xr.DataArray:
        """Reduce this DataArray's data by applying `sum` along some dimension(s).

        By default, works like da.sum(), but has additional features:

        1. Dimension aliases can be used instead of full dimension names everywhere.
        2. Instead of specifying the dimension(s) to reduce via ``dim``, you can specify
           the dimensions that the result should have via ``reduce_to_dim``. Then,
           `sum` will be applied along all other dimensions.
        3. You can specify ``skipna_evaluation_dims`` to skip NA values only if all
           values along the given dimension(s) are NA. Example: If you have a data array
           with the dimensions ``time`` and ``position``, summing over ``time`` with the
           evaluation dimension ``position`` will skip only those values where all
           values with the same ``position`` are NA.

        ``skipna`` and ``min_count`` work like in the :py:func:`xr.sum` function. The behaviour
        of primap1 is reproduced by ``skipna=True, min_count=1``.

        Parameters
        ----------
        dim: str or list of str, optional
          Dimension(s) over which to apply `sum`. Only one of ``dim`` and
          ``reduce_to_dim`` arguments can be supplied. If neither is supplied, then
          the sum is calculated over all dimensions.
        reduce_to_dim: str or list of str, optional
          Dimension(s) of the result. Only one of ``dim`` and ``reduce_to_dim``
          arguments can be supplied. Supplying ``reduce_to_dim="dim_1"`` is therefore
          equivalent to giving ``dim=set(da.dims) - {"dim_1"}``, but more legible.
        skipna: bool, optional
          If True, skip missing values (as marked by NaN). By default, only
          skips missing values for float dtypes; other dtypes either do not
          have a sentinel missing value (int) or skipna=True has not been
          implemented (object, datetime64 or timedelta64).
        skipna_evaluation_dims: str or list of str, optional
          Dimension(s) to evaluate along to determine if values should be skipped.
          Only one of ``skipna`` and ``skipna_evaluation_dims`` can be supplied.
          If all values along the specified dimensions are NA, the values are skipped,
          other NA values are not skipped and will lead to NA in the corresponding
          result.
        keep_attrs: bool, optional
          Keep the attr metadata (default True).
        min_count: int (default None, but set to 1 if skipna=True)
          The minimal number of non-NA values in a sum that is necessary for a non-NA
          result. This only has an effect if ``skipna=True``. As an example: you sum data
          for a region for a certain sector, gas and year. If ``skipna=False``,
          all countries in the region need to have non-NA data for that sector, gas,
          year combination. If ``skipna=True`` and ``min_count=1`` then one country with
          non-NA data is enough for a non-NA result. All NA values will be treated as
          zero. If ``min_count=0`` all NA values will be treated as zero
          also if there is no single non-NA value in the data that is to be summed.


        Returns
        -------
        summed : xr.DataArray
        """
        dim = self._reduce_dim(dim, reduce_to_dim)

        if skipna is not None and skipna_evaluation_dims is not None:
            raise ValueError(
                "Only one of 'skipna' and 'skipna_evaluation_dims' may be supplied, not"
                " both."
            )

        if skipna_evaluation_dims is not None:
            skipna = False
            da = self.fill_all_na(dim=skipna_evaluation_dims, value=0)
        else:
            da = self._da
            if skipna:
                if min_count is None:
                    min_count = 1

        return da.sum(
            dim=dim, skipna=skipna, keep_attrs=keep_attrs, min_count=min_count
        )

    @alias_dims(["dim"])
    def fill_all_na(self, dim: Iterable[Hashable] | str, value=0) -> xr.DataArray:
        """Fill NA values only where all values along the specified dimension(s) are NA.

        Example: having a data array with dimensions ``time`` and ``position``,
        filling it along ``time`` will only fill those values where all points that
        differ only in their ``time`` are NA, i.e. those points where all points with
        the same ``position`` are NA.

        Parameters
        ----------
        dim: str or list of str
          Dimension(s) to evaluate. NA values are only filled if all values along these
          dimension(s) are also NA.
        value: optional
          Fill value to use. Default: 0.

        Returns
        -------
        filled : xr.DataArray
        """
        if not dim:
            return self._da
        else:
            return self._da.where(~np.isnan(self._da).all(dim=dim), value)


class DatasetAggregationAccessor(BaseDatasetAccessor):
    @staticmethod
    def _apply_fill_all_na(
        da: xr.DataArray, dim: Iterable[Hashable] | str, value
    ) -> xr.DataArray:
        if isinstance(dim, str):
            dim = [dim]
        # dims which don't exist for a particular data variable can be excluded for that
        # data variable because if a value is NA in the other dimensions it is NA for
        # all values of a non-existing dimension
        dim = [x for x in dim if x in da.dims]
        return da.pr.fill_all_na(dim=dim, value=value)

    def _all_vars_all_dimensions(self):
        return (
            np.array([len(var.dims) for var in self._ds.values()])
            == [len(self._ds.dims)]
        ).all()

    @alias_dims(["dim"])
    def fill_all_na(self, dim: DimOrDimsT, value=0) -> xr.Dataset:
        """Fill NA values only where all values along the specified dimension(s) are NA.

        Example: if you have a Dataset with dimensions ``time`` and ``positions``,
        filling it along ``time`` will only fill those values where all time-points are
        NA in each DataArray.

        Parameters
        ----------
        dim: str or list of str
          Dimension(s) to evaluate. NA values are only filled if all values along these
          dimension(s) are also NA.
        value: optional
          Fill value to use. Default: 0.

        Returns
        -------
        filled : xr.Dataset
        """
        return self._ds.map(
            self._apply_fill_all_na, dim=dim, value=value, keep_attrs=True
        )

    def _reduce_dim(
        self, dim: DimOrDimsT | None, reduce_to_dim: DimOrDimsT | None
    ) -> Iterable[Hashable] | None:
        if dim is not None and reduce_to_dim is not None:
            raise ValueError(
                "Only one of 'dim' and 'reduce_to_dim' may be supplied, not both."
            )

        if dim is None:
            if reduce_to_dim is not None:
                if isinstance(reduce_to_dim, str):
                    reduce_to_dim = [reduce_to_dim]
                dims = set(self._ds.dims)
                dims.add("entity")
                dim = dims - set(reduce_to_dim)

        if isinstance(dim, str):
            dim = [dim]

        return dim

    @alias_dims(["dim"], wraps=xr.Dataset.any)
    def any(self, *args, **kwargs):
        return self._ds.any(*args, **kwargs)

    @alias_dims(["dim", "reduce_to_dim"], additional_allowed_values=["entity"])
    def count(
        self,
        dim: DimOrDimsT | None = None,
        *,
        reduce_to_dim: DimOrDimsT | None = None,
        keep_attrs: bool = True,
        **kwargs,
    ) -> DatasetOrDataArray:
        """Reduce this Dataset by counting along some dimension(s).

        By default, works like ds.count(), but with some additional features:

        1. Dimension aliases can be used instead of full dimension names everywhere.
        2. Instead of specifying the dimension(s) to reduce via ``dim``, you can specify
           the dimensions that the result should have via ``reduce_to_dim``. Then,
           `count` will be applied along all other dimensions (including ``entity``).
        3. If you specify ``entity`` in "dim", the Dataset is converted to a DataArray
           and counted along the data variables.

        Parameters
        ----------
        dim: str or list of str, optional
          Dimension(s) over which to apply `count`. Only one of ``dim`` and
          ``reduce_to_dim`` arguments can be supplied. If neither is supplied, then
          the count is calculated over all dimensions. Use "entity" to convert to a
          DataArray and sum along the data variables.
        reduce_to_dim: str or list of str, optional
          Dimension(s) of the result. Only one of ``dim`` and ``reduce_to_dim``
          arguments can be supplied. Supplying ``reduce_to_dim="dim_1"`` is therefore
          equivalent to giving ``dim=set(da.dims) + {"entity"} - {"dim_1"}``, but more
          legible.
        keep_attrs: bool, optional
          Keep the attr metadata (default True).
        **kwargs: dict
          Additional keyword arguments are passed directly to xarray's ds.count().

        Returns
        -------
        counted : xr.Dataset or xr.DataArray if "entity" in dims.
        """
        dim = self._reduce_dim(dim, reduce_to_dim)

        if dim is not None and "entity" in dim:
            ndim = set(dim) - {"entity"}

            ds = self._ds.count(dim=ndim, keep_attrs=keep_attrs, **kwargs)

            if not ds.pr._all_vars_all_dimensions():
                raise NotImplementedError(
                    "Counting along the entity dimension is only supported "
                    "when all entities share the dimensions remaining after counting."
                )

            return ds.to_array("entity").sum(dim="entity", keep_attrs=True)
        else:
            return self._ds.count(dim=dim, keep_attrs=keep_attrs, **kwargs)

    @alias_dims(["dim", "reduce_to_dim"], additional_allowed_values=["entity"])
    def sum(
        self,
        dim: DimOrDimsT | None = None,
        *,
        reduce_to_dim: DimOrDimsT | None = None,
        skipna: bool | None = None,
        skipna_evaluation_dims: DimOrDimsT | None = None,
        keep_attrs: bool = True,
        min_count: int | None = None,
    ) -> DatasetOrDataArray:
        """Reduce this Dataset's data by applying `sum` along some dimension(s).

        By default, works like ds.sum(), but has additional features:

        1. Dimension aliases can be used instead of full dimension names everywhere.
        2. Instead of specifying the dimension(s) to reduce via ``dim``, you can specify
           the dimensions that the result should have via ``reduce_to_dim``. Then,
           `sum` will be applied along all other dimensions (including ``entity``).
        3. You can specify ``skipna_evaluation_dims`` to skip NA values only if all
           values along the given dimension(s) are NA. Example: If you have a data array
           with the dimensions ``time`` and ``position``, summing over ``time`` with the
           evaluation dimension ``position`` will skip only those values where all
           values with the same ``position`` are NA.
        4. If you specify ``entity`` in "dim", the Dataset is converted to a DataArray
           and summed along the data variables (which will only work if the units of
           the DataArrays are compatible).

        ``skipna`` and ``min_count`` work like in the :py:func:`xr.sum` function. The behaviour
        of primap1 is reproduced by ``skipna=True, min_count=1``.

        Parameters
        ----------
        dim: str or list of str, optional
          Dimension(s) over which to apply `sum`. Only one of ``dim`` and
          ``reduce_to_dim`` arguments can be supplied. If neither is supplied, then
          the sum is calculated over all dimensions. Use "entity" to convert to a
          DataArray and sum along the data variables.
        reduce_to_dim: str or list of str, optional
          Dimension(s) of the result. Only one of ``dim`` and ``reduce_to_dim``
          arguments can be supplied. Supplying ``reduce_to_dim="dim_1"`` is therefore
          equivalent to giving ``dim=set(da.dims) + {"entity"} - {"dim_1"}``, but more
          legible.
        skipna: bool, optional
          If True (default), skip missing values (as marked by NaN). By default, only
          skips missing values for float dtypes; other dtypes either do not
          have a sentinel missing value (int) or skipna=True has not been
          implemented (object, datetime64 or timedelta64).
        skipna_evaluation_dims: str or list of str, optional
          Dimension(s) to evaluate along to determine if values should be skipped.
          Only one of ``skipna`` and ``skipna_evaluation_dims`` can be supplied.
          If all values along the specified dimensions are NA, the values are skipped,
          other NA values are not skipped and will lead to NA in the corresponding
          result.
        keep_attrs: bool, optional
          Keep the attr metadata (default True).
        min_count: int (default None, but set to 1 if skipna=True)
          The minimal number of non-NA values in a sum that is necessary for a non-NA
          result. This only has an effect if ``skipna=True``. As an example: you sum data
          for a region for a certain sector, gas and year. If ``skipna=False``,
          all countries in the region need to have non-NA data for that sector, gas,
          year combination. If ``skipna=True`` and ``min_count=1`` then one country with
          non-NA data is enough for a non-NA result. All NA values will be treated as
          zero. If ``min_count=0`` all NA values will be treated as zero
          also if there is no single non-NA value in the data that is to be summed.

        Returns
        -------
        summed : xr.DataArray
        """
        dim = self._reduce_dim(dim, reduce_to_dim)

        if skipna is not None and skipna_evaluation_dims is not None:
            raise ValueError(
                "Only one of 'skipna' and 'skipna_evaluation_dims' may be supplied, not"
                " both."
            )

        if skipna_evaluation_dims is not None:
            skipna = False
            ds = self.fill_all_na(dim=skipna_evaluation_dims, value=0)
        else:
            ds = self._ds
            if skipna:
                if min_count is None:
                    min_count = 1

        if dim is not None and "entity" in dim:
            ndim = set(dim) - {"entity"}

            ds = ds.sum(
                dim=ndim, skipna=skipna, keep_attrs=keep_attrs, min_count=min_count
            )

            if not ds.pr._all_vars_all_dimensions():
                raise NotImplementedError(
                    "Summing along the entity dimension is only supported "
                    "when all entities share the dimensions remaining after summing."
                )
            return ds.to_array("entity").sum(
                dim="entity", skipna=skipna, keep_attrs=keep_attrs, min_count=min_count
            )
        else:
            return ds.sum(
                dim=dim, skipna=skipna, keep_attrs=keep_attrs, min_count=min_count
            )

    def gas_basket_contents_sum(
        self,
        *,
        basket: str,
        basket_contents: Sequence[str],
        skipna: bool | None = None,
        skipna_evaluation_dims: DimOrDimsT | None = None,
        min_count: int | None = None,
    ) -> xr.DataArray:
        """The sum of gas basket contents converted using the global warming potential
        of the gas basket.

        Parameters
        ----------
        basket: str
          The name of the gas basket. A value from `ds.keys()`.
        basket_contents: list of str
          The name of the gases in the gas basket. The sum of all basket_contents
          equals the basket. Values from `ds.keys()`.
        skipna: bool, optional
          If True (default), skip missing values (as marked by NaN). By default, only
          skips missing values for float dtypes; other dtypes either do not
          have a sentinel missing value (int) or skipna=True has not been
          implemented (object, datetime64 or timedelta64).
        skipna_evaluation_dims: str or list of str, optional
          Dimension(s) to evaluate along to determine if values should be skipped.
          Only one of ``skipna`` and ``skipna_evaluation_dims`` can be supplied.
          If all values along the specified dimensions are NA, the values are skipped,
          other NA values are not skipped and will lead to NA in the corresponding
          result.
        min_count: int (default None, but set to 1 if skipna=True)
          The minimal number of non-NA values in a sum that is necessary for a non-NA
          result. This only has an effect if ``skipna=True``. As an example: you sum data
          for a region for a certain sector, gas and year. If ``skipna=False``,
          all countries in the region need to have non-NA data for that sector, gas,
          year combination. If ``skipna=True`` and ``min_count=1`` then one country with
          non-NA data is enough for a non-NA result. All NA values will be treated as
          zero. If ``min_count=0`` all NA values will be treated as zero
          also if there is no single non-NA value in the data that is to be summed.

        Returns
        -------
        summed : xr.DataArray
        """

        basket_contents_converted = xr.Dataset()
        basket_da = self._ds[basket]
        for var in basket_contents:
            da: xr.DataArray = self._ds[var]
            basket_contents_converted[var] = da.pr.convert_to_gwp_like(like=basket_da)

        da = basket_contents_converted.pr.sum(
            dim="entity",
            skipna_evaluation_dims=skipna_evaluation_dims,
            skipna=skipna,
            min_count=min_count,
        )
        da.attrs["gwp_context"] = basket_da.attrs["gwp_context"]
        da.attrs["entity"] = basket_da.attrs["entity"]
        da.name = basket_da.name
        return da

    def fill_na_gas_basket_from_contents(
        self,
        *,
        basket: str,
        basket_contents: Sequence[str],
        sel: Mapping[Hashable, Sequence] | None = None,
        skipna: bool | None = None,
        skipna_evaluation_dims: DimOrDimsT | None = None,
        min_count: int | None = None,
    ) -> xr.DataArray:
        """Fill NA values in a gas basket using the sum of its contents.

        To calculate the sum of the gas basket contents, the global warming potential
        context defined on the gas basket will be used.

        Parameters
        ----------
        basket: str
          The name of the gas basket. A value from `ds.keys()`.
        basket_contents: list of str
          The name of the gases in the gas basket. The sum of all basket_contents
          equals the basket. Values from `ds.keys()`.
        sel: Selection dict, optional
          If the filling should only be done on a subset of the Dataset while
          retaining all other values unchanged, give a selection dictionary. The
          filling will be done on `ds.loc[sel]`.
        skipna: bool, optional
          If True (default), skip missing values (as marked by NaN). By default, only
          skips missing values for float dtypes; other dtypes either do not
          have a sentinel missing value (int) or skipna=True has not been
          implemented (object, datetime64 or timedelta64).
        skipna_evaluation_dims: str or list of str, optional
          Dimension(s) to evaluate along to determine if values should be skipped.
          Only one of ``skipna`` and ``skipna_evaluation_dims`` can be supplied.
          If all values along the specified dimensions are NA, the values are skipped,
          other NA values are not skipped and will lead to NA in the corresponding
          result.
        min_count: int (default None, but set to 1 if skipna=True)
          The minimal number of non-NA values in a sum that is necessary for a non-NA
          result. This only has an effect if ``skipna=True``. As an example: you sum data
          for a region for a certain sector, gas and year. If ``skipna=False``,
          all countries in the region need to have non-NA data for that sector, gas,
          year combination. If ``skipna=True`` and ``min_count=1`` then one country with
          non-NA data is enough for a non-NA result. All NA values will be treated as
          zero. If ``min_count=0`` all NA values will be treated as zero
          also if there is no single non-NA value in the data that is to be summed.

        Returns
        -------
        filled : xr.DataArray
        """
        ds_sel = select_no_scalar_dimension(self._ds, sel)
        return self._ds[basket].fillna(
            ds_sel.pr.gas_basket_contents_sum(
                basket=basket,
                basket_contents=basket_contents,
                skipna_evaluation_dims=skipna_evaluation_dims,
                skipna=skipna,
                min_count=min_count,
            )
        )
