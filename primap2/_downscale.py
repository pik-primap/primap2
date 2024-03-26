from collections.abc import Hashable, Sequence

import xarray as xr

from ._accessor_base import BaseDataArrayAccessor, BaseDatasetAccessor
from ._aggregate import select_no_scalar_dimension
from ._units import ureg


class DataArrayDownscalingAccessor(BaseDataArrayAccessor):
    def downscale_timeseries(
        self,
        *,
        dim: Hashable,
        basket: Hashable,
        basket_contents: Sequence[Hashable],
        check_consistency: bool = True,
        sel: dict[Hashable, Sequence] | None = None,
        skipna_evaluation_dims: None | Sequence[Hashable] = None,
        skipna: bool = True,
        tolerance: float = 0.01,
    ) -> xr.DataArray:
        """Downscale timeseries along a dimension using a basket defined on a
        broader timeseries.

        This is useful if you have data for many points in time for a total, for example
        the entire Energy sector, and higher-resolution data (e.g. fossil and non-fossil
        energies separately) for only a few points in time. In the example, the Energy
        sector is the ``basket`` and fossil and non-fossil energies are the basket
        contents.
        From any time points where all the basket contents are known, the
        relative shares of the basket contents are determined, and then interpolated
        linearly and extrapolated constantly to the full timeseries. The shares are then
        used to downscale the basket to its contents, which is used to fill gaps in the
        timeseries of the basket contents.

        Parameters
        ----------
        dim: str
          The name of the dimension which contains the basket and its contents, has to
          be one of the dimensions in ``da.dims``.
        basket: str
          The name of the super-category for which values are known at higher temporal
          resolution and/or for a wider range. A value from ``da[dimension]``.
        basket_contents: list of str
          The name of the sub-categories. The sum of all sub-categories equals the
          basket. Values from ``da[dimension]``.
        check_consistency: bool, default True
          If for all points where the basket and all basket_contents are defined,
          it should be checked if the sum of the basket_contents actually equals
          the basket. A ``ValueError`` is raised if the consistency check fails.
        sel: Selection dict, optional
          If the downscaling should only be done on a subset of the Dataset while
          retaining all other values unchanged, give a selection dictionary. The
          downscaling will be done on ``da.loc[sel]``.
        skipna_evaluation_dims: list of str, optional
          Dimensions which should be evaluated to determine if NA values should be
          skipped entirely if missing fully. By default, no NA values are skipped.
        skipna: bool, default True
          If true it will be passed on to xarray's ds.sum function with min_count=1
          for the calculation of the basket.
          The effect is that NA values in a sum will be ignored and treated as zero
          in the sum unless all values are NA which results in NA.
        tolerance: float
          If given it overrides the default tolerance for deviations of sums of
          individual timeseries to given aggregate timeseries. Default is 0.01 (1%)

        Returns
        -------
        downscaled: xr.DataArray
        """
        da_sel = select_no_scalar_dimension(self._da, sel)

        basket_contents_da = da_sel.loc[{dim: basket_contents}]
        basket_da = da_sel.loc[{dim: basket}]

        if skipna_evaluation_dims is not None:
            if skipna:
                raise ValueError(
                    "Only one of 'skipna' and 'skipna_evaluation_dims' may be supplied, not"
                    " both."
                )
            else:
                skipna = None

        basket_sum = basket_contents_da.pr.sum(
            dim=dim,
            skipna=skipna,
            min_count=1,
            skipna_evaluation_dims=skipna_evaluation_dims,
        )

        if check_consistency:
            deviation: xr.DataArray = abs(basket_da / basket_sum - 1)
            devmax = float(deviation.max().pint.dequantify().data)
            if devmax > tolerance:
                raise ValueError(
                    f"Sum of the basket_contents {basket_contents!r} deviates"
                    f" {devmax * 100} % from the basket"
                    f" {basket!r}, which is more than the allowed 1 %. "
                    "To continue regardless, set check_consistency=False."
                )

        # inter- and extrapolate
        shares: xr.DataArray = (
            (basket_contents_da / basket_sum)
            .pint.to("")
            .pint.dequantify()
            .interpolate_na(dim="time", method="linear")
            .ffill(dim="time")
            .bfill(dim="time")
        )

        downscaled: xr.DataArray = basket_da * shares

        return self._da.fillna(downscaled)


class DatasetDownscalingAccessor(BaseDatasetAccessor):
    def downscale_timeseries(
        self,
        *,
        dim: Hashable,
        basket: Hashable,
        basket_contents: Sequence[Hashable],
        check_consistency: bool = True,
        sel: dict[Hashable, Sequence] | None = None,
        skipna_evaluation_dims: Sequence[Hashable] | None = None,
        skipna: bool = True,
        tolerance: float = 0.01,
    ) -> xr.Dataset:
        """Downscale timeseries along a dimension using a basket defined on a
        broader timeseries.

        This is useful if you have data for many points in time for a total, for example
        the entire Energy sector, and higher-resolution data (e.g. fossil and non-fossil
        energies separately) for only a few points in time. In the example, the Energy
        sector is the ``basket`` and fossil and non-fossil energies are the basket
        contents.
        From any time points where all the basket contents are known, the
        relative shares of the basket contents are determined, and then interpolated
        linearly and extrapolated constantly to the full timeseries. The shares are then
        used to downscale the basket to its contents, which is used to fill gaps in the
        timeseries of the basket contents.

        Parameters
        ----------
        dim: str
          The name of the dimension which contains the basket and its contents, has to
          be one of the dimensions in ``ds.dims``.
        basket: str
          The name of the super-category for which values are known at higher temporal
          resolution and/or for a wider range. A value from ``ds[dimension]``.
        basket_contents: list of str
          The name of the sub-categories. The sum of all sub-categories equals the
          basket. Values from ``ds[dimension]``.
        check_consistency: bool, default True
          If for all points where the basket and all basket_contents are defined,
          it should be checked if the sum of the basket_contents actually equals
          the basket. A ``ValueError`` is raised if the consistency check fails.
        sel: Selection dict, optional
          If the downscaling should only be done on a subset of the Dataset while
          retaining all other values unchanged, give a selection dictionary. The
          downscaling will be done on ``ds.loc[sel]``.
        skipna_evaluation_dims: list of str, optional
          Dimensions which should be evaluated to determine if NA values should be
          skipped entirely if missing fully. By default, no NA values are skipped.
        skipna: bool, default True
          If true it will be passed on to xarray's ds.sum function with min_count=1 for
          the calculation of the basket.
          The effect is that NA values in a sum will be ignored and treated as zero
          in the sum unless all values are NA which results in NA.
        tolerance: float
          If given it overrides the default tolerance for deviations of sums of
          individual timeseries to given aggregate timeseries. Default is 0.01 (1%)

        Notes
        -----
        To downscale along the entity dimension, i.e. to downscale gases, see
        :py:meth:`downscale_gas_timeseries`, which handles gwp conversions
        appropriately.

        Returns
        -------
        downscaled: xr.Dataset
        """
        ds_sel = select_no_scalar_dimension(self._ds, sel)

        basket_contents_ds = ds_sel.loc[{dim: basket_contents}]
        basket_ds = ds_sel.loc[{dim: basket}]

        if skipna_evaluation_dims is not None:
            if skipna:
                raise ValueError(
                    "Only one of 'skipna' and 'skipna_evaluation_dims' may be supplied, not"
                    " both."
                )
            else:
                skipna = None

        basket_sum = basket_contents_ds.pr.sum(
            dim=dim,
            skipna=skipna,
            min_count=1,
            skipna_evaluation_dims=skipna_evaluation_dims,
        )

        if check_consistency:
            deviation = abs(basket_ds / basket_sum - 1)
            devmax = deviation.to_array().max()
            if devmax > tolerance:
                raise ValueError(
                    f"Sum of the basket_contents {basket_contents!r} deviates"
                    f" {devmax * 100} % from the basket"
                    f" {basket!r}, which is more than the allowed 1 %. "
                    "To continue regardless, set check_consistency=False."
                )

        # inter- and extrapolate
        shares: xr.Dataset = (
            (basket_contents_ds / basket_sum)
            .pint.to({x: "" for x in basket_contents_ds.keys()})
            .pint.dequantify()
            .interpolate_na(dim="time", method="linear")
            .ffill(dim="time")
            .bfill(dim="time")
        )

        downscaled: xr.Dataset = basket_ds * shares

        return self._ds.fillna(downscaled)

    def downscale_gas_timeseries(
        self,
        *,
        basket: Hashable,
        basket_contents: Sequence[Hashable],
        check_consistency: bool = True,
        sel: dict[Hashable, Sequence] | None = None,
        skipna_evaluation_dims: Sequence[Hashable] | None = None,
        skipna: bool = True,
        tolerance: float = 0.01,
    ) -> xr.Dataset:
        """Downscale a gas basket defined on a broader timeseries to its contents
        known on fewer time points.

        This is useful if you have data for many points in time for a gas basket, for
        example KYOTOGHG, and higher-resolution data (e.g. the individual green house
        gases included in KYOTOGHG) for only a few points in time.
        From any time points where all the basket contents are known, the relative
        shares of the basket contents are determined, and then interpolated linearly
        and extrapolated constantly to the full timeseries. The shares are then
        used to downscale the basket to its contents, which is used to fill gaps in the
        timeseries of the basket contents.
        Basket contents are converted to the units of the basket for downscaling, and
        converted back afterwards; for both conversions the gwp conversions of the
        basket are used.

        Parameters
        ----------
        basket: str
          The name of the gas basket for which values are known at higher temporal
          resolution and/or for a wider range. A value from ``ds.keys()``.
        basket_contents: list of str
          The name of the gases in the gas basket. The sum of all basket_contents
          equals the basket. Values from ``ds.keys()``.
        check_consistency: bool, default True
          If for all points where the basket and all basket_contents are defined,
          it should be checked if the sum of the basket_contents actually equals
          the basket. A ``ValueError`` is raised if the consistency check fails.
        sel: Selection dict, optional
          If the downscaling should only be done on a subset of the Dataset while
          retaining all other values unchanged, give a selection dictionary. The
          downscaling will be done on ``ds.loc[sel]``.
        skipna_evaluation_dims: list of str, optional
          Dimensions which should be evaluated to determine if NA values should be
          skipped entirely if missing fully. By default, no NA values are skipped.
        skipna: bool, optional
          If true it will be passed on to xarray's ds.sum function with min_count=1 for
          the calculation of the basket.
          The effect is that NA values in a sum will be ignored and treated as zero
          in the sum unless all values are NA which results in NA.
        tolerance: float
          If given it overrides the default tolerance for deviations of sums of
          individual timeseries to given aggregate timeseries. Default is 0.01 (1%)

        Returns
        -------
        downscaled: xr.Dataset
        """
        ds_sel = select_no_scalar_dimension(self._ds, sel)

        basket_contents_converted = xr.Dataset()
        da_basket = ds_sel[basket]
        for var in basket_contents:
            da: xr.DataArray = ds_sel[var]
            basket_contents_converted[var] = da.pr.convert_to_gwp_like(like=da_basket)

        if skipna_evaluation_dims is not None:
            if skipna:
                raise ValueError(
                    "Only one of 'skipna' and 'skipna_evaluation_dims' may be supplied, not"
                    " both."
                )
            else:
                skipna = None

        basket_sum = basket_contents_converted.pr.sum(
            dim="entity",
            skipna=skipna,
            min_count=1,
            skipna_evaluation_dims=skipna_evaluation_dims,
        )

        if check_consistency:
            deviation = abs(da_basket / basket_sum - 1)
            devmax = deviation.max().item()
            if devmax > tolerance:
                raise ValueError(
                    f"Sum of the basket_contents {basket_contents!r} deviates"
                    f" {devmax * 100} % from the basket"
                    f" {basket!r}, which is more than the allowed 1 %. "
                    f" {deviation}"
                    "To continue regardless, set check_consistency=False."
                )

        # inter- and extrapolate
        shares = (
            (basket_contents_converted / basket_sum)
            .pint.to({x: "" for x in basket_contents_converted.keys()})
            .pint.dequantify()
            .interpolate_na(dim="time", method="linear")
            .ffill(dim="time")
            .bfill(dim="time")
        )

        downscaled = da_basket * shares

        downscaled_converted = xr.Dataset()

        with ureg.context(da_basket.attrs["gwp_context"]):
            for var in basket_contents:
                downscaled_converted[var] = ds_sel[var].fillna(
                    downscaled[var].pint.to(ds_sel[var].pint.units)
                )

        return self._ds.fillna(downscaled_converted)
