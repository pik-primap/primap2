"""Tests for _convert.py"""

import climate_categories as cc
import pytest
import xarray as xr

import primap2


def test_convert_ipcc(empty_ds: xr.Dataset):
    # build a DA categorized by IPCC1996 and with 1 everywhere so results are easy
    # to see
    da = empty_ds["CO2"]
    da = da.expand_dims({"category (IPCC1996)": list(cc.IPCC1996.keys())})
    da = da.expand_dims({"source (gas)": list(cc.gas.keys())})
    arr = da.data.copy()
    arr[:] = 1 * primap2.ureg("Gg CO2 / year")
    da.data = arr

    with pytest.raises(ValueError, match="The conversion uses auxiliary categories"):
        da.pr.convert("category", "IPCC2006", sum_rule="extensive")

    result = da.pr.convert(
        "category",
        "IPCC2006",
        sum_rule="extensive",
        auxiliary_dimensions={"gas": "source (gas)"},
    )

    assert (
        (result.pr.loc[{"category": "1"}] == 1.0 * primap2.ureg("Gg CO2 / year"))
        .all()
        .item()
    )
    assert (
        (result.pr.loc[{"category": "2"}] == 2.0 * primap2.ureg("Gg CO2 / year"))
        .all()
        .item()
    )
