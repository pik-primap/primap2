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

    assert (result.pr.loc[{"category": "1"}] == 1.0 * primap2.ureg("Gg CO2 / year")).all().item()
    assert (result.pr.loc[{"category": "2"}] == 2.0 * primap2.ureg("Gg CO2 / year")).all().item()

def test_convert_BURDI(empty_ds: xr.Dataset):
    # build a DA categorized by BURDI and with 1 everywhere so results are easy
    # to see

    # TODO this should come from climate categories
    mapping_BURDI_to_IPCC2006_PRIMAP = {
            "1" : "1",
            "1.A" : "1.A",
            "1.A.1" : "1.A.1",
            "1.A.2" : "1.A.2",
            "1.A.3" : "1.A.3",
            "1.A.4" : "1.A.4",
            "1.A.5" : "1.A.5",
            "1.B" : "1.B",
            "1.B.1" : "1.B.1",
            "1.B.2" : "1.B.2",
            "2" : "M.2.BURDI",
            "2.A" : "2.A",
            "2.B" : "M.2.B_2.B",
            "2.C" : "2.C",
            "2.D" : "M.2.H.1_2",
            "2.E" : "M.2.B_2.E",
            "2.F" : "2.F",
            "2.G" : "2.H.3",
            "3" : "2.D",
            "4" : "M.AG",
            "4.A" : "3.A.1",
            "4.B" : "3.A.2",
            "4.C" : "3.C.7",
            "4.D" : "M.3.C.45.AG",
            "4.E" : "3.C.1.c",
            "4.F" : "3.C.1.b",
            "4.G" : "3.C.8",
            "5" : "M.LULUCF",
            "6" : "4",
            "6.A" : "4.A",
            "6.B" : "4.D",
            "6.C" : "4.C",
            "6.D" : "4.E",
            "24540" : "0",
            "15163" : "M.0.EL",
            "14637" : "M.BK",
            "14424" : "M.BK.A",
            "14423" : "M.BK.M",
            "14638" : "M.BIO",
            "7" : "5",
    }  # 5.A-D ignored as not fitting 2006 cats


    da = empty_ds["CO2"]
    da = da.expand_dims({"category (IPCC1996)": list(mapping_BURDI_to_IPCC2006_PRIMAP.keys())})
    da = da.expand_dims({"source (gas)": list(cc.gas.keys())})
    arr = da.data.copy()

    result = da.pr.convert(
        "category",
        "IPCC2006",
        sum_rule="extensive",
        auxiliary_dimensions={"gas" : "source (gas)"},
    )

    # TODO
    assert False
