"""Tests for _setters.py"""

import re

import numpy as np
import pint
import pytest
import xarray as xr

from primap2 import ureg

from .utils import allclose, assert_aligned_equal, assert_ds_aligned_equal


@pytest.fixture
def da(minimal_ds) -> xr.DataArray:
    da = minimal_ds["CO2"]
    # cast coord explicitly to object, because calling set() often casts to object
    # as a side effect of modifying the coords, and we are fine with that.
    da["area (ISO3)"] = da["area (ISO3)"].astype(object)
    return da


@pytest.fixture
def ts() -> np.ndarray:
    return np.linspace(0, 20, 21)


@pytest.fixture
def co2() -> pint.Unit:
    return ureg("Gg CO2 / year")


@pytest.fixture(params=["fillna_empty", "error", "fillna", "overwrite", None])
def existing(request) -> dict[str, str]:
    if request.param is not None:
        return {"existing": request.param}
    else:
        return {}


@pytest.fixture(params=["error", "extend", None])
def new(request) -> dict[str, str]:
    if request.param is not None:
        return {"new": request.param}
    else:
        return {}


class TestDASetter:
    def test_new_error(self, da: xr.DataArray, ts, co2, existing):
        with pytest.raises(
            KeyError,
            match=re.escape(
                "Values {'CUB'} not in 'area (ISO3)', use new='extend' to automatically"
                " insert new values into dim."
            ),
        ):
            da.pr.set("area", "CUB", ts * co2, new="error", **existing)

    def test_new_works(self, da: xr.DataArray, ts, co2, existing):
        actual = da.pr.set("area", ["CUB"], 2 * ts * co2, new="extend", **existing)
        expected = da.reindex({"area (ISO3)": [*da["area (ISO3)"].values, "CUB"]})
        expected.loc[{"area (ISO3)": "CUB"}] = ts[..., np.newaxis] * 2 * co2
        assert_aligned_equal(actual, expected)

    def test_exists_default_error(
        self, da: xr.DataArray, ts: np.ndarray, co2: pint.Unit, new
    ):
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Values {'COL'} for 'area (ISO3)' already exist and contain data."
                " Use existing='overwrite' or 'fillna' to avoid this error."
            ),
        ):
            da.pr.set("area", "COL", ts * co2, **new)

    def test_exists_empty_default(
        self, da: xr.DataArray, ts: np.ndarray, co2: pint.Unit, new
    ):
        da.loc[{"area (ISO3)": "COL"}].pint.magnitude[:] = np.nan
        actual = da.pr.set("area", "COL", ts * co2, **new)
        expected = da
        expected.loc[{"area (ISO3)": "COL"}] = ts[..., np.newaxis] * co2
        assert_aligned_equal(actual, expected)

    def test_exists_somena_default(
        self, da: xr.DataArray, ts: np.ndarray, co2: pint.Unit, new
    ):
        da.loc[{"area (ISO3)": "COL"}].pint.magnitude[:] = np.nan
        da.loc[{"area (ISO3)": "COL", "time": "2001"}] = 2 * co2
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Values {'COL'} for 'area (ISO3)' already exist and contain data."
                " Use existing='overwrite' or 'fillna' to avoid this error."
            ),
        ):
            da.pr.set("area", "COL", ts * co2, **new)

    def test_exists_error(self, da: xr.DataArray, ts: np.ndarray, co2: pint.Unit, new):
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Values {'COL'} for 'area (ISO3)' already exist."
                " Use existing='overwrite' or 'fillna' to avoid this error."
            ),
        ):
            da.pr.set("area", "COL", ts * co2, existing="error", **new)

    def test_exists_overwrite(
        self, da: xr.DataArray, ts: np.ndarray, co2: pint.Unit, new
    ):
        actual = da.pr.set("area", "COL", ts * co2, existing="overwrite", **new)
        expected = da.copy()
        expected.loc[{"area (ISO3)": "COL"}] = ts[..., np.newaxis] * co2
        assert_aligned_equal(actual, expected)

    def test_exists_fillna(self, da: xr.DataArray, ts: np.ndarray, co2: pint.Unit, new):
        expected = da.copy()
        expected.loc[{"area (ISO3)": "COL", "time": "2009"}] = np.nan * co2
        actual = expected.pr.set("area", "COL", ts * co2, existing="fillna", **new)
        expected.loc[{"area (ISO3)": "COL", "time": "2009"}] = 9 * co2
        assert_aligned_equal(actual, expected)

    def test_mixed_default(self, da: xr.DataArray, ts: np.ndarray, co2: pint.Unit):
        da.loc[{"area (ISO3)": "COL"}].pint.magnitude[:] = np.nan
        actual = da.pr.set(
            "area",
            ["COL", "CUB"],
            np.array([ts, 2 * ts]).T * co2,
        )
        expected = da.reindex({"area (ISO3)": [*da["area (ISO3)"].values, "CUB"]})
        expected.loc[{"area (ISO3)": "COL"}] = ts[..., np.newaxis] * co2
        expected.loc[{"area (ISO3)": "CUB"}] = ts[..., np.newaxis] * 2 * co2
        assert_aligned_equal(actual, expected)

    def test_mixed_error(self, da: xr.DataArray, ts: np.ndarray, co2: pint.Unit):
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Values {'COL'} for 'area (ISO3)' already exist."
                " Use existing='overwrite' or 'fillna' to avoid this error."
            ),
        ):
            da.pr.set(
                "area",
                ["COL", "CUB"],
                ts * co2,
                existing="error",
            )

    def test_mixed_overwrite(self, da: xr.DataArray, ts: np.ndarray, co2: pint.Unit):
        actual = da.pr.set(
            "area",
            ["COL", "CUB"],
            np.array([ts, 2 * ts]).T * co2,
            existing="overwrite",
        )
        expected = da.reindex({"area (ISO3)": [*da["area (ISO3)"].values, "CUB"]})
        expected.loc[{"area (ISO3)": "COL"}] = ts[..., np.newaxis] * co2
        expected.loc[{"area (ISO3)": "CUB"}] = ts[..., np.newaxis] * 2 * co2
        assert_aligned_equal(actual, expected)

        # with explicit value_dims
        actual = da.pr.set(
            "area",
            ["COL", "CUB"],
            np.array([ts, 2 * ts]) * co2,
            value_dims=["area (ISO3)", "time"],
            existing="overwrite",
        )
        assert_aligned_equal(actual, expected)

    def test_mixed_fillna(self, da: xr.DataArray, ts: np.ndarray, co2: pint.Unit):
        tda = da.copy()
        tda.loc[{"area (ISO3)": "COL", "time": "2009"}] = np.nan * co2
        actual = tda.pr.set(
            "area",
            ["COL", "CUB"],
            np.array([ts, 2 * ts]).T * co2,
            existing="fillna",
        )
        expected = da.reindex({"area (ISO3)": [*da["area (ISO3)"].values, "CUB"]})
        expected.loc[{"area (ISO3)": "COL", "time": "2009"}] = 9 * co2
        expected.loc[{"area (ISO3)": "CUB"}] = ts[..., np.newaxis] * 2 * co2
        assert_aligned_equal(actual, expected)

    def test_new_from_array(self, da: xr.DataArray, ts: np.ndarray, co2: pint.Unit):
        # with scalar dimension
        actual = da.pr.set("area", "CUB", 2 * da.pr.loc[{"area": "COL"}])
        expected = da.reindex({"area (ISO3)": [*da["area (ISO3)"].values, "CUB"]})
        expected = expected.fillna(expected.loc[{"area (ISO3)": "COL"}] * 2)
        assert_aligned_equal(actual, expected)

        actual_error = da.pr.set(
            "area", "CUB", 2 * da.pr.loc[{"area": "COL"}], existing="error"
        )
        assert_aligned_equal(actual_error, expected)

        # with non-scalar dimension
        with pytest.raises(KeyError, match="not all values found in index"):
            da.pr.set("area", "CUB", 2 * da.pr.loc[{"area": ["COL"]}])

    def test_mixed_from_data_array_overwrite(
        self, da: xr.DataArray, ts: np.ndarray, co2: pint.Unit
    ):
        actual = da.pr.set(
            "area", ["CUB", "COL"], 2 * da.pr.loc[{"area": "COL"}], existing="overwrite"
        )
        expected = da.reindex({"area (ISO3)": [*da["area (ISO3)"].values, "CUB"]})
        expected.loc[{"area (ISO3)": "COL"}].pint.magnitude[:] = np.nan
        expected = expected.fillna(da.loc[{"area (ISO3)": "COL"}] * 2)
        assert_aligned_equal(actual, expected)

    @pytest.fixture
    def mda(self):
        return xr.DataArray(
            np.zeros((3, 4, 5)),
            coords=[
                ("a", ["a1", "a2", "a3"]),
                ("b", ["b1", "b2", "b3", "b4"]),
                ("c", ["c1", "c2", "c3", "c4", "c5"]),
            ],
            attrs={"test": "attr"},
            name="thing",
        )

    def test_multidimensional_ndarray(self, mda: xr.DataArray):
        actual = mda.pr.set("a", "a4", np.ones((4, 5)))
        expected = mda.reindex({"a": ["a1", "a2", "a3", "a4"]}).fillna(1)
        assert_aligned_equal(actual, expected)

        actual = mda.pr.set("a", "a4", np.ones(4), value_dims=["b"])
        assert_aligned_equal(actual, expected)

        actual = mda.pr.set("a", "a4", np.ones(5), value_dims=["c"])
        assert_aligned_equal(actual, expected)

        actual = mda.pr.set("a", "a4", np.ones((4, 5)), value_dims=["b", "c"])
        assert_aligned_equal(actual, expected)

    @pytest.mark.parametrize("shape", [3, 4, 5])
    def test_multidimensional_ndarray_underspecified(
        self, mda: xr.DataArray, new, shape
    ):
        match = (
            "Could not automatically determine value dimensions, please use the"
            " value_dims parameter."
        )
        with pytest.raises(ValueError, match=match):
            mda.pr.set("a", "a3", np.ones(shape), existing="overwrite", **new)

    def test_multidimensional_data_array(self, mda: xr.DataArray):
        actual = mda.pr.set(
            "a", "a4", xr.DataArray(np.ones((4, 5)), coords=[mda["b"], mda["c"]])
        )
        expected = mda.reindex({"a": ["a1", "a2", "a3", "a4"]}).fillna(1)
        assert_aligned_equal(actual, expected)

        actual = mda.pr.set("a", "a4", xr.DataArray(np.ones(4), coords=[mda["b"]]))
        assert_aligned_equal(actual, expected)

        actual = mda.pr.set("a", "a4", xr.DataArray(np.ones(5), coords=[mda["c"]]))
        assert_aligned_equal(actual, expected)

    def test_multidimensional_data_array_existing(self, mda: xr.DataArray):
        actual = mda.pr.set(
            "a",
            "a3",
            xr.DataArray(np.ones((4, 5)), coords=[mda["b"], mda["c"]]),
            existing="overwrite",
            new="error",
        )
        expected = mda
        expected.loc[{"a": "a3"}] = 1
        assert_aligned_equal(actual, expected)

        actual = mda.pr.set(
            "a",
            "a3",
            xr.DataArray(np.ones(4), coords=[mda["b"]]),
            existing="overwrite",
            new="error",
        )
        assert_aligned_equal(actual, expected)

        actual = mda.pr.set(
            "a",
            "a3",
            xr.DataArray(np.ones(5), coords=[mda["c"]]),
            existing="overwrite",
            new="error",
        )
        assert_aligned_equal(actual, expected)

    def test_multidimensional_data_array_dim_contained(self, mda: xr.DataArray):
        actual = mda.pr.set(
            "a",
            ["a3", "a4", "a5"],
            xr.DataArray(
                np.ones((3, 4, 5)),
                coords=[("a", ["a3", "a4", "a5"]), mda["b"], mda["c"]],  # type: ignore
            ),
            existing="overwrite",
        )
        expected = mda.reindex({"a": ["a1", "a2", "a3", "a4", "a5"]}).fillna(1)
        expected.loc[{"a": "a3"}] = 1
        assert_aligned_equal(actual, expected)

    def test_multidimensional_data_array_dim_oversized(self, mda: xr.DataArray):
        actual = mda.pr.set(
            "a",
            ["a3", "a4", "a5"],
            xr.DataArray(
                np.ones((4, 4, 5)),
                coords=[
                    ("a", ["a2", "a3", "a4", "a5"]),
                    mda["b"],
                    mda["c"],
                ],  # type: ignore
            ),
            existing="overwrite",
        )
        expected = mda.reindex({"a": ["a1", "a2", "a3", "a4", "a5"]}).fillna(1)
        expected.loc[{"a": "a3"}] = 1
        assert_aligned_equal(actual, expected)

    def test_over_specific(self, da: xr.DataArray, ts: np.ndarray, new):
        with pytest.raises(
            ValueError, match="value_dims given, but value is already a DataArray."
        ):
            da.pr.set(
                "area",
                "COL",
                da.pr.loc[{"area": "BOL"}],
                value_dims=["area", "time", "source"],
                existing="overwrite",
                **new,
            )

    def test_incompatible_units(self, da: xr.DataArray, ts: np.ndarray, new):
        with pytest.raises(pint.errors.DimensionalityError, match="Cannot convert"):
            da.pr.set("area", "COL", ts * ureg("kg"), existing="overwrite", **new)

    def test_automatic_unit_conversion(
        self, da: xr.DataArray, ts: np.ndarray, co2, new
    ):
        actual = da.pr.set(
            "area", "COL", ts * ureg("Mg CO2 / year"), existing="overwrite", **new
        )
        expected = da
        expected.loc[{"area (ISO3)": "COL"}] = 1e-3 * ts[..., np.newaxis] * co2
        assert_aligned_equal(actual, expected)

    def test_dim_does_not_exist(self, da: xr.DataArray, ts, existing, new):
        with pytest.raises(ValueError, match="Dimension 'asdf' does not exist."):
            da.pr.set("asdf", ["COL"], ts, **existing, **new)

    def test_existing_wrong(self, da: xr.DataArray, ts, new):
        with pytest.raises(
            ValueError,
            match="If given, 'existing' must specify one of 'error', 'overwrite', "
            "'fillna_empty', or 'fillna', not 'asdf'.",
        ):
            da.pr.set("area", ["COL"], ts, existing="asdf", **new)

    def test_new_wrong(self, da: xr.DataArray, ts, existing):
        with pytest.raises(
            ValueError,
            match="If given, 'new' must specify one of 'error' or 'extend', not"
            " 'asdf'.",
        ):
            da.pr.set("area", ["CUB"], ts, new="asdf", **existing)


class TestDsSetter:
    def test_new(self, minimal_ds: xr.Dataset, existing):
        actual = minimal_ds.pr.set(
            "area", "CUB", minimal_ds.pr.loc[{"area": "COL"}] * 2, **existing
        )
        expected = minimal_ds.reindex(
            {"area (ISO3)": [*minimal_ds["area (ISO3)"].values, "CUB"]}
        )
        for key in expected.keys():
            expected[key] = expected[key].fillna(
                expected[key].pr.loc[{"area": "COL"}] * 2
            )
        assert_ds_aligned_equal(actual, expected)

    def test_new_error(self, minimal_ds: xr.Dataset, existing):
        with pytest.raises(
            KeyError,
            match=re.escape(
                "Values {'CUB'} not in 'area (ISO3)', use new='extend' to automatically"
                " insert new values into dim."
            ),
        ):
            minimal_ds.pr.set(
                "area",
                "CUB",
                minimal_ds.pr.loc[{"area": "COL"}] * 2,
                new="error",
                **existing,
            )

    def test_existing_default(self, minimal_ds: xr.Dataset, new):
        with pytest.raises(ValueError):
            minimal_ds.pr.set(
                "area", "COL", minimal_ds.pr.loc[{"area": "COL"}] * 2, **new
            )

    def test_existing_overwrite(self, minimal_ds: xr.Dataset, new):
        actual = minimal_ds.pr.set(
            "area",
            "COL",
            minimal_ds.pr.loc[{"area": "COL"}] * 2,
            existing="overwrite",
            **new,
        )
        expected = minimal_ds.pint.dequantify()
        for key in expected:
            expected[key].loc[{"area (ISO3)": "COL"}] = (
                expected[key].loc[{"area (ISO3)": "COL"}] * 2
            )
        expected = expected.pr.quantify()
        assert_ds_aligned_equal(actual, expected)
        assert not allclose(minimal_ds["CO2"], actual["CO2"])

    def test_existing_overwrite_nan(self, minimal_ds: xr.Dataset):
        actual = minimal_ds.pr.set(
            "area",
            "COL",
            xr.full_like(minimal_ds.pr.loc[{"area": "COL"}], np.nan),
            existing="overwrite",
        )
        assert all(actual["CO2"].pr.loc[{"area": "COL"}].isnull())
        assert all(~actual["CO2"].pr.loc[{"area": "ARG"}].isnull())

    def test_existing_fillna(self, minimal_ds: xr.Dataset, new):
        minimal_ds["CO2"].pr.loc[{"area": "COL", "time": "2001"}].pint.magnitude[:] = (
            np.nan
        )
        actual = minimal_ds.pr.set(
            "area", "COL", minimal_ds.pr.loc[{"area": "MEX"}], existing="fillna", **new
        )
        expected = minimal_ds.fillna(minimal_ds.pr.loc[{"area": "MEX"}])
        assert_ds_aligned_equal(actual, expected)

    def test_existing_wrong_type(self, minimal_ds: xr.Dataset, new):
        with pytest.raises(TypeError, match="value must be a Dataset, not"):
            minimal_ds.pr.set("area", "COL", np.zeros((3, 4)), **new)

    def test_wrong_dim(self, minimal_ds: xr.Dataset, existing, new):
        with pytest.raises(ValueError, match="Dimension 'asdf' does not exist."):
            minimal_ds.pr.set(
                "asdf", "COL", minimal_ds.pr.loc[{"area": "COL"}], **existing, **new
            )

    def test_inhomogeneous(self, minimal_ds: xr.Dataset):
        minimal_ds["population"] = minimal_ds["CO2"].pr.dequantify().sum("area (ISO3)")
        actual = minimal_ds.pr.set(
            "area", "CUB", minimal_ds.pr.loc[{"area": "COL"}] * 2
        )
        expected = minimal_ds.reindex(
            {"area (ISO3)": [*minimal_ds["area (ISO3)"].values, "CUB"]}
        )
        for key in expected.keys():
            if key == "population":
                continue
            expected[key] = expected[key].fillna(
                expected[key].pr.loc[{"area": "COL"}] * 2
            )
        assert_ds_aligned_equal(actual, expected)

    def test_opulent_complex(self, opulent_ds: xr.Dataset):
        ds = opulent_ds
        ds.pr.set(
            "category",
            "0",
            ds.pr.loc[{"category": ["1", "2", "3", "4", "5"]}].pr.sum("category"),
            existing="overwrite",
        )
