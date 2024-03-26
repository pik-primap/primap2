"""Tests for csg/_compose.py"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import primap2.csg
import primap2.csg._compose
import primap2.csg._models


def get_single_ts(
    *,
    time: pd.DatetimeIndex | None = None,
    data: np.ndarray | None = None,
    dims: Sequence[str] | None = None,
    coords: dict[str, str | Sequence[str]] | None = None,
) -> xr.DataArray:
    if time is None:
        time = pd.date_range("1850-01-01", "2022-01-01", freq="YS")
    if dims is None:
        dims = []
    if data is None:
        data = np.linspace(0.0, 1.0, len(time))
    if coords is None:
        coords = {}
    return xr.DataArray(
        data,
        dims=["time", *dims],
        coords={"time": time, **coords},
    )


def test_substitution_strategy():
    ts = get_single_ts(data=1.0)
    ts[0] = np.nan
    fill_ts = get_single_ts(data=2.0)

    (
        result_ts,
        result_descriptions,
    ) = primap2.csg.SubstitutionStrategy().fill(
        ts=ts, fill_ts=fill_ts, fill_ts_repr="B"
    )
    assert result_ts[0] == 2.0
    assert (result_ts[1:] == 1.0).all()
    assert len(result_descriptions) == 1
    assert result_descriptions[0].time == np.array(["1850"], dtype=np.datetime64)
    assert (
        result_descriptions[0].processing_description
        == "substituted with corresponding values from B"
    )
    assert "source" not in result_ts.coords.keys()


def test_selector_match():
    da = get_single_ts(coords={"source": "A", "category": "1.A"})

    assert primap2.csg.StrategyDefinition.match(selector={"source": "A"}, fill_ts=da)
    assert not primap2.csg.StrategyDefinition.match(
        selector={"source": "B"}, fill_ts=da
    )
    assert primap2.csg.StrategyDefinition.match(
        selector={"source": "A", "category": "1.A"}, fill_ts=da
    )
    assert not primap2.csg.StrategyDefinition.match(
        selector={"source": "A", "category": "1"}, fill_ts=da
    )


def test_selector_match_single_dim():
    assert primap2.csg.StrategyDefinition.match_single_dim(
        selector={"source": "A"}, dim="source", value="A"
    )
    assert not primap2.csg.StrategyDefinition.match_single_dim(
        selector={"source": "B"}, dim="source", value="A"
    )
    assert primap2.csg.StrategyDefinition.match_single_dim(
        selector={"source": ["A", "B"], "category": "1.A"}, dim="source", value="A"
    )
    assert primap2.csg.StrategyDefinition.match_single_dim(
        selector={"source": "A", "category": "1"}, dim="other", value="any"
    )


def test_strategy_definition():
    da = get_single_ts(coords={"source": "A", "category": "1.A"})

    assert (
        primap2.csg._models.StrategyDefinition(
            [({"source": "A", "category": "1"}, 1), ({"source": "A"}, 2)]
        ).find_strategy(da)
        == 2
    )
    assert (
        primap2.csg._models.StrategyDefinition(
            [
                ({"source": "A", "category": "1"}, 1),
                ({"source": "A", "category": "1.A"}, 2),
            ]
        ).find_strategy(da)
        == 2
    )
    with pytest.raises(KeyError):
        primap2.csg._models.StrategyDefinition(
            [
                ({"source": "A", "category": "1"}, 1),
                ({"source": "A", "category": "1.B"}, 2),
                ({"source": "B", "category": "1.B"}, 3),
            ]
        ).find_strategy(da)


def test_compose_trivial():
    input_data = primap2.tests.examples.opulent_ds()
    input_data = input_data.drop_vars(["population", "SF6 (SARGWP100)"])
    # we now have dimensions time, area (ISO3), category (IPCC 2006), animal (FAOSTAT)
    # product (FAOSTAT), scenario (FAOSTAT), provenance, model, source
    # We have variables (entities): CO2, SF6, CH4
    # We have sources: RAND2020, RAND2021
    # We have scenarios: highpop, lowpop
    # Idea: we use source, scenario as priority dimensions, everything else are fixed
    # coordinates.

    # generally, prefer source RAND2020, scenario lowpop,
    # then use source RAND2021, scenario highpop.
    # however, for Columbia, use source RAND2020, scenario highpop as highest priority
    # (this combination is not used at all otherwise).
    priority_definition = primap2.csg.PriorityDefinition(
        selection_dimensions=["source", "scenario (FAOSTAT)"],
        priorities=[
            {
                "area (ISO3)": "COL",
                "source": "RAND2020",
                "scenario (FAOSTAT)": "highpop",
            },
            {"source": "RAND2020", "scenario (FAOSTAT)": "lowpop"},
            {"source": "RAND2021", "scenario (FAOSTAT)": "highpop"},
        ],
    )
    # we use straight substitution always.
    strategy_definition = primap2.csg.StrategyDefinition(
        strategies=[
            (
                {},
                primap2.csg.SubstitutionStrategy(),
            )
        ]
    )

    result = primap2.csg.compose(
        input_data=input_data,
        priority_definition=priority_definition,
        strategy_definition=strategy_definition,
    )
    assert "CO2" in result.keys()
    assert "Processing of CO2" in result.keys()
    result_col = result["CO2"].loc[{"area (ISO3)": "COL"}]
    expected_col = (
        input_data["CO2"]
        .loc[
            {
                "area (ISO3)": "COL",
                "source": "RAND2020",
                "scenario (FAOSTAT)": "highpop",
            }
        ]
        .drop("scenario (FAOSTAT)")
        .drop("source")
    )
    xr.testing.assert_identical(result_col, expected_col)
    result_col_proc = (
        result["Processing of CO2"].loc[{"area (ISO3)": "COL"}].values.flat[0]
    )
    assert len(result_col_proc.steps) == 1
    assert result_col_proc.steps[0].time == "all"
    assert result_col_proc.steps[0].strategy == "initial"
    assert "'source': 'RAND2020'" in result_col_proc.steps[0].processing_description
    assert (
        "'scenario (FAOSTAT)': 'highpop'"
        in result_col_proc.steps[0].processing_description
    )

    result_arg = result["CO2"].loc[{"area (ISO3)": "ARG"}]
    expected_arg = (
        input_data["CO2"]
        .loc[
            {
                "area (ISO3)": "ARG",
                "source": "RAND2020",
                "scenario (FAOSTAT)": "lowpop",
            }
        ]
        .drop("scenario (FAOSTAT)")
        .drop("source")
    )
    xr.testing.assert_identical(result_arg, expected_arg)
    result_arg_proc = (
        result["Processing of CO2"].loc[{"area (ISO3)": "ARG"}].values.flat[0]
    )
    assert len(result_arg_proc.steps) == 1
    assert result_arg_proc.steps[0].time == "all"
    assert result_arg_proc.steps[0].strategy == "initial"
    assert "'source': 'RAND2020'" in result_arg_proc.steps[0].processing_description
    assert (
        "'scenario (FAOSTAT)': 'lowpop'"
        in result_arg_proc.steps[0].processing_description
    )


def test_compose_pbar():
    input_data = primap2.tests.examples.opulent_ds()
    input_data = input_data.drop_vars(["population", "SF6 (SARGWP100)"])
    priority_definition = primap2.csg.PriorityDefinition(
        selection_dimensions=["source", "scenario (FAOSTAT)"],
        priorities=[
            {"source": "RAND2020", "scenario (FAOSTAT)": "lowpop"},
            {"source": "RAND2021", "scenario (FAOSTAT)": "highpop"},
        ],
    )
    strategy_definition = primap2.csg.StrategyDefinition(
        strategies=[
            (
                {},
                primap2.csg.SubstitutionStrategy(),
            )
        ]
    )

    result = primap2.csg.compose(
        input_data=input_data,
        priority_definition=priority_definition,
        strategy_definition=strategy_definition,
        progress_bar=False,
    )
    assert "CO2" in result.keys()


def test_compose_timeseries_trivial():
    priority_definition = primap2.csg.PriorityDefinition(
        selection_dimensions=["source"], priorities=[{"source": "A"}, {"source": "B"}]
    )
    strategy_definition = primap2.csg.StrategyDefinition(
        strategies=[
            (
                {"source": "B"},
                primap2.csg.SubstitutionStrategy(),
            )
        ]
    )

    time = pd.date_range("1850-01-01", "2022-01-01", freq="YS")
    anp = np.linspace(0.0, 1.0, len(time))
    anp[0] = np.nan
    anp[1] = np.nan
    da_a = get_single_ts(
        coords={"source": "A", "category": "1", "area (ISO3)": "MEX"},
        data=anp,
        time=time,
    )
    bnp = np.linspace(1000.0, 2000.0, len(time))
    bnp[1] = np.nan
    bnp[2] = np.nan
    da_b = get_single_ts(
        coords={"source": "B", "category": "1", "area (ISO3)": "MEX"},
        data=bnp,
        time=time,
    )

    input_data = xr.concat((da_a, da_b), dim="source", join="exact")

    result_ts, result_description = primap2.csg._compose.compose_timeseries(
        input_data=input_data,
        priority_definition=priority_definition,
        strategy_definition=strategy_definition,
    )

    enp = np.linspace(0.0, 1.0, len(time))  # same as A
    enp[0] = 1000.0  # filled from B
    enp[1] = np.nan  # not filled, NaN in A and B
    expected_ts = get_single_ts(
        data=enp, coords={"category": "1", "area (ISO3)": "MEX"}, time=time
    )

    xr.testing.assert_identical(result_ts, expected_ts)

    assert len(result_description.steps) == 2
    assert result_description.steps[0].time == "all"
    assert result_description.steps[0].strategy == "initial"
    assert result_description.steps[0].processing_description == "used values from 'A'"
    assert len(result_description.steps[1].time) == 1
    assert result_description.steps[1].time[0] == np.datetime64("1850-01-01")
    assert result_description.steps[1].strategy == "substitution"
    assert (
        result_description.steps[1].processing_description
        == "substituted with corresponding values from 'B'"
    )


def test_compose_timeseries_no_match(caplog):
    priority_definition = primap2.csg.PriorityDefinition(
        selection_dimensions=["source"],
        priorities=[{"source": "C"}, {"source": "A"}, {"source": "B"}],
    )
    strategy_definition = primap2.csg.StrategyDefinition(
        strategies=[
            (
                {"source": "B"},
                primap2.csg.SubstitutionStrategy(),
            )
        ]
    )
    da_a = get_single_ts(
        coords={"source": "A", "category": "1", "area (ISO3)": "MEX"},
    )
    da_b = get_single_ts(
        coords={"source": "B", "category": "1", "area (ISO3)": "MEX"},
    )
    input_data = xr.concat((da_a, da_b), dim="source", join="exact")

    primap2.csg._compose.compose_timeseries(
        input_data=input_data,
        priority_definition=priority_definition,
        strategy_definition=strategy_definition,
    )

    assert "selector={'source': 'C'} matched no input_data, skipping." in caplog.text


def test_compose_timeseries_all_null():
    priority_definition = primap2.csg.PriorityDefinition(
        selection_dimensions=["source"],
        priorities=[{"source": "A"}, {"source": "B"}],
    )
    strategy_definition = primap2.csg.StrategyDefinition(
        strategies=[
            (
                {"source": "B"},
                primap2.csg.SubstitutionStrategy(),
            )
        ]
    )
    da_a = get_single_ts(
        coords={"source": "A", "category": "1", "area (ISO3)": "MEX"},
    )
    da_a[0] = np.nan
    da_b = get_single_ts(
        data=np.nan,
        coords={"source": "B", "category": "1", "area (ISO3)": "MEX"},
    )
    input_data = xr.concat((da_a, da_b), dim="source", join="exact")

    result, result_description = primap2.csg._compose.compose_timeseries(
        input_data=input_data,
        priority_definition=priority_definition,
        strategy_definition=strategy_definition,
    )

    print(result_description)
    assert (
        result_description.steps[1].processing_description
        == "'B' is fully NaN, skipped"
    )


def test_compose_timeseries_priorities_wrong():
    priority_definition = primap2.csg.PriorityDefinition(
        selection_dimensions=["source"],
        priorities=[{"source": "C"}],
    )
    strategy_definition = primap2.csg.StrategyDefinition(
        strategies=[
            (
                {"source": "C"},
                primap2.csg.SubstitutionStrategy(),
            )
        ]
    )
    da_a = get_single_ts(
        coords={"source": "A", "category": "1", "area (ISO3)": "MEX"},
    )
    da_b = get_single_ts(
        data=np.nan,
        coords={"source": "B", "category": "1", "area (ISO3)": "MEX"},
    )
    input_data = xr.concat((da_a, da_b), dim="source", join="exact")

    with pytest.raises(ValueError):
        primap2.csg._compose.compose_timeseries(
            input_data=input_data,
            priority_definition=priority_definition,
            strategy_definition=strategy_definition,
        )


def test_priority_coordinates_repr():
    assert (
        primap2.csg._compose.priority_coordinates_repr(
            fill_ts=get_single_ts(coords={"source": "A"}),
            priority_dimensions=["source"],
        )
        == "'A'"
    )
    assert (
        primap2.csg._compose.priority_coordinates_repr(
            fill_ts=get_single_ts(coords={"source": "A", "scenario": "S"}),
            priority_dimensions=["source", "scenario"],
        )
        == "{'source': 'A', 'scenario': 'S'}"
    )
    assert (
        primap2.csg._compose.priority_coordinates_repr(
            fill_ts=get_single_ts(coords={"source": "A", "scenario": "S"}),
            priority_dimensions=["scenario"],
        )
        == "'S'"
    )
