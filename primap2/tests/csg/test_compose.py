"""Tests for csg/test_compose.py"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import primap2.csg._compose


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


def test_timeseries_selector():
    da = get_single_ts(coords={"source": "A", "category": "1.A"})

    assert primap2.csg._compose.TimeseriesSelector({"source": "A"}).match(da)
    assert not primap2.csg._compose.TimeseriesSelector({"source": "B"}).match(da)
    assert primap2.csg._compose.TimeseriesSelector(
        {"source": "A", "category": "1.A"}
    ).match(da)
    assert not primap2.csg._compose.TimeseriesSelector(
        {"source": "A", "category": "1"}
    ).match(da)


def test_strategy_definition():
    da = get_single_ts(coords={"source": "A", "category": "1.A"})

    ts = primap2.csg._compose.TimeseriesSelector
    assert (
        primap2.csg._compose.StrategyDefinition(
            [(ts({"source": "A", "category": "1"}), 1), (ts({"source": "A"}), 2)]
        ).find_strategy(da)
        == 2
    )
    assert (
        primap2.csg._compose.StrategyDefinition(
            [
                (ts({"source": "A", "category": "1"}), 1),
                (ts({"source": "A", "category": "1.A"}), 2),
            ]
        ).find_strategy(da)
        == 2
    )
    with pytest.raises(KeyError):
        primap2.csg._compose.StrategyDefinition(
            [
                (ts({"source": "A", "category": "1"}), 1),
                (ts({"source": "A", "category": "1.B"}), 2),
                (ts({"source": "B", "category": "1.B"}), 3),
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
    priority_definition = primap2.csg._compose.PriorityDefinition(
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
    strategy_definition = primap2.csg._compose.StrategyDefinition(
        strategies=[
            (
                primap2.csg._compose.TimeseriesSelector({}),
                primap2.csg._compose.SubstitutionStrategy(),
            )
        ]
    )

    result_ts, result_sources_ts = primap2.csg._compose.compose(
        input_data=input_data,
        priority_definition=priority_definition,
        strategy_definition=strategy_definition,
    )


def test_compose_timeseries_trivial():
    priority_definition = primap2.csg._compose.PriorityDefinition(
        selection_dimensions=["source"], priorities=[{"source": "A"}, {"source": "B"}]
    )
    strategy_definition = primap2.csg._compose.StrategyDefinition(
        strategies=[
            (
                primap2.csg._compose.TimeseriesSelector({"source": "B"}),
                primap2.csg._compose.SubstitutionStrategy(),
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

    result_ts, result_sources_ts = primap2.csg._compose.compose_timeseries(
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

    esnp = np.full_like(enp, "'A'", dtype=object)
    esnp[0] = "'B'"  # filled from B
    esnp[1] = np.nan  # not filled, NaN in A and B, so no source
    expected_sources_ts = get_single_ts(
        data=esnp, coords={"category": "1", "area (ISO3)": "MEX"}, time=time
    )

    xr.testing.assert_identical(result_ts, expected_ts)
    xr.testing.assert_identical(result_sources_ts, expected_sources_ts)


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
