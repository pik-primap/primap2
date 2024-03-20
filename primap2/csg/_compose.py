"""Compose a harmonized dataset from multiple input datasets."""

import math
import typing
from collections.abc import Hashable

import numpy as np
import tqdm
import xarray as xr
from attrs import define
from loguru import logger
from uncertainties import unumpy


@define
class PriorityDefinition:
    """
    Defines source priorities for composing a full dataset or a single timeseries.

    Attributes
    ----------
    selection_dimensions
        List of dimensions from which source timeseries are selected. These are the
        priority dimensions, and each priority has to specify all selection dimensions.
    priorities
        List of priority selections. Higher priority selections come first. Each
        selection consists of a dict which maps dimension names to values. Each
        selection has to specify all selection_dimensions, but may specify additional
        dimensions (fixed dimensions) to limit the selection to specific cases.
        Examples:
        [{"area (ISO3)": "COL", "source": "A"}, {"source": "B"}]
        would select source "A" for Columbia, but source "B" for all other countries.
        In this case, selection_dimensions = ["source"].
    """

    selection_dimensions: list[Hashable]
    priorities: list[dict[Hashable, str]]

    def limit(self, dim: Hashable, value: str) -> typing.Self:
        """Remove one additional dimension by limiting to a single value.

        You can't remove selection dimensions, only additional (fixed) dimensions.
        """
        new_priorities = []
        for sel in self.priorities:
            if dim not in sel:
                new_priorities.append(sel)
            elif sel[dim] == value:
                # filter out the matching value
                new_priorities.append({k: v for k, v in sel.items() if k != dim})
        return PriorityDefinition(
            selection_dimensions=self.selection_dimensions, priorities=new_priorities
        )


class FillingStrategyModel(typing.Protocol):
    """
    Fill missing data in a timeseries using another timeseries.
    """

    type: str

    def fill(
        self,
        *,
        ts: xr.DataArray,
        fill_ts: xr.DataArray,
        fill_ts_repr: str,
    ) -> xr.DataArray:
        """Fill gaps in ts using data from the fill_ts.

        Parameters
        ----------
        ts
            Base timeseries. Missing data (NaNs) in this timeseries will be filled.
            This function must not modify the data in ts.
        fill_ts
            Fill timeseries. Data from this timeseries will be used (possibly after
            modification) to fill missing data in the base timeseries.
            This function must not modify the data in fill_ts.
        fill_ts_repr
            String representation of fill_ts. Human-readable short representation of
            the fill_ts (e.g. the source).

        Returns
        -------
            filled_ts. filled_ts contains the result, where missing
            data in ts is (partly) filled using information from fill_ts.
        """
        ...


class SubstitutionStrategy:
    """Fill missing data in the result dataset by copying.

    The NaNs in the result dataset are substituted with data from the filling
    dataset.
    """

    type = "substitution"

    def fill(
        self,
        *,
        ts: xr.DataArray,
        fill_ts: xr.DataArray,
        fill_ts_repr: str,
    ) -> xr.DataArray:
        """Fill gaps in ts using data from the fill_ts.

        Parameters
        ----------
        ts
            Base timeseries. Missing data (NaNs) in this timeseries will be filled.
            This function does not modify the data in ts.
        fill_ts
            Fill timeseries. Data from this timeseries will be used (possibly after
            modification) to fill missing data in the base timeseries.
            This function does not modify the data in fill_ts.
        fill_ts_repr
            String representation of fill_ts. Human-readable short representation of
            the fill_ts (e.g. the source).

        Returns
        -------
            filled_ts, descriptions. filled_ts contains the result, where missing
            data in ts is (partly) filled using unmodified data from fill_ts.
        """
        # xarray.fillna doesn't work with ufloats out of the box, hand-roll it.
        nan_mask = np.isnan(unumpy.nominal_values(ts.values))
        filled_ts = ts.copy(deep=True)
        filled_ts.values[nan_mask] = fill_ts.values[nan_mask]
        return filled_ts


@define
class TimeseriesSelector:
    selections: dict[Hashable, str | list[str]]
    """
    Examples:
    {"source": "FAOSTAT", "scenario": "high"}
    {"source": ["FAOSTAT", "UNFCCC"]}
    """

    def match(self, fill_ts: xr.DataArray) -> bool:
        """Check if a selected timeseries for filling matches this selector."""
        return all(fill_ts.coords[k] == v for (k, v) in self.selections.items())

    def match_single_dim(self, dim: Hashable, value: str) -> bool:
        """Check if a literal value in one dimension can match this selector."""
        if dim not in self.selections.keys():
            return True
        if isinstance(self.selections[dim], list):
            return value in self.selections[dim]
        return value == self.selections[dim]


@define
class StrategyDefinition:
    """
    Defines filling strategies for a single timeseries.

    Attributes
    ----------
    strategies
        List of mappings from a timeseries selector to a filling strategy. When a
        timeseries will be used to fill missing data, the list will be checked from the
        start, and the first matching TimeseriesSelector determines the FillingStrategy.
        Example: [(("source", ["FAOSTAT", "UNFCCC]), StraightStrategy),
                  ((, ), GlobalStrategy)]
        Note that the strategy can depend on fixed coordinates as well as priority
        coordinates.

    """

    strategies: list[tuple[TimeseriesSelector, FillingStrategyModel]]

    def find_strategy(self, fill_ts: xr.DataArray) -> FillingStrategyModel:
        """Find the strategy to use for the given filling timeseries."""
        for selector, strategy in self.strategies:
            if selector.match(fill_ts):
                return strategy
        raise KeyError(f"No matching strategy found for {fill_ts.coords}")

    def limit(self, dim: Hashable, value: str) -> typing.Self:
        """Limit this strategy definition to strategies applicable with the limit."""
        return StrategyDefinition(
            strategies=[
                (sel, strat)
                for (sel, strat) in self.strategies
                if sel.match_single_dim(dim, value)
            ]
        )


def priority_coordinates_repr(
    *, fill_ts: xr.DataArray, priority_dimensions: list[str]
) -> str:
    """Reduce the priority coordinates to a short string representation"""
    priority_coordinates: dict[str, str] = {
        k: fill_ts[k].item() for k in priority_dimensions
    }
    if len(priority_coordinates) == 1:
        # only one priority dimension, just output the value because it is clear what is
        # meant
        return repr(next(iter(priority_coordinates.values())))
    return repr(priority_coordinates)


@np.vectorize
def set_tag(x, tag):
    x.tag = tag
    return x


def compose_timeseries(
    *,
    input_data: xr.DataArray,
    priority_definition: PriorityDefinition,
    strategy_definition: StrategyDefinition,
) -> xr.DataArray:
    """
    Compute a single timeseries from given input data, priorities, and strategies.

    Parameters
    ----------
    input_data
        The input data which has dimensions time plus the priority dimensions. The
        fixed coordinates are supplied as zero-dimensional coordinates.
    priority_definition
        The definition of priorities within input_data. Each priority selects a single
        timeseries (i.e. an array with only time as the dimension), so has to specify
        values for all priority dimensions.
    strategy_definition
        The definition of strategies for timeseries in input_data.

    Returns
    -------
        result_ts, result_sources_ts. In result_ts is the numerical result, in
        result_sources_ts is the string representation of the derivation of each data
        point. Both are a single timeseries with the same fixed coordinates as
        input_data and the same time dimension like input_data.
    """
    context_logger = logger.bind(
        fixed_coordinates={k: v for k, v in input_data.coords.items() if v.shape == ()},
        priority_coordinates={
            k: list(v.data) for k, v in input_data.coords.items() if v.shape != ()
        },
        priorities=priority_definition.priorities,
        strategies=strategy_definition.strategies,
    )

    result_ts: None | xr.DataArray = None
    for selector in priority_definition.priorities:
        try:
            fill_ts = input_data.loc[selector]
        except KeyError:
            context_logger.debug(f"{selector=} matched no input_data, skipping.")
            continue

        fill_ts_repr = priority_coordinates_repr(
            fill_ts=fill_ts,
            priority_dimensions=priority_definition.selection_dimensions,
        )
        # remove priority dimension information, it messes with automatic alignment
        # in computations. The corresponding information is now in fill_ts_repr.
        fill_ts_no_prio_dims = fill_ts.drop_vars(
            priority_definition.selection_dimensions
        )
        # convert to uncertainties.unumpy.uarray and add tagging information to trace
        # the source of data
        fill_ts_tagged = fill_ts_no_prio_dims.copy()
        fill_ts_tagged.data = unumpy.uarray(
            fill_ts_tagged.data, np.zeros_like(fill_ts_tagged.data)
        )
        fill_ts_tagged.data = set_tag(fill_ts_tagged.data, fill_ts_repr)

        if result_ts is None:
            context_logger.debug(
                f"{fill_ts_repr} is the highest-priority source, using as the "
                f"basis to fill."
            )
            result_ts = fill_ts_tagged
        else:
            context_logger.debug(f"Filling with {fill_ts_repr} now.")
            strategy = strategy_definition.find_strategy(fill_ts)
            result_ts, descriptions = strategy.fill(
                ts=result_ts,
                fill_ts=fill_ts_tagged,
                fill_ts_repr=fill_ts_repr,
            )

        if not result_ts.isnull().any():
            context_logger.debug("No NaNs remaining, skipping the rest of the sources.")
            break

    if result_ts is None:
        raise ValueError(
            f"No selector matched for \n{input_data.coords}\n{priority_definition=}"
        )

    return result_ts


def compose(
    *,
    input_data: xr.Dataset,
    priority_definition: PriorityDefinition,
    strategy_definition: StrategyDefinition,
) -> xr.Dataset:
    result_das = {}

    entity_pbar = tqdm.tqdm(input_data)
    for entity in entity_pbar:
        entity_pbar.set_postfix_str(entity)
        input_da = input_data[entity].pr.dequantify()
        # all dimensions are either time, priority selection dimensions, or need to
        # be iterated over
        group_by_dimensions = tuple(
            dim
            for dim in input_da.dims
            if dim != "time" and dim not in priority_definition.selection_dimensions
        )
        # pre-allocate result arrays which will be filled timeseries-by-timeseries
        result_dimensions = ["time", *group_by_dimensions]
        result_shape = tuple(len(input_da[dim]) for dim in result_dimensions)
        result_das[entity] = xr.DataArray(
            data=np.empty(result_shape, dtype=object),
            dims=result_dimensions,
            coords=[input_da.coords[dim] for dim in result_dimensions],
        )
        number_of_timeseries = math.prod(
            len(input_da[dim]) for dim in group_by_dimensions
        )
        pbar = tqdm.tqdm(total=number_of_timeseries, unit="ts", unit_scale=True)
        iterate_next_fixed_dimension(
            input_da=input_da,
            priority_definition=priority_definition,
            strategy_definition=strategy_definition,
            group_by_dimensions=group_by_dimensions,
            result_da=result_das[entity],
            progress_bar=pbar,
        )
        pbar.close()

    return xr.Dataset(result_das).pr.quantify()


def iterate_next_fixed_dimension(
    *,
    input_da: xr.DataArray,
    priority_definition: PriorityDefinition,
    strategy_definition: StrategyDefinition,
    group_by_dimensions: tuple[Hashable, ...],
    result_da: xr.DataArray,
    progress_bar: tqdm.tqdm,
) -> None:
    my_dim = group_by_dimensions[0]
    new_group_by_dimensions = group_by_dimensions[1:]
    for val_array in input_da[my_dim]:
        val = val_array.item()
        strategy_definition = strategy_definition.limit(dim=my_dim, value=val)
        priority_definition = priority_definition.limit(dim=my_dim, value=val)
        if new_group_by_dimensions:
            # have to iterate further until all dimensions are consumed
            iterate_next_fixed_dimension(
                input_da=input_da.loc[{my_dim: val}],
                priority_definition=priority_definition,
                strategy_definition=strategy_definition,
                group_by_dimensions=new_group_by_dimensions,
                result_da=result_da.loc[{my_dim: val}],
                progress_bar=progress_bar,
            )
        else:
            # actually compute results
            result_da.loc[{my_dim: val}] = compose_timeseries(
                input_data=input_da.loc[{my_dim: val}],
                priority_definition=priority_definition,
                strategy_definition=strategy_definition,
            )
            progress_bar.update()


def create_description(ts: xr.DataArray) -> str:
    sources_times = {}
    for time in ts["time"]:
        tag = ts.loc[{"time": time}].item().tag
        if tag not in sources_times:
            sources_times[tag] = [time.dt.year.item()]
        else:
            sources_times[tag].append(time.dt.year.item())

    source_descriptions = [
        f"For time={time}: {source}" for source, time in sources_times.items()
    ]
    return "\n".join(source_descriptions)
