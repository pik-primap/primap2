"""Models for the composite source generator."""

import typing
from collections.abc import Hashable

import numpy as np
import xarray as xr
from attr import define


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

    def limit(self, dim: Hashable, value: str) -> "PriorityDefinition":
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


@define
class ProcessingStepDescription:
    """Structured description of a processing step done on a timeseries."""

    time: np.ndarray[np.datetime64] | typing.Literal["all"]
    processing_description: str
    strategy: str

    def __str__(self) -> str:
        return f"Using strategy={self.strategy} for times={self.time}: {self.processing_description}"


@define
class TimeseriesProcessingDescription:
    """Structured description of all processing steps done on a timeseries."""

    steps: list[ProcessingStepDescription]

    def __str__(self) -> str:
        return "\n".join(str(step) for step in self.steps)


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
    ) -> tuple[xr.DataArray, list[ProcessingStepDescription]]:
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
            filled_ts, descriptions. filled_ts contains the result, where missing
            data in ts is (partly) filled using information from fill_ts.
            descriptions contains human-readable, structured descriptions of how the
            data was processed, grouped by years for which the same processing steps
            were taken. Every year for which data was changed has to be described and
            no year for which data was not changed is allowed to be described.
        """
        ...


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
        Example: [({"source": ["FAOSTAT", "UNFCCC]}, StraightStrategy),
                  ({}, GlobalStrategy)]
        Note that the strategy can depend on fixed coordinates as well as priority
        coordinates.

    """

    strategies: list[tuple[dict[Hashable, str | list[str]], FillingStrategyModel]]

    def find_strategy(self, fill_ts: xr.DataArray) -> FillingStrategyModel:
        """Find the strategy to use for the given filling timeseries."""
        for selector, strategy in self.strategies:
            if self.match(selector=selector, fill_ts=fill_ts):
                return strategy
        raise KeyError(f"No matching strategy found for {fill_ts.coords}")

    def limit(self, dim: Hashable, value: str) -> "StrategyDefinition":
        """Limit this strategy definition to strategies applicable with the limit."""
        return StrategyDefinition(
            strategies=[
                (sel, strat)
                for (sel, strat) in self.strategies
                if self.match_single_dim(selector=sel, dim=dim, value=value)
            ]
        )

    @staticmethod
    def match(
        *, selector: dict[Hashable, str | list[str]], fill_ts: xr.DataArray
    ) -> bool:
        """Check if a selected timeseries for filling matches the selector."""
        return all(fill_ts.coords[k] == v for (k, v) in selector.items())

    @staticmethod
    def match_single_dim(
        *, selector: dict[Hashable, str | list[str]], dim: Hashable, value: str
    ) -> bool:
        """Check if a literal value in one dimension can match the selector."""
        if dim not in selector.keys():
            return True
        if isinstance(selector[dim], list):
            return value in selector[dim]
        return value == selector[dim]
