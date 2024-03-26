"""Compose a harmonized dataset from multiple input datasets."""

import math
from collections.abc import Hashable

import numpy as np
import tqdm
import xarray as xr
from loguru import logger

from . import _models


def compose(
    *,
    input_data: xr.Dataset,
    priority_definition: _models.PriorityDefinition,
    strategy_definition: _models.StrategyDefinition,
    progress_bar: bool = True,
) -> xr.Dataset:
    """
    Compose a harmonized dataset from multiple input datasets.

    The input datasets are treated at the timeseries level, for each timeseries:

    * the highest priority dataset is chosen according to the priority_definition
    * if the dataset contains any nan values (denoting missing information), the next
      highest priority dataset is selected to fill the missing values
    * a filling strategy is chosen according to the strategy_definition and the filling
      is done with the dataset selected in the previous step
    * if afterwards there are still missing nan values, the next highest priority
      dataset is selected and the previous step is repeated
    * when all missing values are filled or all input datasets are exhausted, the
      timeseries is finished and written to the result.

    In addition to the harmonized data, also a description of the processing steps
    done for each timeseries is returned in the result dataset, where for each
    entity, a variable of the form "Processing of $entity" is returned, with the same
    dimensions as the entity, apart from the time dimension.

    Parameters
    ----------
    input_data
        Dataset with the input data. The input data dimensions determine the output
        data dimensions: the output data has the same dimensions as the input data,
        minus the priority dimensions defined in the priority_definition. From the
        priority dimensions, the different datasets for the filling are selected, so
        they vanish in the result.
    priority_definition
        Defines the priorities to select timeseries from the input data. Priorities
        are formed by a list of selections and are used "from left to right", where the
        first matching selection has the highest priority. Each selection has to specify
        values for all priority dimensions (so that exactly one timeseries is selected
        from the input data), but can also specify other dimensions. That way it is,
        e.g., possible to define a different priority for a specific country by listing
        it early (i.e. with high priority) before the more general rules which should
        be applied for all other countries.
    strategy_definition
        Defines the filling strategies to be used when filling timeseries with other
        timeseries. Again, the priority is defined by a list of selections and
        corresponding strategies which are used "from left to right". Selections can use
        any dimension and don't have to apply to only one timeseries. For example, to
        define a default strategy which should be used for all timeseries unless
        something else is configured, configure an empty selection as the last
        (rightmost) entry.
    progress_bar
        If True (default), show progress bars using the tqdm package during the
        operation.

    Returns
    -------
        result. Dataset with the same entities and dimensions as input_data, but with
        following changes: the data is composed and filled according to the rules,
        the priority dimensions are reduced and not included in the result, and
        additional variables of the form "Processing of $entity" are added which
        describe the processing steps done for each timeseries.
    """
    result_das = {}

    if progress_bar:
        entity_iterator = tqdm.tqdm(input_data)
    else:
        entity_iterator = input_data
    for entity in entity_iterator:
        if progress_bar:
            entity_iterator.set_postfix_str(entity)
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
        result_das[entity] = xr.DataArray(
            data=np.nan,
            dims=result_dimensions,
            coords={
                dim: input_da.coords[dim]
                for dim in input_da.coords
                if dim not in priority_definition.selection_dimensions
            },
            attrs=input_da.attrs,
        )
        result_das[f"Processing of {entity}"] = xr.DataArray(
            data=np.empty(
                [len(input_da.coords[dim]) for dim in group_by_dimensions], dtype=object
            ),
            dims=group_by_dimensions,
            coords=[input_da.coords[dim] for dim in group_by_dimensions],
        )
        number_of_timeseries = math.prod(
            len(input_da[dim]) for dim in group_by_dimensions
        )
        if progress_bar:
            pbar = tqdm.tqdm(total=number_of_timeseries, unit="ts", unit_scale=True)
        else:
            pbar = None
        iterate_next_fixed_dimension(
            input_da=input_da,
            priority_definition=priority_definition,
            strategy_definition=strategy_definition,
            group_by_dimensions=group_by_dimensions,
            result_da=result_das[entity],
            result_processing_da=result_das[f"Processing of {entity}"],
            progress_bar=pbar,
        )
        if progress_bar:
            pbar.close()

    return xr.Dataset(result_das).pr.quantify()


def priority_coordinates_repr(
    *, fill_ts: xr.DataArray, priority_dimensions: list[str]
) -> str:
    """Reduce the priority coordinates to a short string representation."""
    priority_coordinates: dict[str, str] = {
        k: fill_ts[k].item() for k in priority_dimensions
    }
    if len(priority_coordinates) == 1:
        # only one priority dimension, just output the value because it is clear what is
        # meant
        return repr(next(iter(priority_coordinates.values())))
    return repr(priority_coordinates)


def iterate_next_fixed_dimension(
    *,
    input_da: xr.DataArray,
    priority_definition: _models.PriorityDefinition,
    strategy_definition: _models.StrategyDefinition,
    group_by_dimensions: tuple[Hashable, ...],
    result_da: xr.DataArray,
    result_processing_da: xr.DataArray,
    progress_bar: tqdm.tqdm | None,
) -> None:
    """Recursively iterate over dimensions in group_by_dimensions.

    If there is only one dimension left, actually compute results and store
    them in the result_da and the result_processing_da. Otherwise, iterate over one
    dimension and recursively call iterate_next_fixed_dimension again.
    """
    my_dim = group_by_dimensions[0]
    new_group_by_dimensions = group_by_dimensions[1:]
    for val_array in input_da[my_dim]:
        val = val_array.item()
        limited_strategy_definition = strategy_definition.limit(dim=my_dim, value=val)
        limited_priority_definition = priority_definition.limit(dim=my_dim, value=val)
        if new_group_by_dimensions:
            # have to iterate further until all dimensions are consumed
            iterate_next_fixed_dimension(
                input_da=input_da.loc[{my_dim: val}],
                priority_definition=limited_priority_definition,
                strategy_definition=limited_strategy_definition,
                group_by_dimensions=new_group_by_dimensions,
                result_da=result_da.loc[{my_dim: val}],
                result_processing_da=result_processing_da.loc[{my_dim: val}],
                progress_bar=progress_bar,
            )
        else:
            # actually compute results
            (
                result_da.loc[{my_dim: val}],
                result_processing_da.loc[{my_dim: val}],
            ) = compose_timeseries(
                input_data=input_da.loc[{my_dim: val}],
                priority_definition=limited_priority_definition,
                strategy_definition=limited_strategy_definition,
            )
            if progress_bar is not None:
                progress_bar.update()


def compose_timeseries(
    *,
    input_data: xr.DataArray,
    priority_definition: _models.PriorityDefinition,
    strategy_definition: _models.StrategyDefinition,
) -> tuple[xr.DataArray, _models.TimeseriesProcessingDescription]:
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
        result_ts, processing_description. In result_ts is the numerical result, with
        the time as the only dimension.
        processing_description is the representation of the processing steps taken to
        derive the result.
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
    processing_steps_descriptions = []
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

        if result_ts is None:
            context_logger.debug(
                f"{fill_ts_repr} is the highest-priority source, using as the "
                f"basis to fill."
            )
            processing_steps_descriptions.append(
                _models.ProcessingStepDescription(
                    time="all",
                    processing_description=f"used values from {fill_ts_repr}",
                    strategy="initial",
                )
            )
            result_ts = fill_ts_no_prio_dims
        else:
            if fill_ts.isnull().all():
                processing_steps_descriptions.append(
                    _models.ProcessingStepDescription(
                        time="all",
                        processing_description=f"{fill_ts_repr} is fully NaN, skipped",
                        strategy="none",
                    )
                )
                continue

            context_logger.debug(f"Filling with {fill_ts_repr} now.")
            strategy = strategy_definition.find_strategy(fill_ts)
            result_ts, descriptions = strategy.fill(
                ts=result_ts,
                fill_ts=fill_ts_no_prio_dims,
                fill_ts_repr=fill_ts_repr,
            )
            processing_steps_descriptions += descriptions

        if not result_ts.isnull().any():
            context_logger.debug("No NaNs remaining, skipping the rest of the sources.")
            break

    if result_ts is None:
        raise ValueError(
            f"No priority selector matched for "
            f"\n{input_data.coords}\n{priority_definition=}"
        )

    return result_ts, _models.TimeseriesProcessingDescription(
        steps=processing_steps_descriptions
    )
