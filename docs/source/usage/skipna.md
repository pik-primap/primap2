---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Dealing with missing information

## Aggregation

xarray provides robust functions for aggregation ({py:meth}`xarray.DataArray.sum`).
PRIMAP2 adds functions which skip missing data points if the
information is missing at all points along certain axes, for example for
a whole time series.
Let's first create an example with missing information:

```{code-cell} ipython3
import pandas as pd
import numpy as np
import xarray as xr
import primap2

time = pd.date_range("2000-01-01", "2003-01-01", freq="YS")
area_iso3 = np.array(["COL", "ARG", "MEX"])
coords = [("area (ISO3)", area_iso3), ("time", time)]
da = xr.DataArray(
    data=[
        [1, 2, 3, 4],
        [np.nan, np.nan, np.nan, np.nan],
        [1, 2, 3, np.nan],
    ],
    coords=coords,
    name="test data"
)

da.pr.to_df()
```

Now, we can use the primap2 {py:meth}`xarray.DataArray.pr.sum` function to evaluate the sum of countries
while ignoring only those countries where the whole timeseries is missing, using the
`skipna_evaluation_dims` parameter:

```{code-cell} ipython3
da.pr.sum(dim="area", skipna_evaluation_dims="time").pr.to_df()
```

If you instead want to skip all NA values, use the `skipna` parameter:

```{code-cell} ipython3
da.pr.sum(dim="area", skipna=True).pr.to_df()
```

```{code-cell} ipython3
# compare this to the result of the standard xarray sum - it also skips NA values by default:

da.sum(dim="area (ISO3)").pr.to_df()
```

## Infilling

The same functionality is available for filling in missing information using the
{py:meth}`xarray.DataArray.pr.fill_all_na` function.
In this example, we fill missing information only where the whole time series is missing.

```{code-cell} ipython3
da.pr.fill_all_na("time", value=10).pr.to_df()
```

## Bulk aggregation

For larger aggregation tasks, e.g. aggregating several gas baskets from individual gases or aggregating a full category tree from leaves we have the functions {py:meth}`xarray.Dataset.pr.add_aggregates_variables`, {py:meth}`xarray.Dataset.pr.add_aggregates_coordinates`, and {py:meth}`xarray.DataArray.pr.add_aggregates_coordinates` which are highly configurable, but can also be used in a simplified mode for quick aggregation tasks. In the following we give a few examples. For the full feature set we refer to function descriptions linked above. The functions internally work with {py:meth}`xarray.Dataset.pr.merge` / {py:meth}`xarray.DataArray.pr.merge` to allow for consistency checks when target timeseries exist.

### Add aggregates for variables

The {py:meth}`xarray.Dataset.pr.add_aggregates_variables` function aggregates data from individual variables to new variables (usually gas baskets). Several variables can be created in one call where the order of definition is the order of creation. Filters can be specified to limit aggregation to certain coordinate values.

#### Examples

Sum gases in the minimal example dataset

```{code-cell} ipython3
ds_min = primap2.open_dataset("../minimal_ds.nc")
summed_ds = ds_min.pr.add_aggregates_variables(
    gas_baskets={
        "test (SARGWP100)": {
            "sources": ["CO2", "SF6", "CH4"],
        },
    },
)
summed_ds["test (SARGWP100)"]
```

We can also use a filter to limit the aggregation to e.g. a single country:

```{code-cell} ipython3
filtered_ds = ds_min.pr.add_aggregates_variables(
    gas_baskets={
        "test (SARGWP100)": {
            "sources": ["CO2", "SF6", "CH4"],
            "filter": {"area (ISO3)": ["COL"]},
        },
    },
)
filtered_ds["test (SARGWP100)"]
```
When filtering it is important to note that entities and variables are not the same thing. The difference between the `entity` and `variable` filters is that `'entity': ['SF6']` will match both variables `'SF6'` and `'SF6 (SARGWP100)'` (as both variables are for the entity `'SF6'`) while `'variable': ['SF6']` will match only the variable `'SF6'`.

If we recompute an existing timeseries it has to be consistent with the existing data. Here we use the simple mode to specify the aggregation rules. The example below fails because the result is inconsistent with existing data.

```{code-cell} ipython3
from xarray import MergeError

try:
    recomputed_ds = filtered_ds.pr.add_aggregates_variables(
        gas_baskets={
            "test (SARGWP100)": ["CO2", "CH4"],
        },
    )
    recomputed_ds["test (SARGWP100)"]
except MergeError as err:
    print(err)
```

We can set the tolerance high enough such that the tests passes and no error is thrown. This is only possible in the complex mode for the aggregation rules.

```{code-cell} ipython3
recomputed_ds = filtered_ds.pr.add_aggregates_variables(
    gas_baskets={
        "test (SARGWP100)": {
            "sources": ["CO2", "CH4"],
            "tolerance": 1,  #  100%
        },
    },
)
recomputed_ds["test (SARGWP100)"]
```

### Add aggregates for coordinates

The {py:meth}`xarray.Dataset.pr.add_aggregates_coordinates` function aggregates data from individual coordinate values to new values (e.g. from subcategories to categories). Several values for several coordinates can be created in one call where the order of definition is the order of creation. Filters can be specified to limit aggregation to certain coordinate values, entities or variables. Most of the operation is similar to the variable aggregation. Thus we keep the examples here shorter. The {py:meth}`xarray.DataArray.pr.add_aggregates_coordinates` function uses the same syntax.

#### Examples

Sum countries in the minimal example dataset

```{code-cell} ipython3
test_ds = ds_min.pr.add_aggregates_coordinates(
    agg_info={
        "area (ISO3)": {
            "all": {
                "sources": ["COL", "ARG", "MEX", "BOL"],
            }
        }
    }
)
test_ds
```
