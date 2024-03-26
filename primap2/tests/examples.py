import datetime

import numpy as np
import pandas as pd
import xarray as xr

import primap2  # noqa: F401
from primap2 import ureg


def minimal_ds() -> xr.Dataset:
    """A valid, minimal dataset."""
    time = pd.date_range("2000-01-01", "2020-01-01", freq="YS")
    area_iso3 = np.array(["COL", "ARG", "MEX", "BOL"])

    # seed the rng with a constant to achieve predictable "randomness"
    rng = np.random.default_rng(1)

    minimal = xr.Dataset(
        {
            ent: xr.DataArray(
                data=rng.random((len(time), len(area_iso3), 1)),
                coords={
                    "time": time,
                    "area (ISO3)": area_iso3,
                    "source": ["RAND2020"],
                },
                dims=["time", "area (ISO3)", "source"],
                attrs={"units": f"{ent} Gg / year", "entity": ent},
            )
            for ent in ("CO2", "SF6", "CH4")
        },
        attrs={"area": "area (ISO3)"},
    ).pr.quantify()

    with ureg.context("SARGWP100"):
        minimal["SF6 (SARGWP100)"] = minimal["SF6"].pint.to("CO2 Gg / year")
    minimal["SF6 (SARGWP100)"].attrs["gwp_context"] = "SARGWP100"
    return minimal


COORDS = {
    "time": pd.date_range("2000-01-01", "2020-01-01", freq="YS"),
    "area (ISO3)": np.array(["COL", "ARG", "MEX", "BOL"]),
    "category (IPCC 2006)": np.array(["0", "1", "2", "3", "4", "5", "1.A", "1.B"]),
    "animal (FAOSTAT)": np.array(["cow", "swine", "goat"]),
    "product (FAOSTAT)": np.array(["milk", "meat"]),
    "scenario (FAOSTAT)": np.array(["highpop", "lowpop"]),
    "provenance": np.array(["projected"]),
    "model": np.array(["FANCYFAO"]),
    "source": np.array(["RAND2020", "RAND2021"]),
}


def opulent_ds() -> xr.Dataset:
    """A valid dataset using lots of features."""

    # seed the rng with a constant to achieve predictable "randomness"
    rng = np.random.default_rng(1)

    opulent = xr.Dataset(
        {
            ent: xr.DataArray(
                data=rng.random(tuple(len(x) for x in COORDS.values())),
                coords=COORDS,
                dims=list(COORDS.keys()),
                attrs={"units": f"{ent} Gg / year", "entity": ent},
            )
            for ent in ("CO2", "SF6", "CH4")
        },
        attrs={
            "entity_terminology": "primap2",
            "area": "area (ISO3)",
            "cat": "category (IPCC 2006)",
            "sec_cats": ["animal (FAOSTAT)", "product (FAOSTAT)"],
            "scen": "scenario (FAOSTAT)",
            "references": "doi:10.1012",
            "rights": "Use however you want.",
            "contact": "lol_no_one_will_answer@example.com",
            "title": "Completely invented GHG inventory data",
            "comment": "GHG inventory data ...",
            "institution": "PIK",
            "history": "2021-01-14 14:50 data invented\n"
            "2021-01-14 14:51 additional processing step",
            "publication_date": datetime.date(2099, 12, 31),
        },
    )

    pop_coords = {
        x: COORDS[x]
        for x in (
            "time",
            "area (ISO3)",
            "provenance",
            "model",
            "source",
        )
    }
    pop_shape = tuple(len(x) for x in pop_coords.values())
    opulent["population"] = xr.DataArray(
        data=rng.random(pop_shape),
        coords=pop_coords,
        dims=list(pop_coords.keys()),
        attrs={"entity": "population", "units": ""},
    )

    opulent = opulent.assign_coords(
        {
            "category_names": xr.DataArray(
                data=np.array(
                    [
                        "total",
                        "industry",
                        "energy",
                        "transportation",
                        "residential",
                        "land use",
                        "heavy industry",
                        "light industry",
                    ]
                ),
                coords={"category (IPCC 2006)": COORDS["category (IPCC 2006)"]},
                dims=["category (IPCC 2006)"],
            )
        }
    )

    opulent = opulent.pint.quantify(unit_registry=ureg)

    with ureg.context("SARGWP100"):
        opulent["SF6 (SARGWP100)"] = opulent["SF6"].pint.to("CO2 Gg / year")
    opulent["SF6 (SARGWP100)"].attrs["gwp_context"] = "SARGWP100"

    return opulent


def opulent_str_ds() -> xr.Dataset:
    """Like the opulent dataset, but additionally with a stringly typed data variable
    "method"."""
    opulent = opulent_ds()

    method_coords = {
        x: COORDS[x]
        for x in (
            "time",
            "area (ISO3)",
            "model",
            "source",
        )
    }
    method_shape = tuple(len(x) for x in method_coords.values())
    opulent["method"] = xr.DataArray(
        data=np.ones(method_shape, dtype=str).astype(object),
        coords=method_coords,
        dims=list(method_coords.keys()),
        attrs={"entity": "method"},
    )
    opulent["method"].pr.loc[{"time": "2000", "area": "COL", "source": "RAND2020"}] = (
        "text"
    )

    return opulent


def empty_ds() -> xr.Dataset:
    """An empty hull of a dataset with missing data."""
    time = pd.date_range("2000-01-01", "2020-01-01", freq="YS")
    area_iso3 = np.array(["COL", "ARG", "MEX", "BOL"])
    coords = {
        "time": time,
        "area (ISO3)": area_iso3,
        "source": ["RAND2020"],
    }
    dims = ["time", "area (ISO3)", "source"]
    empty = xr.Dataset(
        {
            ent: xr.DataArray(
                data=np.zeros((len(time), len(area_iso3), 1), dtype=float),
                coords=coords,
                dims=dims,
                attrs={"units": f"{ent} Gg / year", "entity": ent},
            )
            for ent in ("CO2", "SF6", "CH4")
        },
        attrs={"area": "area (ISO3)"},
    ).pr.quantify()

    empty["KYOTOGHG (AR4GWP100)"] = xr.DataArray(
        data=np.zeros((len(time), len(area_iso3), 1), dtype=float),
        coords=coords,
        dims=dims,
        attrs={
            "units": "CO2 Gg / year",
            "entity": "KYOTOGHG",
            "gwp_context": "AR4GWP100",
        },
    ).pr.quantify()

    return empty
