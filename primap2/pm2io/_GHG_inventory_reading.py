import re
from typing import Any, Dict, Union

import pandas as pd

# this file contains functions for reading of country GHG inventories
# from National Inventory Reports (NIR), biannual Update Reports (BUR),
# and other official country emissions inventories
# Most of the functions in this file are exposed to the outside yet they
# currently do not undergo the strict testing applied to the rest of PRIMAP2 as
# they are added during the process of reading an preparing data for the PRIMAP-hist
# update. Testing will be added in the future.


def nir_add_unit_information(
    df_nir: pd.DataFrame, *, unit_row: Union[str, int], unit_info: Dict[str, Any]
) -> pd.DataFrame:
    """
    This functions adds unit information to the header of a "entity-wide" file as
    present in the standard table format of National Inventory Reports (NIRs). The
    unit and entity information is extracted from combined unit and entity information
    in the row defined by `unit_row`. The parameter `unit_info` determines how this is
    done by regular expressions for unit and entity and a dict for manual replacements.
    For each column the routine tries to extract a unit using the regular expression.
    If this fails it looks in the `unit_info.manual_repl` dict for unit and entity
    information. If there is no information the default unit given in
    `unit_info.default_unit` is used. In this case the analyzed value is used as entity
    unchanged.

    Parameters
    ----------
    df_nir : pd.DataFrame
        Pandas DataFrame with the table to process
    unit_row : str or int
        String "header" to indicate that the column header should be used to derive the
        unit information or an integer specifying the row to use for unit information.
    unit_info : dict
        A dict with the fields
        * regexp_entity: regular expression that extracts the entity from the cell value
        * regexp_unit: regular expression that extracts the unit from the cell value
        * manual_repl: optional: dict defining unit an d entity for given cell values
        * default_unit: unit to be used if no unit can be extracted an no unit is given

    Returns
    -------
    pd.DataFrame
        DataFrame with explicit unit information (as column header)
    """

    if "manual_repl" not in unit_info:
        unit_info["manual_repl"] = {}

    # get the data to extract the units and entities from
    # can be either the header row or a regular row
    if unit_row == "header":
        values_for_units = list(df_nir.columns)
    else:
        # unit_row must be an integer
        values_for_units = list(df_nir.iloc[unit_row])

    re_unit = re.compile(unit_info["regexp_unit"])
    re_entity = re.compile(unit_info["regexp_entity"])

    units = values_for_units.copy()
    entities = values_for_units.copy()

    for idx, value in enumerate(values_for_units):
        if value in unit_info["manual_repl"]:
            units[idx] = unit_info["manual_repl"][value][1]
            entities[idx] = unit_info["manual_repl"][value][0]
        else:
            unit = re_unit.findall(value)
            if unit:
                units[idx] = unit[0]
            else:
                units[idx] = unit_info["default_unit"]
            entity = re_entity.findall(value)
            if entity:
                entities[idx] = entity[0]
            else:
                entities[idx] = value

    new_header = [entities, units]

    df_nir.columns = new_header
    if unit_row != "header":
        df_nir.drop(unit_row)

    return df_nir


def nir_convert_df_to_long(df_nir: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    This function converts an entity-wide NIR table for a single year to a long format
    DataFrame. The input DataFrame is required to have the following structure:
    * Columns for category, original category name, and data in this order, where
    category and original category name form a multiindex.
    * Column header as multiindex for entity and unit
    A column for the year is added during the conversion.

    Parameters
    ----------
    df_nir: pd.DataFrame
        Pandas DataFrame with the NIR table to be converted
    year: int
        Year of the given data

    Returns
    -------

    """
    header_long = ["category", "orig_cat_name", "entity", "unit", "time", "data"]

    df_stacked = df_nir.stack([0, 1]).to_frame()
    df_stacked.insert(0, "year", str(year))
    df_stacked.reset_index(inplace=True)
    df_stacked.columns = header_long
    return df_stacked
