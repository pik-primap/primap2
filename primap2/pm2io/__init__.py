"""Data reading module of the PRIMAP2 climate policy analysis package."""


from ._data_reading import (
    from_interchange_format,
    read_interchange_format,
    read_wide_csv_file_if,
    write_interchange_format,
)

__all__ = [
    "read_wide_csv_file_if",
    "from_interchange_format",
    "write_interchange_format",
    "read_interchange_format",
]
