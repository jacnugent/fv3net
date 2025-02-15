import numpy as np
from typing import Sequence
import xarray as xr

from vcm.select import meridional_ring
import vcm


def nearest_time(select_time: str, times: Sequence[str]):
    select_datetime = vcm.parse_datetime_from_str(select_time)
    datetimes = np.vectorize(vcm.parse_datetime_from_str)(times)
    closest_datetime = min(datetimes, key=lambda d: abs(d - select_datetime))
    return vcm.encode_time(closest_datetime)


def meridional_transect(ds: xr.Dataset):
    transect_coords = meridional_ring()
    return vcm.interpolate_unstructured(ds, transect_coords)


def plot_transect(
    data: xr.DataArray,
    xaxis: str = "lat",
    yaxis: str = "pressure",
    column_dim: str = "derivation",
    dataset_dim: str = "dataset",
):
    row_dim = dataset_dim if dataset_dim in data.dims else None
    num_datasets = len(data[dataset_dim]) if dataset_dim in data.dims else 1
    figsize = (10, 4 * num_datasets)
    facetgrid = data.plot(
        y=yaxis,
        x=xaxis,
        yincrease=False,
        col=column_dim,
        row=row_dim,
        figsize=figsize,
        robust=True,
    )
    facetgrid.set_ylabels("Pressure [Pa]")
    facetgrid.set_xlabels("Latitude [deg]")

    f = facetgrid.fig
    return f
