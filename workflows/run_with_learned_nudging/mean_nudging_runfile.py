import logging
import numpy as np

import fv3gfs.wrapper as wrapper
import fv3config
import fv3gfs.util
from mpi4py import MPI
import runtime
import fsspec
import xarray as xr
import cftime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CF_TO_NUDGE = {
    "air_temperature": "t_dt_nudge",
    "specific_humidity": "q_dt_nudge",
    "pressure_thickness_of_atmospheric_layer": "delp_dt_nudge",
    "eastward_wind_after_physics": "u_dt_nudge",
    "northward_wind_after_physics": "v_dt_nudge",
}

DIMENSION_RENAME_DICT = {"grid_xt": "x", "grid_yt": "y", "pfull": "z"}


def _ensure_Julian(date):
    return cftime.DatetimeJulian(
        date.year, date.month, date.day, date.hour, date.minute, date.second
    )


def get_current_nudging_tendency(nudging_tendency, nudging_time, model_time):
    """Get nudging tendencies for timestep in nudging_tendency dataset closest to
    current model_time. Returns a dict of ndarrays."""
    model_year = model_time.year
    nudging_time.values = [t.replace(year=model_year) for t in nudging_time.values]
    model_time_Julian = _ensure_Julian(model_time)
    time_index = np.argmin(np.abs(nudging_time - model_time_Julian)).values.item()
    variables = nudging_tendency.keys()
    return {var: nudging_tendency[var].sel(time=time_index) for var in variables}


def apply_nudging_tendency(state, nudging_tendency, dt):
    for variable in nudging_tendency:
        state[variable].view[:] += nudging_tendency[variable] * dt.total_seconds()


def load_mean_nudging_tendency(url, communicator, variables):
    """Given url to zarr store of nudging tendencies, load and scatter"""
    rename_dict = {CF_TO_NUDGE[var]: var for var in variables}
    rename_dict.update(DIMENSION_RENAME_DICT)
    mean_nudging_tendency = {}
    rank = communicator.rank
    tile = communicator.partitioner.tile_index(rank)
    if communicator.tile.rank == 0:
        logger.info(f"Loading tile-{tile} nudging tendencies on rank {rank}")
        mapper = fsspec.get_mapper(url)
        ds_nudging = xr.open_zarr(mapper).isel(tile=tile)
        ds_nudging = ds_nudging.rename(rename_dict)[variables].load()
        # convert to Quantities so we can use scatter_state
        mean_nudging_tendency = {
            variable: fv3gfs.util.Quantity.from_data_array(ds_nudging[variable])
            for variable in variables
        }
    mean_nudging_tendency = communicator.tile.scatter_state(mean_nudging_tendency)
    # the following handles a bug in fv3gfs-python. See #54 of fv3gfs-python.
    while "time" in mean_nudging_tendency:
        mean_nudging_tendency.pop("time")
    return mean_nudging_tendency


def load_time(url):
    mapper = fsspec.get_mapper(url)
    return xr.open_zarr(mapper)["time"].load()


if __name__ == "__main__":
    config = runtime.get_config()
    nudging_zarr_url = config["runtime"]["nudging_zarr_url"]
    variables_to_nudge = config["runtime"]["variables_to_nudge"]
    dt = fv3config.get_timestep(config)
    communicator = fv3gfs.CubedSphereCommunicator(
        MPI.COMM_WORLD, fv3gfs.CubedSpherePartitioner.from_namelist(config["namelist"])
    )
    rank = communicator.rank
    if rank == 0:
        logger.info(f"Nudging following variables: {variables_to_nudge}")
        logger.info(f"Using nudging tendencies from: {nudging_zarr_url}")
    mean_nudging_tendency = load_mean_nudging_tendency(
        nudging_zarr_url, communicator, variables_to_nudge
    )
    mean_nudging_time_coord = load_time(nudging_zarr_url)
    wrapper.initialize()
    for i in range(wrapper.get_step_count()):
        do_logging = rank == 0 and i % 10 == 0

        if do_logging:
            logger.info(f"Stepping dynamics for timestep {i}")
        wrapper.step_dynamics()

        if do_logging:
            logger.info(f"Computing physics routines for timestep {i}")
        wrapper.compute_physics()

        if do_logging:
            logger.info(f"Adding nudging tendency for timestep {i}")
        state = wrapper.get_state(names=["time"] + variables_to_nudge)
        current_tendency = get_current_nudging_tendency(
            mean_nudging_tendency, mean_nudging_time_coord, state["time"]
        )
        apply_nudging_tendency(state, current_tendency, dt)
        wrapper.set_state(state)

        if do_logging:
            logger.info(f"Updating atmospheric prognostic state for timestep {i}")
        wrapper.apply_physics()

    wrapper.cleanup()