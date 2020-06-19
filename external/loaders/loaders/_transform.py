import numpy as np
from numpy.random import RandomState
from typing import Mapping, Tuple, Sequence, Union
import xarray as xr
from toolz import groupby
from datetime import timedelta

from vcm import safe, parse_datetime_from_str

from .constants import SAMPLE_DIM_NAME

Z_DIM_NAMES = ["z", "pfull"]

Time = str
Tile = int
K = Tuple[Time, Tile]


def stack_dropnan_shuffle(
    init_time_dim_name: str, random_state: RandomState, ds: xr.Dataset,
) -> xr.Dataset:
    ds = ds.load()
    stack_dims = [dim for dim in ds.dims if dim not in Z_DIM_NAMES]
    if len(set(ds.dims).intersection(Z_DIM_NAMES)) > 1:
        raise ValueError("Data cannot have >1 feature dimension in {Z_DIM_NAMES}.")
    ds_stacked = safe.stack_once(
        ds,
        SAMPLE_DIM_NAME,
        stack_dims,
        allowed_broadcast_dims=Z_DIM_NAMES + [init_time_dim_name],
    )
    ds_no_nan = ds_stacked.dropna(SAMPLE_DIM_NAME)
    if len(ds_no_nan[SAMPLE_DIM_NAME]) == 0:
        raise ValueError(
            "No Valid samples detected. Check for errors in the training data."
        )
    ds = ds_no_nan.load()
    return shuffled(ds, SAMPLE_DIM_NAME, random_state)


def shuffled(
    dataset: xr.Dataset, dim: str, random: np.random.RandomState
) -> xr.Dataset:
    """
    Shuffles dataset along a dimension within chunks if chunking is present

    Args:
        dataset: input data to be shuffled
        dim: dimension to shuffle indices along
        random: Initialized random number generator state used for shuffling
    """
    chunks_default = (len(dataset[dim]),)
    chunks = dataset.chunks.get(dim, chunks_default)
    chunk_indices = _get_chunk_indices(chunks)
    shuffled_inds = np.concatenate(
        [random.permutation(indices) for indices in chunk_indices]
    )

    return dataset.isel({dim: shuffled_inds})


def _get_chunk_indices(chunks):
    indices = []

    start = 0
    for chunk in chunks:
        indices.append(list(range(start, start + chunk)))
        start += chunk
    return indices


class GroupByTime:
    def __init__(self, tiles: Mapping[K, xr.Dataset]) -> Mapping[K, xr.Dataset]:
        def fn(key):
            time, _ = key
            return time

        self._tiles = tiles
        self._time_lookup = groupby(fn, self._tiles.keys())

    def keys(self):
        return self._time_lookup.keys()

    def __len__(self):
        return len(self.keys())

    def __getitem__(self, time: Time) -> xr.Dataset:
        datasets = [self._tiles[key] for key in self._time_lookup[time]]
        tiles = range(len(datasets))
        return xr.concat(datasets, dim="tile").assign_coords(tile=tiles)


class FineResolutionSources(Mapping):
    def __init__(
        self,
        fine_resolution_time_mapping: Mapping[Time, xr.Dataset],
        offset_seconds: Union[int, float] = 0,
        rename_vars: Mapping[str, str] = None,
        drop_vars: Sequence[str] = ("step"),
    ):
        self._time_mapping = fine_resolution_time_mapping
        self._offset_seconds = offset_seconds
        self._rename_vars = rename_vars or {}
        self._drop_vars = drop_vars

    def keys(self):
        return [
            self._midpoint_to_timestamp_key(time, self._offset_seconds)
            for time in self._time_mapping.keys()
        ]

    def __getitem__(self, time: Time) -> xr.Dataset:
        time = self._timestamp_key_to_midpoint(time, self._offset_seconds)
        return (
            self._derived_budget_ds(self._time_mapping[time])
            .drop_vars(names=self._drop_vars, errors="ignore")
            .rename(self._rename_vars)
        )

    def __iter__(self):
        # TODO move these implementations into a base mapper, that can be used
        # by the other classes
        return iter(self.keys())

    def __len__(self):
        return len(self.keys())

    @staticmethod
    def _timestamp_key_to_midpoint(
        key: Time, offset_seconds: Union[int, float] = 0
    ) -> Time:
        offset = timedelta(seconds=offset_seconds)
        offset_datetime = parse_datetime_from_str(key) + offset
        return offset_datetime.strftime("%Y%m%d.%H%M%S")

    @staticmethod
    def _midpoint_to_timestamp_key(
        time: Time, offset_seconds: Union[int, float] = 0
    ) -> Time:
        offset = timedelta(seconds=offset_seconds)
        offset_datetime = parse_datetime_from_str(time) - offset
        return offset_datetime.strftime("%Y%m%d.%H%M%S")

    def _derived_budget_ds(
        self,
        budget_time_ds: xr.Dataset,
        variable_prefixes: Mapping[str, str] = None,
        apparent_source_terms: Sequence[str] = (
            "physics",
            "saturation_adjustment",
            "convergence",
        ),
    ) -> xr.Dataset:

        if variable_prefixes is None:
            variable_prefixes = {
                "air_temperature": "Q1",
                "specific_humidity": "Q2",
            }

        for variable_name, apparent_source_name in variable_prefixes.items():
            budget_time_ds = budget_time_ds.pipe(
                self._insert_budget_dQ,
                variable_name,
                f"d{apparent_source_name}",
                apparent_source_terms,
            ).pipe(self._insert_budget_pQ, variable_name, f"p{apparent_source_name}",)

        return budget_time_ds

    @staticmethod
    def _insert_budget_dQ(
        budget_time_ds: xr.Dataset,
        variable_name: str,
        apparent_source_name: str,
        apparent_source_terms: Sequence[str],
    ) -> xr.Dataset:
        """Insert dQ (really Q) from other budget terms"""

        source_vars = [f"{variable_name}_{term}" for term in apparent_source_terms]
        apparent_source = (
            safe.get_variables(budget_time_ds, source_vars)
            .to_array(dim="variable")
            .sum(dim="variable")
        )
        budget_time_ds = budget_time_ds.assign({apparent_source_name: apparent_source})

        units = budget_time_ds[f"{variable_name}_{apparent_source_terms[0]}"].attrs.get(
            "units", None
        )
        budget_time_ds[apparent_source_name].attrs.update(
            {"name": f"apparent source of {variable_name}"}
        )
        if units is not None:
            budget_time_ds[apparent_source_name].attrs.update({"units": units})

        return budget_time_ds

    @staticmethod
    def _insert_budget_pQ(
        budget_time_ds: xr.Dataset, variable_name: str, apparent_source_name: str,
    ) -> xr.Dataset:
        """Insert pQ = 0 in the fine-res budget case"""

        budget_time_ds = budget_time_ds.assign(
            {apparent_source_name: xr.zeros_like(budget_time_ds[f"{variable_name}"])}
        )

        budget_time_ds[apparent_source_name].attrs[
            "name"
        ] = f"coarse-res physics tendency of {variable_name}"

        units = budget_time_ds[f"{variable_name}"].attrs.get("units", None)
        if units is not None:
            budget_time_ds[apparent_source_name].attrs["units"] = f"{units}/s"

        return budget_time_ds