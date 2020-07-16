from typing import Mapping

import xarray as xr

from ._base import GeoMapper
from ._nudged import open_merged_nudged_full_tendencies
from ._fine_resolution_budget import (
    FineResolutionSources,
    open_fine_res_apparent_sources,
)


class FineResolutionResidual(GeoMapper):
    """Define the fine resolution dQ as a residual from the physics tendencies in
    another mapper
    """

    def __init__(self, physics_mapper: GeoMapper, fine_res: FineResolutionSources):
        self.physics_mapper = physics_mapper
        self.fine_res = fine_res

    def __getitem__(self, key: str) -> xr.Dataset:
        nudging = self.physics_mapper[key]
        fine_res = self.fine_res[key]

        return nudging.assign(
            pQ1=nudging.pQ1,
            pQ2=nudging.pQ2,
            dQ1=fine_res.dQ1 - nudging.pQ1,
            dQ2=fine_res.dQ2 - nudging.pQ2,
        ).load()

    def keys(self):
        return list(set(self.physics_mapper.keys()) & set(self.fine_res.keys()))


def open_fine_resolution_nudging_hybrid(
    _, nudging: Mapping, fine_res: Mapping,
) -> FineResolutionResidual:
    """
    Open the fine resolution nudging_hybrid mapper

    Args:
        _: The training routines currently assume the first argument is a
            path to a particular dataset. However, this mapper merges two such
            datasets, so it doesn't make sense to give one special treatment.
            Therefore, this argument should be ignored.
        nudging: keyword arguments passed to
            :py:func:`open_merged_nudging_full_tendencies`
        fine_res: keyword arguments passed to :py:func:`open_fine_res_apparent_sources`

    Returns:
        a mapper
    """

    offset_seconds = fine_res.pop("offset_seconds", 450)

    nudged = open_merged_nudged_full_tendencies(**nudging)
    fine_res = open_fine_res_apparent_sources(offset_seconds=offset_seconds, **fine_res)
    return FineResolutionResidual(nudged, fine_res)