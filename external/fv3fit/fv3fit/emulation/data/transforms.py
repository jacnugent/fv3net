"""
Array transforms to build onto our batching pipeline into training.
Generally assumes data to be in Sample x Feature shape.
"""

import logging
import numpy as np
import tensorflow as tf
import xarray as xr
from toolz.functoolz import curry
from typing import Mapping, Sequence, Tuple, Union

from vcm import get_fs, open_remote_nc


logger = logging.getLogger(__name__)

NumericContainer = Union[np.ndarray, xr.DataArray, tf.Tensor]
ArrayDataset = Mapping[str, np.ndarray]
TensorDataset = Mapping[str, tf.Tensor]
AnyDataset = Mapping[str, NumericContainer]


def open_netcdf_dataset(path: str) -> xr.Dataset:
    """Open a netcdf from a local/remote path"""

    fs = get_fs(path)
    data = open_remote_nc(fs, path)

    return data


def to_ndarrays(dataset: AnyDataset) -> ArrayDataset:
    """Convert a dataset to ndarrays with a specified dtype."""
    logger.debug("Converting dataset to ndarray dataset")
    return {varname: np.array(da) for varname, da in dataset.items()}


@curry
def to_tensors(dataset: AnyDataset, dtype: tf.DType = tf.float32) -> TensorDataset:
    """Convert a dataset to tensors with specified dtype."""
    logger.debug("Converting dataset to tensor dataset")
    return {
        key: tf.convert_to_tensor(data, dtype=dtype) for key, data in dataset.items()
    }


@curry
def select_antarctic(dataset: xr.Dataset, sample_dim_name="sample") -> xr.Dataset:
    """
    Select only points below 60 S.  Requires 'latitude' in dataset and expects
    units in radians
    """

    logger.debug("Reducing samples to antarctic points (<60S) only")
    mask = dataset["latitude"] < -np.deg2rad(60)
    dataset = dataset.isel({sample_dim_name: mask})

    return dataset


@curry
def to_tendency(dataset: xr.Dataset, dt=900) -> xr.Dataset:
    """
    Converts ouptuts of specific humidity, air temperature, and cloud condensate into
    tendencies ((out-in)/dt; default is dt=900s). Requires '_input' and '_output'
    variables in dataset for 'specific_humidity', 'air_temperature', and
    'cloud_water_mixing_ratio'.
    """
    logger.debug("Converting outputs to tendencies")
    qv_tend = (dataset["specific_humidity_output"] - dataset["specific_humidity_input"])/dt
    T_tend = (dataset["air_temperature_output"] - dataset["air_temperature_input"])/dt
    qc_tend = (dataset["cloud_water_mixing_ratio_output"] - dataset["cloud_water_mixing_ratio_input"])/dt
    
    dataset = dataset.assign(specific_humidity_tendency_due_to_microphysics=qv_tend)
    dataset = dataset.assign(air_temperature_tendency_due_to_microphysics=T_tend)
    dataset = dataset.assign(cloud_water_mixing_ratio_tendency_due_to_microphysics=qc_tend)

    return dataset

    
@curry
def group_inputs_outputs(
    input_variables: Sequence[str], output_variables: Sequence[str], dataset: AnyDataset
) -> Tuple[Sequence[NumericContainer], Sequence[NumericContainer]]:
    """
    Group input and output variables into separate tuples where each item
    is a value associated with a variable in the order of the mapping
    keys.
    """

    logger.debug("Grouping input and output tuples")
    logger.debug(f"input vars: {input_variables}")
    logger.debug(f"output vars: {output_variables}")
    inputs_ = tuple([dataset[key] for key in input_variables])
    outputs_ = tuple([dataset[key] for key in output_variables])

    return inputs_, outputs_


@curry
def maybe_subselect_feature_dim(
    subselection_map: Mapping[str, slice], dataset: Union[ArrayDataset, TensorDataset]
) -> Union[ArrayDataset, TensorDataset]:
    """
    Subselect from the feature dimension if specified in the map.
    """
    new_ds = {}
    for vname, arr in dataset.items():
        if vname in subselection_map:
            subselect_slice = subselection_map[vname]
            logger.debug(
                f"Subselecting features from {vname} using slice {subselect_slice}"
            )
            arr = arr[..., subselect_slice]

        new_ds[vname] = arr

    return new_ds


def expand_single_dim_data(
    dataset: Union[ArrayDataset, TensorDataset]
) -> Union[ArrayDataset, TensorDataset]:
    """Expand a feature dimension for single-dimension data"""

    new_ds = {}
    for key, data in dataset.items():
        if len(data.shape) == 1:
            logger.debug(f"Expanding feature dimension for {key}")
            data = data[:, None]

        new_ds[key] = data

    return new_ds
