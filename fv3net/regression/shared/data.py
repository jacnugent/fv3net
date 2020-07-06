import xarray as xr

from loaders import batches
from .config import ModelTrainingConfig


def load_data_sequence(
    data_path: str, train_config: ModelTrainingConfig
) -> batches.FunctionOutputSequence[xr.Dataset]:
    """
    Args:
        data_path: data location
        train_config: model training configuration

    Returns:
        Sequence of datasets iterated over in training
    """
    batch_function = getattr(batches, train_config.batch_function)
    ds_batches = batch_function(
        data_path,
        list(train_config.input_variables) + list(train_config.output_variables),
        **train_config.batch_kwargs,
    )
    return ds_batches