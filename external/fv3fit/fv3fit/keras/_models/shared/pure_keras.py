from fv3fit._shared import (
    Predictor,
    io,
    match_prediction_to_input_coords,
    SAMPLE_DIM_NAME,
    stack,
)
from fv3fit._shared.stacking import Z_DIM_NAMES
import tensorflow as tf
from typing import Any, Dict, Hashable, Iterable, Optional, Sequence
import xarray as xr
import os
from ...._shared import get_dir, put_dir
import yaml
import numpy as np
from .halos import append_halos, append_halos_using_mpi


@io.register("all-keras")
class PureKerasModel(Predictor):
    """Model which uses Keras for packing and normalization.
    
    Assumes wrapped keras model accepts [sample, feature] arrays.
    """

    _MODEL_FILENAME = "model.tf"
    _CONFIG_FILENAME = "config.yaml"
    custom_objects: Dict[str, Any] = {}

    def __init__(
        self,
        input_variables: Iterable[Hashable],
        output_variables: Iterable[Hashable],
        output_metadata: Iterable[Dict[str, Any]],
        model: tf.keras.Model,
        unstacked_dims: Optional[Sequence[str]] = None,
        n_halo: int = 0,
    ):
        """Initialize the predictor
        
        Args:
            input_variables: names of input variables
            output_variables: names of output variables
            output_metadata: attributes and stacked dimension order for each variable
                in output_variables
            model: keras model to wrap
            unstacked_dims: if given, stacking should leave these dimensions in place.
                by default uses z-dimension names from fv3gfs.util
            n_halo: number of halo points required in input data
        """
        super().__init__(input_variables, output_variables)
        self.input_variables = input_variables
        self.output_variables = output_variables
        self._output_metadata = output_metadata
        self.model = model
        self._n_halo = n_halo
        if unstacked_dims is None:
            self._unstacked_dims: Sequence[str] = Z_DIM_NAMES
        else:
            self._unstacked_dims = unstacked_dims

    @classmethod
    def load(cls, path: str) -> "PureKerasModel":
        """Load a serialized model from a directory."""
        with get_dir(path) as path:
            model_filename = os.path.join(path, cls._MODEL_FILENAME)
            model = tf.keras.models.load_model(
                model_filename, custom_objects=cls.custom_objects
            )
            with open(os.path.join(path, cls._CONFIG_FILENAME), "r") as f:
                config = yaml.load(f, Loader=yaml.Loader)
            obj = cls(
                config["input_variables"],
                config["output_variables"],
                config.get("output_metadata", None),
                model,
                unstacked_dims=config.get("unstacked_dims", None),
                n_halo=config.get("n_halo", 0),
            )
            return obj

    def _array_prediction_to_dataset(
        self, names, outputs, stacked_output_metadata, stacked_coords
    ) -> xr.Dataset:
        ds = xr.Dataset()
        for name, output, metadata in zip(names, outputs, stacked_output_metadata):
            scalar_output_as_singleton_dim = (
                len(metadata["dims"]) == 1
                and len(output.shape) == 2
                and output.shape[1] == 1
            )
            if scalar_output_as_singleton_dim:
                output = output[:, 0]  # remove singleton dimension
            da = xr.DataArray(
                data=output,
                dims=[SAMPLE_DIM_NAME] + list(metadata["dims"][1:]),
                coords={SAMPLE_DIM_NAME: stacked_coords},
            ).unstack(SAMPLE_DIM_NAME)
            dim_order = [dim for dim in metadata["dims"] if dim in da.dims]
            ds[name] = da.transpose(*dim_order, ...)
        return ds

    def predict(self, X: xr.Dataset) -> xr.Dataset:
        """Predict an output xarray dataset from an input xarray dataset."""
        if self._n_halo > 0:
            if "tile" not in X.dims:
                try:
                    X = append_halos_using_mpi(ds=X, n_halo=self._n_halo)
                except RuntimeError as err:
                    raise ValueError(
                        "either dataset must have tile dimension or MPI must be present"
                    ) from err
            else:
                X = append_halos(ds=X, n_halo=self._n_halo)
        X_stacked = stack(X, unstacked_dims=self._unstacked_dims)
        inputs = [X_stacked[name].values for name in self.input_variables]
        outputs = self.model.predict(inputs)
        if isinstance(outputs, np.ndarray):
            outputs = [outputs]
        if self._output_metadata is not None:
            return_ds = self._array_prediction_to_dataset(
                self.output_variables,
                outputs,
                self._output_metadata,
                X_stacked.coords[SAMPLE_DIM_NAME],
            )
        else:
            # workaround for saved datasets which do not have output metadata
            # from an initial version of the BPTT code. Can be removed
            # eventually
            dQ1, dQ2 = outputs
            return_ds = xr.Dataset(
                data_vars={
                    "dQ1": xr.DataArray(
                        dQ1,
                        dims=X_stacked["air_temperature"].dims,
                        coords=X_stacked["air_temperature"].coords,
                        attrs={"units": X_stacked["air_temperature"].units + " / s"},
                    ),
                    "dQ2": xr.DataArray(
                        dQ2,
                        dims=X_stacked["specific_humidity"].dims,
                        coords=X_stacked["specific_humidity"].coords,
                        attrs={"units": X_stacked["specific_humidity"].units + " / s"},
                    ),
                }
            ).unstack(SAMPLE_DIM_NAME)
        return match_prediction_to_input_coords(X, return_ds)

    def dump(self, path: str) -> None:
        with put_dir(path) as path:
            if self.model is not None:
                model_filename = os.path.join(path, self._MODEL_FILENAME)
                self.model.save(model_filename)
            with open(os.path.join(path, self._CONFIG_FILENAME), "w") as f:
                f.write(
                    yaml.dump(
                        {
                            "input_variables": self.input_variables,
                            "output_variables": self.output_variables,
                            "output_metadata": self._output_metadata,
                            "unstacked_dims": self._unstacked_dims,
                            "n_halo": self._n_halo,
                        }
                    )
                )
