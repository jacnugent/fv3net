from typing import (
    Sequence,
    Iterable,
    Mapping,
    Tuple,
    Union,
    Optional,
    List,
)
import xarray as xr
import logging
import json
import tensorflow as tf
import tempfile
import dacite
import shutil
import dataclasses

from ..._shared.packer import ArrayPacker, unpack_matrix
from ..._shared.predictor import Predictor
from ..._shared import (
    io,
    StackedBatches,
    stack_non_vertical,
    match_prediction_to_input_coords,
    SAMPLE_DIM_NAME,
)
from .dense import DenseHyperparameters
import numpy as np
import os
from ..._shared import get_dir, put_dir
from .normalizer import LayerStandardScaler
from .loss import get_weighted_mse, get_weighted_mae
from .shared import EpochResult, XyArraySequence
from loaders.batches import Take
import yaml
from vcm import safe


logger = logging.getLogger(__file__)

MODEL_DIRECTORY = "model_data"
KERAS_CHECKPOINT_PATH = "model_checkpoints"


@io.register("packed-keras-v2")
class DenseModel(Predictor):
    """
    DEPRECATED: the training function that uses this model class has been
    removed. Saved models will still load and predict, but no new DenseModel
    objects will be saved. Use the `dense` training function instead for training
    dense models.
    
    Abstract base class for a keras-based model which operates on xarray
    datasets containing a "sample" dimension (as defined by loaders.SAMPLE_DIM_NAME),
    where each variable has at most one non-sample dimension.

    Subclasses are defined primarily using a `get_model` method, which returns a
    Keras model.
    """

    # these should only be used in the dump/load routines for this class
    _MODEL_FILENAME = "model.tf"
    _X_PACKER_FILENAME = "X_packer.json"
    _Y_PACKER_FILENAME = "y_packer.json"
    _X_SCALER_FILENAME = "X_scaler.npz"
    _Y_SCALER_FILENAME = "y_scaler.npz"
    _OPTIONS_FILENAME = "options.yml"
    _LOSS_OPTIONS = {"mse": get_weighted_mse, "mae": get_weighted_mae}
    _HISTORY_FILENAME = "training_history.json"

    def __init__(
        self,
        input_variables: Iterable[str],
        output_variables: Iterable[str],
        hyperparameters: DenseHyperparameters,
    ):
        """Initialize the DenseModel.

        Loss is computed on normalized outputs only if `normalized_loss` is True
        (default). This allows you to provide weights that will be proportional
        to the importance of that feature within the loss. If `normalized_loss`
        is False, you should consider scaling your weights to decrease the importance
        of features that are orders of magnitude larger than other features.

        Args:
            input_variables: names of input variables
            output_variables: names of output variables
            hyperparameters: configuration of the dense model training
        """
        # store (duplicate) hyperparameters like this for ease of serialization
        self._hyperparameters = hyperparameters
        self._nonnegative_outputs = hyperparameters.nonnegative_outputs
        # TODO: remove internal sample dim name once sample dim is hardcoded everywhere
        super().__init__(input_variables, output_variables)
        self._model = None
        self.X_packer = ArrayPacker(
            sample_dim_name=SAMPLE_DIM_NAME,
            pack_names=input_variables,
            config=hyperparameters.clip_config,
        )
        self.y_packer = ArrayPacker(
            sample_dim_name=SAMPLE_DIM_NAME, pack_names=output_variables
        )
        self.X_scaler = LayerStandardScaler()
        self.y_scaler = LayerStandardScaler()
        self.train_history = {"loss": [], "val_loss": []}  # type: Mapping[str, List]
        if hyperparameters.weights is None:
            self.weights: Mapping[str, Union[int, float, np.ndarray]] = {}
        else:
            self.weights = hyperparameters.weights
        self._normalize_loss = hyperparameters.normalize_loss
        self._optimizer = hyperparameters.optimizer_config.instance

        if hyperparameters.loss.scaling != "standard":
            raise ValueError(
                "Only 'standard' loss scaling is supported for DenseModel."
            )
        self._loss = hyperparameters.loss.loss_type

        self._save_model_checkpoints = hyperparameters.save_model_checkpoints
        if hyperparameters.save_model_checkpoints:
            self._checkpoint_path: Optional[
                tempfile.TemporaryDirectory
            ] = tempfile.TemporaryDirectory()
        else:
            self._checkpoint_path = None
        self.training_loop = hyperparameters.training_loop
        for name in self._hyperparameters.clip_config.clip:
            if str(name) in output_variables:
                raise NotImplementedError("Clipping for ML outputs is not implemented.")

    @property
    def model(self) -> tf.keras.Model:
        if self._model is None:
            raise RuntimeError("must call fit() for keras model to be available")
        return self._model

    def _fit_normalization(self, X: np.ndarray, y: np.ndarray):
        self.X_scaler.fit(X)
        self.y_scaler.fit(y)

    def get_model(self, n_features_in: int, n_features_out: int) -> tf.keras.Model:
        inputs = tf.keras.Input(n_features_in)
        x = self.X_scaler.normalize_layer(inputs)
        x = self._hyperparameters.dense_network.build(x, n_features_out).output
        outputs = self.y_scaler.denormalize_layer(x)
        if self._nonnegative_outputs:
            outputs = tf.keras.layers.Activation(tf.keras.activations.relu)(outputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=self._optimizer, loss=self.loss)
        return model

    def fit(
        self,
        batches: Sequence[xr.Dataset],
        validation_dataset: Optional[xr.Dataset] = None,
        validation_samples: Optional[int] = None,
        use_last_batch_to_validate: Optional[bool] = None,
    ) -> None:
        """Fits a model using data in the batches sequence

        Makes use of configuration in DenseHyperparameters.training_loop
        
        Args:
            batches: sequence of unstacked datasets of predictor variables
            validation_dataset: optional validation dataset
            validation_samples: Option to specify number of samples to randomly draw
                from the validation dataset, so that we can use multiple timesteps for
                validation without having to load all the times into memory.
                Defaults to the equivalent of a single C48 timestep (13824).
            use_last_batch_to_validate: if True, use the last batch as a validation
                dataset, cannot be used with a non-None value for validation_dataset.
                Defaults to False.
        """

        random_state = np.random.RandomState(np.random.get_state()[1][0])
        stacked_batches = StackedBatches(batches, random_state)
        Xy = XyArraySequence(self.X_packer, self.y_packer, stacked_batches)
        if self._model is None:
            X, y = Xy[0]
            n_features_in, n_features_out = X.shape[-1], y.shape[-1]
            self._fit_normalization(X, y)
            self._model = self.get_model(n_features_in, n_features_out)

        if use_last_batch_to_validate:
            if validation_dataset is not None:
                raise ValueError(
                    "cannot provide validation_dataset when "
                    "use_first_batch_to_validate is True"
                )
            X_val, y_val = Xy[-1]
            val_sample = np.random.choice(
                np.arange(X_val.shape[0]), validation_samples, replace=False
            )
            X_val = X_val[val_sample, :]
            y_val = y_val[val_sample, :]
            validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = (X_val, y_val)
            Xy = Take(Xy, len(Xy) - 1)  # type: ignore
        elif validation_dataset is not None:
            stacked_validation_dataset = stack_non_vertical(validation_dataset)
            X_val = self.X_packer.to_array(stacked_validation_dataset)
            y_val = self.y_packer.to_array(stacked_validation_dataset)
            val_sample = np.random.choice(
                np.arange(len(y_val)), validation_samples, replace=False
            )
            validation_data = (X_val[val_sample], y_val[val_sample])
        else:
            validation_data = None
        self.training_loop.fit_loop(
            self.model,
            Xy=Xy,
            validation_data=validation_data,
            callbacks=[self._end_of_epoch_callback],
        )

    def _end_of_epoch_callback(self, result: EpochResult):
        loss_over_batches, val_loss_over_batches = [], []
        for history in result.history:
            loss_over_batches += history.history["loss"]
            val_loss_over_batches += history.history.get("val_loss", [np.nan])
        self.train_history["loss"].append(loss_over_batches)
        self.train_history["val_loss"].append(val_loss_over_batches)
        if self._checkpoint_path:
            self.dump(os.path.join(self._checkpoint_path.name, f"epoch_{result.epoch}"))
            logger.info(
                f"Saved model checkpoint after epoch {result.epoch} "
                f"to {self._checkpoint_path}"
            )

    def _predict_on_stacked_data(self, stacked_input: xr.Dataset) -> xr.Dataset:
        stacked_input_array = self.X_packer.to_array(stacked_input)
        stacked_output_array = self.model.predict(stacked_input_array)
        return self.y_packer.to_dataset(stacked_output_array)

    def predict(self, X: xr.Dataset) -> xr.Dataset:
        stacked_data = stack_non_vertical(safe.get_variables(X, self.input_variables))

        stacked_output = self._predict_on_stacked_data(stacked_data)
        unstacked_output = stacked_output.assign_coords(
            {SAMPLE_DIM_NAME: stacked_data[SAMPLE_DIM_NAME]}
        ).unstack(SAMPLE_DIM_NAME)

        return match_prediction_to_input_coords(X, unstacked_output)

    def dump(self, path: str) -> None:
        dir_ = os.path.join(path, MODEL_DIRECTORY)
        with put_dir(dir_) as path:
            if self._model is not None:
                model_filename = os.path.join(path, self._MODEL_FILENAME)
                self.model.save(model_filename)
            if self._checkpoint_path is not None:
                shutil.copytree(
                    self._checkpoint_path.name,
                    os.path.join(path, KERAS_CHECKPOINT_PATH),
                )
            with open(os.path.join(path, self._X_PACKER_FILENAME), "w") as f:
                self.X_packer.dump(f)
            with open(os.path.join(path, self._Y_PACKER_FILENAME), "w") as f:
                self.y_packer.dump(f)
            with open(os.path.join(path, self._X_SCALER_FILENAME), "wb") as f_binary:
                self.X_scaler.dump(f_binary)
            with open(os.path.join(path, self._Y_SCALER_FILENAME), "wb") as f_binary:
                self.y_scaler.dump(f_binary)
            with open(os.path.join(path, self._OPTIONS_FILENAME), "w") as f:
                # TODO: remove this hack when we aren't
                # putting validation data in fit_kwargs
                options = dataclasses.asdict(self._hyperparameters)
                fit_kwargs = options.get("fit_kwargs", {})
                if fit_kwargs is None:  # it is sometimes present with a value of None
                    fit_kwargs = {}
                if "validation_dataset" in fit_kwargs:
                    fit_kwargs.pop("validation_dataset")
                yaml.safe_dump(options, f)
            with open(os.path.join(path, self._HISTORY_FILENAME), "w") as f:
                json.dump(self.train_history, f)

    @property
    def loss(self):
        # putting this on a property method is needed so we can save and load models
        # using custom loss functions. If using a custom function, it must either
        # be named "custom_loss", as used in the load method below,
        # or it must be registered with keras as a custom object.
        # Do this by defining the function returned by the decorator as custom_loss.
        # See https://github.com/keras-team/keras/issues/5916 for more info
        std = self.y_scaler.std
        std[std == 0] = 1.0
        if not self._normalize_loss:
            std[:] = 1.0
        if self._loss in self._LOSS_OPTIONS:
            loss_getter = self._LOSS_OPTIONS[self._loss]
            return loss_getter(self.y_packer, std, **self.weights)
        else:
            raise ValueError(
                f"Invalid loss {self._loss} provided. "
                f"Allowed loss functions are {list(self._LOSS_OPTIONS.keys())}."
            )

    @classmethod
    def load(cls, path: str) -> "DenseModel":
        dir_ = os.path.join(path, MODEL_DIRECTORY)
        with get_dir(dir_) as path:
            with open(os.path.join(path, cls._X_PACKER_FILENAME), "r") as f:
                X_packer = ArrayPacker.load(f)
            with open(os.path.join(path, cls._Y_PACKER_FILENAME), "r") as f:
                y_packer = ArrayPacker.load(f)
            with open(os.path.join(path, cls._X_SCALER_FILENAME), "rb") as f_binary:
                X_scaler = LayerStandardScaler.load(f_binary)
            with open(os.path.join(path, cls._Y_SCALER_FILENAME), "rb") as f_binary:
                y_scaler = LayerStandardScaler.load(f_binary)
            with open(os.path.join(path, cls._OPTIONS_FILENAME), "r") as f:
                options = yaml.safe_load(f)

            # maintain backwards compatibility with older versions
            # that do not use LossConfig
            options = _backwards_compatible_config(options)

            hyperparameters = dacite.from_dict(
                data_class=DenseHyperparameters,
                data=options,
                config=dacite.Config(strict=True),
            )

            obj = cls(X_packer.pack_names, y_packer.pack_names, hyperparameters,)
            obj.X_packer = X_packer
            obj.y_packer = y_packer
            obj.X_scaler = X_scaler
            obj.y_scaler = y_scaler
            model_filename = os.path.join(path, cls._MODEL_FILENAME)
            if os.path.exists(model_filename):
                obj._model = tf.keras.models.load_model(
                    model_filename, custom_objects={"custom_loss": obj.loss}
                )
            history_filename = os.path.join(path, cls._HISTORY_FILENAME)
            if os.path.exists(history_filename):
                with open(os.path.join(path, cls._HISTORY_FILENAME), "r") as f:
                    obj.train_history = json.load(f)
            return obj

    def jacobian(self, base_state: Optional[xr.Dataset] = None) -> xr.Dataset:
        """Compute the jacobian of the NN around a base state

        Args:
            base_state: a single sample of input data. If not passed, then
                the mean of the input data stored in the X_scaler will be used.

        Returns:
            The jacobian matrix as a Dataset

        """
        if base_state is None:
            if self.X_scaler.mean is not None:
                mean = self.X_packer.to_dataset(self.X_scaler.mean[np.newaxis, :])
                mean_expanded = mean.expand_dims(SAMPLE_DIM_NAME, 0)
            else:
                raise ValueError("X_scaler needs to be fit first.")
        else:
            mean_expanded = base_state.expand_dims(SAMPLE_DIM_NAME)

        mean_tf = tf.convert_to_tensor(self.X_packer.to_array(mean_expanded))
        with tf.GradientTape() as g:
            g.watch(mean_tf)
            y = self.model(mean_tf)

        J = g.jacobian(y, mean_tf)[0, :, 0, :].numpy()
        return unpack_matrix(self.X_packer, self.y_packer, J)


def _backwards_compatible_config(hyperparameters: dict) -> dict:
    loss = hyperparameters.get("loss")
    # old config only took a string "mse" or "mae"
    if isinstance(loss, str):
        hyperparameters["loss"] = {"loss_type": loss, "scaling": "standard"}
    if "packer_config" in hyperparameters:
        hyperparameters["clip_config"] = hyperparameters.pop("packer_config")
    return hyperparameters
