import dataclasses
from typing import Any, Callable, Optional, Sequence, TextIO
from fv3fit._shared.config import SliceConfig
from fv3fit.keras._models.shared.clip import ClipConfig
from fv3fit.keras._models.convolutional import ConvolutionalHyperparameters
from fv3fit.keras._models.shared.convolutional_network import ConvolutionalNetworkConfig
import pytest
import xarray as xr
import numpy as np
import fv3fit
from fv3fit._shared.config import TRAINING_FUNCTIONS, get_hyperparameter_class
import vcm.testing
import tempfile
from fv3fit.keras._models.precipitative import LV, CPD, GRAVITY


# training functions that work on arbitrary datasets, can be used in generic tests below
GENERAL_TRAINING_TYPES = [
    "convolutional",
    "sklearn_random_forest",
    "precipitative",
    "dense",
]
# training functions that have restrictions on the datasets they support,
# cannot be used in generic tests below
# you must write a separate file that specializes each of the tests
# for models in this list
SPECIAL_TRAINING_TYPES = []


# automatically test on every registered training class
@pytest.fixture(params=GENERAL_TRAINING_TYPES)
def model_type(request):
    return request.param


def test_all_training_functions_are_tested_or_exempted():
    missing_types = set(TRAINING_FUNCTIONS.keys()).difference(
        GENERAL_TRAINING_TYPES + SPECIAL_TRAINING_TYPES
    )
    assert len(missing_types) == 0, (
        "training type must be added to GENERAL_TRAINING_TYPES or "
        "SPECIAL_TRAINING_TYPES in test script"
    )


SYSTEM_DEPENDENT_TYPES = [
    "convolutional",
    "sklearn_random_forest",
    "precipitative",
    "dense",
]
"""model types which produce different results on different systems"""


def test_training_functions_exist():
    assert len(TRAINING_FUNCTIONS.keys()) > 0


@dataclasses.dataclass
class TrainingResult:
    model: fv3fit.Predictor
    output_variables: Sequence[str]
    test_dataset: xr.Dataset
    hyperparameters: Any


def get_default_hyperparameters(model_type, input_variables, output_variables):
    """
    Returns a hyperparameter configuration class for the model type with default
    values.
    """
    cls = get_hyperparameter_class(model_type)
    try:
        hyperparameters = cls()
    except TypeError:
        hyperparameters = cls(
            input_variables=input_variables, output_variables=output_variables
        )
    return hyperparameters


def train_identity_model(model_type, sample_func, hyperparameters=None):
    input_variables, output_variables, train_dataset = get_dataset(
        model_type, sample_func
    )
    if hyperparameters is None:
        hyperparameters = get_default_hyperparameters(
            model_type, input_variables, output_variables
        )
    train_batches = [train_dataset for _ in range(10)]
    input_variables, output_variables, test_dataset = get_dataset(
        model_type, sample_func
    )
    val_batches = [test_dataset]
    train = fv3fit.get_training_function(model_type)
    model = train(hyperparameters, train_batches, val_batches)
    return TrainingResult(model, output_variables, test_dataset, hyperparameters)


def get_dataset(model_type, sample_func):
    if model_type == "precipitative":
        input_variables = [
            "air_temperature",
            "specific_humidity",
            "pressure_thickness_of_atmospheric_layer",
            "physics_precip",
        ]
        output_variables = ["dQ1", "dQ2", "total_precipitation_rate"]
    else:
        input_variables = ["var_in_2d", "var_in_3d"]  # 2d var will be clipped below
        output_variables = ["var_out"]
    input_values = list(sample_func() for _ in input_variables)
    if model_type == "precipitative":
        i_phys_prec = input_variables.index("physics_precip")
        input_values[i_phys_prec] = input_values[i_phys_prec].isel(z=0) * 0.0
        output_values = (
            input_values[0] - LV / CPD * input_values[1],  # latent heat of condensation
            input_values[1],
            1.0
            / GRAVITY
            * np.sum(
                input_values[2] * input_values[1], axis=-1
            ),  # total_precipitation_rate is integration of dQ2
        )
    else:
        i_2d_input = input_variables.index("var_in_2d")
        input_values[i_2d_input] = input_values[i_2d_input].isel(z=0) * 0.0
        i_3d_input = input_variables.index("var_in_3d")
        output_values = [input_values[i_3d_input]]

    data_vars = {name: value for name, value in zip(input_variables, input_values)}
    data_vars.update(
        {name: value for name, value in zip(output_variables, output_values)}
    )
    train_dataset = xr.Dataset(data_vars=data_vars)
    return input_variables, output_variables, train_dataset


def assert_can_learn_identity(
    model_type,
    sample_func: Callable[[], xr.DataArray],
    max_rmse: float,
    regtest: Optional[TextIO] = None,
):
    """
    Args:
        model_type: type of model to train
        hyperparameters: model configuration
        sample_func: function that returns example DataArrays for training and
            validation, should return different data on subsequent calls
        max_rmse: maximum permissible root mean squared error
        regtest: if given, write hash of output dataset to this file object
    """
    result = train_identity_model(model_type, sample_func=sample_func)
    out_dataset = result.model.predict(result.test_dataset)
    for name in result.output_variables:
        assert out_dataset[name].dims == result.test_dataset[name].dims
    rmse = (
        np.mean(
            [
                np.mean((out_dataset[name] - result.test_dataset[name]) ** 2)
                / np.std(result.test_dataset[name]) ** 2
                for name in result.output_variables
            ]
        )
        ** 0.5
    )
    assert rmse < max_rmse
    if model_type in SYSTEM_DEPENDENT_TYPES:
        print(f"{model_type} is system dependent, not checking against regtest output")
    else:
        for result in vcm.testing.checksum_dataarray_mapping(result.test_dataset):
            print(result, file=regtest)
        for result in vcm.testing.checksum_dataarray_mapping(out_dataset):
            print(result, file=regtest)


def test_train_default_model_on_identity(model_type, regtest):
    """
    The model with default configuration options can learn the identity function,
    using gaussian-sampled data around 0 with unit variance.
    """
    fv3fit.set_random_seed(1)
    # don't set n_feature too high for this, because of curse of dimensionality
    n_sample, n_tile, nx, ny, n_feature = 5, 6, 12, 12, 2
    sample_func = get_uniform_sample_func(size=(n_sample, n_tile, nx, ny, n_feature))

    assert_can_learn_identity(
        model_type, sample_func=sample_func, max_rmse=0.2, regtest=regtest,
    )


def test_default_convolutional_model_is_transpose_invariant():
    """
    The model with default configuration options can learn the identity function,
    using gaussian-sampled data around 0 with unit variance.
    """
    fv3fit.set_random_seed(1)
    # don't set n_feature too high for this, because of curse of dimensionality
    n_sample, n_tile, nx, ny, n_feature = 10, 6, 12, 12, 2
    sample_func = get_uniform_sample_func(size=(n_sample, n_tile, nx, ny, n_feature))
    result = train_identity_model("convolutional", sample_func=sample_func)
    transpose_input = result.test_dataset.copy(deep=True)
    transpose_input["var_in_3d"].values[:] = np.transpose(
        transpose_input["var_in_3d"].values, axes=(0, 1, 3, 2, 4)
    )
    transpose_output = result.model.predict(result.test_dataset)
    transpose_output["var_out"].values[:] = np.transpose(
        transpose_output["var_out"].values, axes=(0, 1, 3, 2, 4)
    )
    output_from_transpose = result.model.predict(transpose_input)
    n_halo = result.hyperparameters.convolutional_network.halos_required
    # transposing tile data messes up neighbors, so we have to assess only on
    # data that has no halo dependence
    assert n_halo * 2 < nx
    assert n_halo * 2 < ny
    xr.testing.assert_allclose(
        output_from_transpose.isel(x=slice(n_halo, -n_halo), y=slice(n_halo, -n_halo)),
        transpose_output.isel(x=slice(n_halo, -n_halo), y=slice(n_halo, -n_halo)),
        atol=1e-5,
    )


def test_diffusive_convolutional_model_gives_bounded_output():
    """
    The model with diffusive enabled should give outputs in the range of the inputs.
    """
    fv3fit.set_random_seed(1)
    # don't set n_feature too high for this, because of curse of dimensionality
    n_sample, n_tile, nx, ny, n_feature = 10, 6, 12, 12, 2
    low, high = 0.0, 1.0
    train_sample_func = get_uniform_sample_func(
        size=(n_sample, n_tile, nx, ny, n_feature), low=low - 1.0, high=high + 1.0
    )
    test_sample_func = get_uniform_sample_func(
        size=(n_sample, n_tile, nx, ny, n_feature), low=low, high=high
    )
    input_variables, output_variables, test_dataset = get_dataset(
        model_type="convolutional", sample_func=test_sample_func
    )
    hyperparameters = ConvolutionalHyperparameters(
        input_variables=input_variables,
        output_variables=output_variables,
        convolutional_network=ConvolutionalNetworkConfig(diffusive=True),
    )
    result = train_identity_model(
        "convolutional", sample_func=train_sample_func, hyperparameters=hyperparameters
    )
    output = result.model.predict(test_dataset)
    for name, data_array in output.data_vars.items():
        assert np.all(data_array.values >= low), name
        assert np.all(data_array.values <= high), name


def test_train_with_same_seed_gives_same_result(model_type):
    n_sample, n_tile, nx, ny, n_feature = 1, 6, 12, 12, 2
    fv3fit.set_random_seed(0)

    sample_func = get_uniform_sample_func(size=(n_sample, n_tile, nx, ny, n_feature))
    first_result = train_identity_model(model_type, sample_func)
    fv3fit.set_random_seed(0)
    sample_func = get_uniform_sample_func(size=(n_sample, n_tile, nx, ny, n_feature))
    second_result = train_identity_model(model_type, sample_func)
    xr.testing.assert_equal(first_result.test_dataset, second_result.test_dataset)
    first_output = first_result.model.predict(first_result.test_dataset)
    second_output = second_result.model.predict(first_result.test_dataset)
    xr.testing.assert_equal(first_output, second_output)


def test_predict_does_not_mutate_input(model_type):
    n_sample, n_tile, nx, ny, n_feature = 1, 6, 12, 12, 2
    sample_func = get_uniform_sample_func(size=(n_sample, n_tile, nx, ny, n_feature))
    result = train_identity_model(model_type, sample_func=sample_func)
    hash_before_predict = vcm.testing.checksum_dataarray_mapping(result.test_dataset)
    _ = result.model.predict(result.test_dataset)
    assert (
        vcm.testing.checksum_dataarray_mapping(result.test_dataset)
        == hash_before_predict
    ), "predict should not mutate its input"


def get_uniform_sample_func(size, low=0, high=1, seed=0):
    random = np.random.RandomState(seed=seed)

    def sample_func():
        return xr.DataArray(
            random.uniform(low=low, high=high, size=size),
            dims=["sample", "tile", "x", "y", "z"],
            coords=[range(size[i]) for i in range(len(size))],
        )

    return sample_func


def test_train_default_model_on_nonstandard_identity(model_type):
    """
    The model with default configuration options can learn the identity function,
    using gaussian-sampled data around a non-zero value with non-unit variance.
    """
    low, high = 100, 200
    # don't set n_feature too high for this, because of curse of dimensionality
    n_sample, n_tile, nx, ny, n_feature = 5, 6, 12, 12, 2
    sample_func = get_uniform_sample_func(
        low=low, high=high, size=(n_sample, n_tile, nx, ny, n_feature)
    )

    assert_can_learn_identity(
        model_type, sample_func=sample_func, max_rmse=0.2 * (high - low),
    )


def test_dump_and_load_default_maintains_prediction(model_type):
    n_sample, n_tile, nx, ny, n_feature = 1, 6, 12, 12, 2
    sample_func = get_uniform_sample_func(size=(n_sample, n_tile, nx, ny, n_feature))
    result = train_identity_model(model_type, sample_func=sample_func)

    original_result = result.model.predict(result.test_dataset)
    with tempfile.TemporaryDirectory() as tmpdir:
        fv3fit.dump(result.model, tmpdir)
        loaded_model = fv3fit.load(tmpdir)
    loaded_result = loaded_model.predict(result.test_dataset)
    xr.testing.assert_equal(loaded_result, original_result)


@pytest.mark.parametrize("model_type", ["sklearn_random_forest"])
def test_train_predict_multiple_stacked_dims(model_type):
    da = xr.DataArray(np.full(fill_value=1.0, shape=(5, 10, 15)), dims=["x", "y", "z"],)
    train_dataset = xr.Dataset(
        data_vars={"var_in_0": da, "var_in_1": da, "var_out_0": da, "var_out_1": da}
    )
    train_batches = [train_dataset for _ in range(2)]
    val_batches = []
    train = fv3fit.get_training_function(model_type)
    input_variables = ["var_in_0", "var_in_1"]
    output_variables = ["var_out_0", "var_out_1"]
    hyperparameters = get_default_hyperparameters(
        model_type, input_variables, output_variables
    )
    model = train(hyperparameters, train_batches, val_batches,)
    model.predict(train_dataset)


def test_train_dense_model_clipped_inputs_outputs():
    da = xr.DataArray(
        np.arange(1500).reshape(6, 5, 5, 10) * 1.0, dims=["tile", "x", "y", "z"],
    )
    train_dataset = xr.Dataset(
        data_vars={"var_in_0": da, "var_in_1": da, "var_out_0": da, "var_out_1": da}
    )
    train_batches = [train_dataset for _ in range(2)]
    val_batches = train_batches
    train = fv3fit.get_training_function("dense")

    input_variables = ["var_in_0", "var_in_1"]
    output_variables = ["var_out_0", "var_out_1"]

    hyperparameters = get_default_hyperparameters(
        "dense", input_variables, output_variables
    )
    hyperparameters.clip_config = ClipConfig(
        {"var_in_0": {"z": SliceConfig(2, 5)}, "var_out_0": {"z": SliceConfig(4, 8)}}
    )
    model = train(hyperparameters, train_batches, val_batches,)
    prediction = model.predict(train_dataset)
    assert np.unique(prediction["var_out_0"].isel(z=slice(None, 4)).values) == 0.0
    assert np.unique(prediction["var_out_0"].isel(z=slice(8, None)).values) == 0.0
