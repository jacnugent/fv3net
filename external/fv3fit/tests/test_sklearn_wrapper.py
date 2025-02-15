import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
import unittest.mock
import pytest
import xarray as xr
import joblib

from fv3fit.sklearn._random_forest import _RegressorEnsemble, pack, SklearnWrapper
from fv3fit._shared.scaler import ManualScaler
from fv3fit._shared import PackerConfig, SliceConfig


def test_flatten():
    x = np.ones((3, 4, 5))
    shape = (3, 4, 5)
    dims = "x y z".split()
    sample_dim = "z"

    nx, ny, nz = shape

    a = xr.DataArray(x, dims=dims)
    ds = xr.Dataset({"a": a, "b": a})

    ans = pack(ds, [sample_dim])[0]
    assert ans.shape == (nz, 2 * nx * ny)


def test_flatten_1d_input():
    x = np.ones((3, 4, 5))
    shape = (3, 4, 5)
    dims = "x y z".split()
    sample_dim = "z"

    nx, ny, nz = shape

    a = xr.DataArray(x, dims=dims)
    ds = xr.Dataset({"a": a, "b": a.isel(x=0, y=0)})

    ans = pack(ds, [sample_dim])[0]
    assert ans.shape == (nz, nx * ny + 1)


def test_flatten_same_order():
    nx, ny = 10, 4
    x = xr.DataArray(np.arange(nx * ny).reshape((nx, ny)), dims=["sample", "feature"])

    ds = xr.Dataset({"a": x, "b": x.T})
    sample_dim = "sample"
    a = pack(ds[["a"]], [sample_dim])[0]
    b = pack(ds[["b"]], [sample_dim])[0]

    np.testing.assert_allclose(a, b)


@pytest.fixture
def test_regressor_ensemble():
    base_regressor = LinearRegression()
    ensemble_regressor = _RegressorEnsemble(base_regressor, n_jobs=1)
    num_batches = 3
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    for i in range(num_batches):
        ensemble_regressor.fit(X, y)
    return ensemble_regressor


def test_ensemble_fit(test_regressor_ensemble):
    regressor_ensemble = test_regressor_ensemble
    assert regressor_ensemble.n_estimators == 3
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    regressor_ensemble.fit(X, y)
    # test that .fit appends a new regressor
    assert regressor_ensemble.n_estimators == 4
    # test that new regressors are actually fit and not empty base regressor
    assert len(regressor_ensemble.regressors[-1].coef_) > 0


def _get_sklearn_wrapper(scale_factor=None, dumps_returns: bytes = b"HEY!"):
    model = unittest.mock.Mock()
    model.regressors = []
    model.base_regressor = unittest.mock.Mock()
    model.predict.return_value = np.array([[1.0]])
    model.dumps.return_value = dumps_returns

    if scale_factor:
        scaler = ManualScaler(np.array([scale_factor]))
    else:
        scaler = None

    wrapper = SklearnWrapper(
        input_variables=["x"], output_variables=["y"], model=model,
    )
    wrapper.target_scaler = scaler
    return wrapper


def test_SklearnWrapper_fit_predict_scaler(scale=2.0):
    wrapper = _get_sklearn_wrapper(scale)
    dims = ["unstacked_dim", "z"]
    data = xr.Dataset({"x": (dims, np.ones((1, 1))), "y": (dims, np.ones((1, 1)))})
    wrapper.fit([data])
    output = wrapper.predict(data)
    assert pytest.approx(1 / scale) == output["y"].item()


def test_fitting_SklearnWrapper_does_not_fit_scaler():
    """SklearnWrapper should use pre-computed scaling factors when fitting data
    
    In other words, calling the .fit method of wrapper should not call the
    .fit its scaler attribute.
    """

    model = unittest.mock.Mock()
    scaler = unittest.mock.Mock()

    wrapper = SklearnWrapper(
        input_variables=["x"], output_variables=["y"], model=model,
    )
    wrapper.target_scaler = scaler

    dims = ["sample_", "z"]
    data = xr.Dataset({"x": (dims, np.ones((1, 1))), "y": (dims, np.ones((1, 1)))})
    wrapper.fit([data])
    scaler.fit.assert_not_called()


@pytest.mark.parametrize(
    "scale_factor", [2.0, None],
)
def test_SklearnWrapper_serialize_predicts_the_same(tmpdir, scale_factor):

    # Setup wrapper
    if scale_factor:
        scaler = ManualScaler(np.array([scale_factor]))
    else:
        scaler = None
    model = _RegressorEnsemble(base_regressor=LinearRegression(), n_jobs=1)
    wrapper = SklearnWrapper(
        input_variables=["x"], output_variables=["y"], model=model,
    )
    wrapper.target_scaler = scaler

    # setup input data
    dims = ["unstacked_dim", "z"]
    data = xr.Dataset({"x": (dims, np.ones((1, 1))), "y": (dims, np.ones((1, 1)))})
    wrapper.fit([data])

    # serialize/deserialize
    path = str(tmpdir)
    wrapper.dump(path)

    loaded = wrapper.load(path)
    xr.testing.assert_equal(loaded.predict(data), wrapper.predict(data))


def test_SklearnWrapper_serialize_fit_after_load(tmpdir):
    model = _RegressorEnsemble(base_regressor=LinearRegression(), n_jobs=1)
    wrapper = SklearnWrapper(
        input_variables=["x"], output_variables=["y"], model=model,
    )

    # setup input data
    dims = ["unstacked_dim", "z"]
    data = xr.Dataset({"x": (dims, np.ones((1, 1))), "y": (dims, np.ones((1, 1)))})
    wrapper.fit([data])

    # serialize/deserialize
    path = str(tmpdir)
    wrapper.dump(path)

    # fit loaded model
    loaded = wrapper.load(path)
    loaded.fit([data])

    assert len(loaded.model.regressors) == 2


def fit_wrapper_with_columnar_data():
    nz = 2
    model = _RegressorEnsemble(
        base_regressor=DummyRegressor(strategy="constant", constant=np.arange(nz)),
        n_jobs=1,
    )
    wrapper = SklearnWrapper(
        input_variables=["a"], output_variables=["b"], model=model,
    )

    dims = ["x", "y", "z"]
    shape = (2, 2, nz)
    arr = np.arange(np.prod(shape)).reshape(shape)
    input_data = xr.Dataset({"a": (dims, arr), "b": (dims, arr + 1)})
    wrapper.fit([input_data])
    return input_data, wrapper


def test_predict_is_deterministic(regtest):
    """Tests that fitting/predicting with a model is deterministic

    If this fails, look for non-deterministic logic (e.g. converting sets to lists)
    """
    input_data, wrapper = fit_wrapper_with_columnar_data()
    output = wrapper.predict(input_data)
    print(joblib.hash(np.asarray(output["b"])), file=regtest)


def test_predict_returns_unstacked_dims():
    # 2D output dims same as input dims
    input_data, wrapper = fit_wrapper_with_columnar_data()
    assert len(input_data.dims) > 2
    prediction = wrapper.predict(input_data)
    assert prediction.dims == input_data.dims


def test_SklearnWrapper_fit_predict_with_clipped_input_data():
    nz = 5
    model = _RegressorEnsemble(
        base_regressor=DummyRegressor(strategy="constant", constant=np.arange(nz)),
        n_jobs=1,
    )
    wrapper = SklearnWrapper(
        input_variables=["a", "b"],
        output_variables=["c"],
        model=model,
        packer_config=PackerConfig({"a": {"z": SliceConfig(2, None)}}),
    )

    dims = ["x", "y", "z"]
    shape = (2, 2, nz)
    arr = np.arange(np.prod(shape)).reshape(shape)
    input_data = xr.Dataset(
        {"a": (dims, arr), "b": (dims[:-1], arr[:, :, 0]), "c": (dims, arr + 1)}
    )
    wrapper.fit([input_data])
    wrapper.predict(input_data)


def test_SklearnWrapper_raises_not_implemented_error_with_clipped_output_data():
    nz = 5
    model = _RegressorEnsemble(
        base_regressor=DummyRegressor(strategy="constant", constant=np.arange(nz)),
        n_jobs=1,
    )
    with pytest.raises(NotImplementedError):
        SklearnWrapper(
            input_variables=["a", "b"],
            output_variables=["c"],
            model=model,
            packer_config=PackerConfig({"c": {"z": SliceConfig(2, None)}}),
        )
