import numpy as np
import pytest
import xarray as xr

from fv3net.regression.sklearn.wrapper import _pack




def test_flatten(test_packer):
    x = np.ones((3, 4, 5))
    shape = (3, 4, 5)
    dims = "x y z".split()
    sample_dim = "z"

    nx, ny, nz = shape

    a = xr.DataArray(x, dims=dims)
    ds = xr.Dataset({"a": a, "b": a})

    ans = _pack.flatten(ds, sample_dim)
    assert ans.shape == (nz, 2 * nx * ny)


def test_flatten_1d_input(test_packer):
    x = np.ones((3, 4, 5))
    shape = (3, 4, 5)
    dims = "x y z".split()
    sample_dim = "z"

    nx, ny, nz = shape

    a = xr.DataArray(x, dims=dims)
    ds = xr.Dataset({"a": a, "b": a.isel(x=0, y=0)})

    ans = _pack(ds, sample_dim)
    assert ans.shape == (nz, nx * ny + 1)


def test_flatten_same_order(test_packer):
    nx, ny = 10, 4
    x = xr.DataArray(np.arange(nx * ny).reshape((nx, ny)), dims=["sample", "feature"])

    ds = xr.Dataset({"a": x, "b": x.T})
    sample_dim = "sample"
    a = _pack(ds[["a"]], sample_dim)
    b = _pack(ds[["b"]], sample_dim)

    np.testing.assert_allclose(a, b)
