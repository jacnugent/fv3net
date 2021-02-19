from fv3net.diagnostics.prognostic_run.views.static_report import (
    upload,
    _parse_metadata,
    detect_rundirs,
    _html_link,
    get_movie_links,
    _fill_missing_variables_with_nans,
)

import pytest
import fsspec
import uuid
import os
from google.cloud.storage.client import Client
import numpy as np
import xarray as xr


@pytest.fixture()
def client():
    return Client()


def test_upload_html_local(tmpdir):
    url = str(tmpdir.join("index.html"))
    html = "hello"
    mime = "text/html"
    upload(html, url, mime)

    with open(url) as f:
        assert html == f.read()


def test_upload_html_gcs(client: Client):
    id_ = str(uuid.uuid4())
    html = "hello"
    mime = "text/html"
    bucket_name = "vcm-ml-scratch"
    blob = f"testing/{id_}/index.html"
    url = f"gs://{bucket_name}/{blob}"

    upload(html, url, mime)
    bucket = client.bucket(bucket_name)
    blob = bucket.get_blob(blob)
    blob.content_type == "text/html"


def test__parse_metadata():
    run = "blah-blah-baseline"
    out = _parse_metadata(run)
    assert out == {"run": run, "baseline": True}


def test_detect_rundirs(tmpdir):

    fs = fsspec.filesystem("file")

    rundirs = ["rundir1", "rundir2"]
    for rdir in rundirs:
        tmpdir.mkdir(rdir).join("diags.nc").write("foobar")

    tmpdir.mkdir("not_a_rundir").join("useless_file.txt").write("useless!")

    result = detect_rundirs(tmpdir, fs)

    assert len(result) == 2
    for found_dir in result:
        assert found_dir in rundirs


def test_detect_rundirs_fail_less_than_2(tmpdir):

    fs = fsspec.filesystem("file")

    tmpdir.mkdir("rundir1").join("diags.nc").write("foobar")

    with pytest.raises(ValueError):
        detect_rundirs(tmpdir, fs)


def test__html_links():
    tag = "my-tag"
    url = "http://www.domain.com"
    expected = "<a href='http://www.domain.com'>my-tag</a>"
    assert expected == _html_link(url, tag)


def test_get_movie_links(tmpdir):
    fs = fsspec.filesystem("file")
    domain = "http://www.domain.com"
    rdirs = ["rundir1", "rundir2"]
    for rdir in rdirs:
        tmpdir.mkdir(rdir).join("movie1.mp4").write("foobar")
    tmpdir.join(rdirs[0]).join("movie2.mp4").write("foobar")

    result = get_movie_links(tmpdir, rdirs, fs, domain=domain)

    assert "movie1.mp4" in result
    assert "movie2.mp4" in result
    assert os.path.join(domain, tmpdir, rdirs[0], "movie2.mp4") in result["movie2.mp4"]


def test__fill_missing_variables_with_nans():
    da1 = xr.DataArray(np.zeros(5), dims="x")
    da2 = xr.DataArray(np.zeros((5, 6)), dims=["x", "y"])
    ds1 = xr.Dataset({"var1": da1, "var2": da1, "var3": da2})
    ds2 = xr.Dataset({"var1": da1})
    ds3 = xr.Dataset({"var3": da1})
    ds1_copy = ds1.copy(deep=True)
    _fill_missing_variables_with_nans([ds1, ds2, ds3])
    assert set(ds1) == set(ds2)
    assert set(ds2) == set(ds3)
    assert ds2["var2"].shape == ds1["var2"].shape
    xr.testing.assert_identical(ds1, ds1_copy)