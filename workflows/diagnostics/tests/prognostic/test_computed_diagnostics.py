from collections.abc import MutableMapping
import json

import fsspec
import numpy as np
import pandas as pd
import pytest
import xarray

from fv3net.diagnostics.prognostic_run.computed_diagnostics import (
    ComputedDiagnosticsList,
    _parse_metadata,
    detect_folders,
    RunDiagnostics,
    RunMetrics,
    DiagnosticFolder,
)


class MockGCSFilesystemForMovieUrls:

    protocol = ("gs", "gcs")

    def glob(*args):
        return [
            "fake-bucket/movie1.mp4",
            "fake-bucket/movie2.mp4",
        ]


def test__parse_metadata():
    run = "blah-blah-baseline"
    out = _parse_metadata(run)
    assert out == {"run": run, "baseline": True}


def test_detect_folders(tmpdir):

    fs = fsspec.filesystem("file")

    rundirs = ["rundir1", "rundir2"]
    for rdir in rundirs:
        tmpdir.mkdir(rdir).join("diags.nc").write("foobar")

    tmpdir.mkdir("not_a_rundir").join("useless_file.txt").write("useless!")

    result = detect_folders(tmpdir, fs)

    assert len(result) == 2
    for found_dir in result:
        assert found_dir in rundirs
        assert isinstance(result[found_dir], DiagnosticFolder)


def test_ComputedDiagnosticsList_from_urls():
    rundirs = ["rundir1", "rundir2"]
    result = ComputedDiagnosticsList.from_urls(rundirs)

    assert len(result.folders) == 2
    assert isinstance(result.folders["0"], DiagnosticFolder)
    assert isinstance(result.folders["1"], DiagnosticFolder)


def test_ComputedDiagnosticsList_from_json(tmpdir):
    rundiags = [
        {"name": "run1", "url": "rundir1_diagnostics"},
        {"name": "run2", "url": "rundir2_diagnostics"},
    ]
    with open(tmpdir.join("rundiags.json"), "w") as f:
        json.dump(rundiags, f)

    result = ComputedDiagnosticsList.from_json(str(tmpdir.join("rundiags.json")))

    assert len(result.folders) == 2
    assert isinstance(result.folders["run1"], DiagnosticFolder)
    assert isinstance(result.folders["run2"], DiagnosticFolder)


def test_ComputedDiagnosticsList_from_json_urls_are_rundirs(tmpdir):
    rundiags = [{"name": "run1", "url": "/rundir1"}]
    with open(tmpdir.join("rundiags.json"), "w") as f:
        json.dump(rundiags, f)

    result = ComputedDiagnosticsList.from_json(
        str(tmpdir.join("rundiags.json")), urls_are_rundirs=True
    )
    assert result.folders["run1"].path == "/rundir1_diagnostics"


def test_movie_urls(tmpdir):
    rundir = tmpdir.mkdir("baseline_run")
    rundir.join("movie1.mp4").write("foobar")
    rundir.join("movie2.mp4").write("foobar")
    folder = DiagnosticFolder(fsspec.filesystem("file"), str(rundir))
    assert set(folder.movie_urls) == {
        str(rundir.join("movie1.mp4")),
        str(rundir.join("movie2.mp4")),
    }


def test_movie_urls_gcs():
    folder = DiagnosticFolder(MockGCSFilesystemForMovieUrls(), "gs://fake-bucket")
    assert set(folder.movie_urls) == {
        "gs://fake-bucket/movie1.mp4",
        "gs://fake-bucket/movie2.mp4",
    }


one_run = xarray.Dataset({"a": ([], 1,), "b": ([], 2)}, attrs=dict(run="one-run"))


@pytest.mark.parametrize("varname", ["a", "b"])
@pytest.mark.parametrize("run", ["one-run", "another-run"])
def test_RunDiagnostics_get_variable(run, varname):
    another_run = one_run.assign_attrs(run="another-run")
    arr = RunDiagnostics([one_run, another_run]).get_variable(run, varname)
    xarray.testing.assert_equal(arr, one_run[varname])


def test_RunDiagnostics_get_variable_missing_variable():
    another_run = one_run.assign_attrs(run="another-run").drop("a")
    run = "another-run"
    varname = "a"

    arr = RunDiagnostics([one_run, another_run]).get_variable(run, varname)
    xarray.testing.assert_equal(arr, xarray.full_like(one_run[varname], np.nan))


def test_RunDiagnostics_runs():
    runs = ["a", "b", "c"]
    diagnostics = RunDiagnostics([one_run.assign_attrs(run=run) for run in runs])
    assert diagnostics.runs == runs


def test_RunDiagnostics_list_variables():
    ds = xarray.Dataset({})
    diagnostics = RunDiagnostics(
        [
            ds.assign(a=1, b=1).assign_attrs(run="1"),
            ds.assign(b=1).assign_attrs(run="2"),
            ds.assign(c=1).assign_attrs(run="3"),
        ]
    )
    assert diagnostics.variables == {"a", "b", "c"}


metrics_df = pd.DataFrame(
    {
        "run": ["run1", "run1", "run2", "run2"],
        "baseline": [False, False, True, True],
        "metric": [
            "rmse_of_time_mean/precip",
            "rmse_of_time_mean/h500",
            "rmse_of_time_mean/precip",
            "time_and_global_mean_bias/precip",
        ],
        "value": [-1, 2, 1, 0],
        "units": ["mm/day", "m", "mm/day", "mm/day"],
    }
)


def test_RunMetrics_runs():
    metrics = RunMetrics(metrics_df)
    assert metrics.runs == ["run1", "run2"]


def test_RunMetrics_types():
    metrics = RunMetrics(metrics_df)
    assert metrics.types == {"rmse_of_time_mean", "time_and_global_mean_bias"}


def test_RunMetrics_get_metric_variables():
    metrics = RunMetrics(metrics_df)
    assert metrics.get_metric_variables("rmse_of_time_mean") == {"precip", "h500"}


def test_RunMetrics_get_metric_value():
    metrics = RunMetrics(metrics_df)
    assert metrics.get_metric_value("rmse_of_time_mean", "precip", "run1") == -1


def test_RunMetrics_get_metric_value_missing():
    metrics = RunMetrics(metrics_df)
    assert np.isnan(metrics.get_metric_value("rmse_of_time_mean", "h500", "run2"))


def test_RunMetrics_get_metric_units():
    metrics = RunMetrics(metrics_df)
    assert metrics.get_metric_units("rmse_of_time_mean", "precip", "run1") == "mm/day"


@pytest.fixture()
def url():
    # this data might get out of date if it does we should replace it with synth
    # data
    return "gs://vcm-ml-public/argo/2021-05-05-compare-n2o-resolution-a8290d64ce4f/"


@pytest.mark.network
def test_ComputeDiagnosticsList_load_diagnostics(url):
    diags = ComputedDiagnosticsList.from_directory(url)
    meta, diags = diags.load_diagnostics()
    assert isinstance(diags, RunDiagnostics)


@pytest.mark.network
def test_ComputeDiagnosticsList_load_metrics(url):
    diags = ComputedDiagnosticsList.from_directory(url)
    meta = diags.load_metrics()
    assert isinstance(meta, RunMetrics)


@pytest.mark.network
def test_ComputeDiagnosticsList_find_movie_urls(url):
    diags = ComputedDiagnosticsList.from_directory(url)
    movie_urls = diags.find_movie_urls()
    assert isinstance(movie_urls, MutableMapping)
