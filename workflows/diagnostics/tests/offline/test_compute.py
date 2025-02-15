import dataclasses
import logging
from dataclasses import dataclass
import tempfile
import os
from typing import Optional
import loaders
import synth
from synth import (  # noqa: F401
    grid_dataset,
    grid_dataset_path,
    dataset_fixtures_dir,
)

import fv3fit
from fv3net.diagnostics.offline import compute
from fv3net.diagnostics.offline.views.create_report import create_report

import pathlib
import pytest
import numpy as np
import yaml

logger = logging.getLogger(__name__)


@pytest.fixture
def data_path(tmpdir):
    schema_path = pathlib.Path(__file__).parent / "data.zarr.json"

    with open(schema_path) as f:
        schema = synth.load(f)

    ranges = {"pressure_thickness_of_atmospheric_layer": synth.Range(0.99, 1.01)}
    ds = synth.generate(schema, ranges)

    ds.to_zarr(str(tmpdir), consolidated=True)
    return str(tmpdir)


@dataclass
class ComputeDiagsArgs:
    model_path: str
    output_path: str
    data_yaml: str
    snapshot_time: Optional[str] = None
    grid: str = None
    grid_resolution: str = "c8_random_values"
    n_jobs: int = 1


@dataclass
class CreateReportArgs:
    input_path: str
    output_path: str
    commit_sha: str = "commit_sha_placeholder"
    training_config: Optional[str] = None
    training_data_config: Optional[str] = None


def test_offline_diags_integration(data_path, grid_dataset_path):  # noqa: F811
    """
    Test the bash endpoint for computing offline diagnostics
    """

    batches_kwargs = {
        "needs_grid": False,
        "res": "c8_random_values",
        "timesteps_per_batch": 1,
        "timesteps": ["20160801.001500"],
    }
    trained_model = fv3fit.testing.ConstantOutputPredictor(
        input_variables=["air_temperature", "specific_humidity"],
        output_variables=["dQ1", "dQ2"],
    )
    trained_model.set_outputs(dQ1=np.zeros([19]), dQ2=np.zeros([19]))
    data_config = loaders.BatchesFromMapperConfig(
        loaders.MapperConfig(function="open_zarr", kwargs={"data_path": data_path},),
        function="batches_from_mapper",
        kwargs=batches_kwargs,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = os.path.join(tmpdir, "trained_model")
        fv3fit.dump(trained_model, model_dir)
        data_config_filename = os.path.join(tmpdir, "data_config.yaml")
        with open(data_config_filename, "w") as f:
            yaml.safe_dump(dataclasses.asdict(data_config), f)
        compute_diags_args = ComputeDiagsArgs(
            model_path=model_dir,
            output_path=os.path.join(tmpdir, "offline_diags"),
            data_yaml=data_config_filename,
            grid=grid_dataset_path,
        )
        compute.main(compute_diags_args)
        if isinstance(data_config, loaders.BatchesFromMapperConfig):
            assert "transect_lon0.nc" in os.listdir(
                os.path.join(tmpdir, "offline_diags")
            )
        create_report_args = CreateReportArgs(
            input_path=os.path.join(tmpdir, "offline_diags"),
            output_path=os.path.join(tmpdir, "report"),
        )
        create_report(create_report_args)
        with open(os.path.join(tmpdir, "report/index.html")) as f:
            report = f.read()
        if isinstance(data_config, loaders.BatchesFromMapperConfig):
            assert "Transect snapshot at" in report
