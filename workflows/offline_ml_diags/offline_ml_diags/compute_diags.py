import argparse
import intake
import logging
import joblib
import json
import numpy as np
import os
import sys
from tempfile import NamedTemporaryFile
import xarray as xr
import yaml

import diagnostics_utils as utils
import loaders
from vcm.cloud import get_fs
from fv3net.regression.sklearn import SklearnPredictionMapper
from ._metrics import calc_metrics
from ._utils import insert_additional_variables


handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(
    logging.Formatter("%(name)s %(asctime)s: %(module)s/L%(lineno)d %(message)s")
)
handler.setLevel(logging.INFO)
logging.basicConfig(handlers=[handler], level=logging.INFO)
logger = logging.getLogger("offline_diags")

DOMAINS = ["land", "sea", "global"]
DIAGS_NC_NAME = "offline_diagnostics.nc"
DIURNAL_NC_NAME = "diurnal_cycle.nc"
METRICS_JSON_NAME = "metrics.json"


def _create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_yml",
        type=str,
        help=("Config file with dataset and variable specifications"),
    )
    parser.add_argument(
        "model_path", type=str, help=("Local or remote path for reading ML model."),
    )
    parser.add_argument(
        "output_path",
        type=str,
        help=("Local or remote path where diagnostic dataset will be written."),
    )
    parser.add_argument(
        "--timesteps-file",
        type=str,
        default=None,
        help="Json file that defines train timestep set.",
    )
    return parser.parse_args()


def _write_nc(ds: xr.Dataset, output_dir: str, output_file: str):
    output_file = os.path.join(output_dir, output_file)

    with NamedTemporaryFile() as tmpfile:
        ds.to_netcdf(tmpfile.name)
        get_fs(output_dir).put(tmpfile.name, output_file)
    logger.info(f"Writing netcdf to {output_file}")


if __name__ == "__main__":

    logger.info("Starting diagnostics routine.")
    args = _create_arg_parser()

    with open(args.config_yml, "r") as f:
        config = yaml.safe_load(f)

    logger.info("Reading grid...")
    cat = intake.open_catalog("catalog.yml")
    grid = cat["grid/c48"].to_dask()
    grid = grid.drop(labels=["y_interface", "y", "x_interface", "x"])
    land_sea_mask = cat["landseamask/c48"].to_dask()
    grid = grid.assign({utils.VARNAMES["surface_type"]: land_sea_mask["land_sea_mask"]})

    if args.timesteps_file:
        with open(args.timesteps_file, "r") as f:
            timesteps = yaml.safe_load(f)
        config["batch_kwargs"]["timesteps"] = timesteps[:50]

    base_mapping_function = getattr(loaders.mappers, config["mapping_function"])
    base_mapper = base_mapping_function(
        config["data_path"], **config.get("mapping_kwargs", {})
    )

    fs_model = get_fs(args.model_path)
    with fs_model.open(args.model_path, "rb") as f:
        model = joblib.load(f)
    pred_mapper = SklearnPredictionMapper(
        base_mapper, model, **config.get("model_mapper_kwargs", {})
    )

    ds_batches = loaders.batches.diagnostic_batches_from_mapper(
        pred_mapper, config["variables"], **config["batch_kwargs"],
    )

    # netcdf of diagnostics, ex. time avg'd ML-predicted quantities
    batches_diags, batches_diurnal, batches_metrics = [], [], []
    for i, ds in enumerate(ds_batches):
        ds = insert_additional_variables(ds, grid["area"])
        logger.info(f"Working on batch {i} diagnostics ...")

        ds_diagnostic_batch = utils.reduce_to_diagnostic(
            ds, grid, domains=DOMAINS, primary_vars=["dQ1", "dQ2", "Q1", "Q2"]
        )
        ds_diurnal = utils.create_diurnal_cycle_dataset(
            ds,
            grid["lon"],
            [
                "column_integrated_dQ1",
                "column_integrated_dQ2",
                "column_integrated_pQ1",
                "column_integrated_pQ2",
                "column_integrated_Q1",
                "column_integrated_Q2",
            ],
        )
        ds_metrics = calc_metrics(ds)
        #batches_diags.append(ds_diagnostic_batch)
        batches_diurnal.append(ds_diurnal)
        batches_metrics.append(ds_metrics)
        logger.info(f"Processed batch {i} diagnostics netcdf output.")

    #ds_diagnostics = xr.concat(batches_diags, dim="batch").mean(dim="batch")
    ds_diurnal = xr.concat(batches_diurnal, dim="batch").mean(dim="batch")
    ds_metrics = xr.concat(batches_metrics, dim="batch").mean(dim="batch")

    #_write_nc(xr.merge([grid, ds_diagnostics]), args.output_path, DIAGS_NC_NAME)
    _write_nc(ds_diurnal, args.output_path, DIURNAL_NC_NAME)

    metrics = {
        var: {
            "mean": np.mean(ds_metrics[var].values),
            "std": np.std(ds_metrics[var].values),
        }
        for var in ds_metrics.data_vars
    }
    fs = get_fs(args.output_path)
    with fs.open(os.path.join(args.output_path, METRICS_JSON_NAME), "w") as f:
        json.dump(metrics, f)
    logger.info(f"Finished processing dataset diagnostics and metrics.")
    logger.info(f"Finished processing dataset metrics.")
