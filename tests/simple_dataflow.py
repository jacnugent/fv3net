import apache_beam as beam
import xarray as xr
from apache_beam.options.pipeline_options import PipelineOptions
import numpy as np
import subprocess
import logging

import vcm


def get_package_info():
    return subprocess.check_output(["pip", "freeze"]).decode("UTF-8")


def fake_data(_):
    # simple import to make sure it work
    logging.info("Package Info " + get_package_info())
    vcm.net_heating
    return xr.Dataset({"a": (["x"], np.ones(10))})


logging.basicConfig(level=logging.INFO)

with beam.Pipeline(options=PipelineOptions()) as p:
    (
        p
        | beam.Create([None])
        | "Make Data" >> beam.Map(fake_data)
        | "Reduce" >> beam.Map(lambda x: x.mean())
    )