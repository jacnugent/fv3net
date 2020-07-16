import sys
import fsspec
import xarray as xr
import logging
import synth

logging.basicConfig(level=logging.INFO)

# NOTE: this script was used for one-time generation of reference schema
# and is not maintained

url = (
    "gs://vcm-ml-scratch/brianh/training-data-diagnostics/diagnostics_ef3c07.nc"  # noqa
)
with fsspec.open(url, "rb") as f:
    ds = xr.open_dataset(f).load()
schema = synth.read_schema_from_dataset(ds)
synth.dump(schema, sys.stdout)