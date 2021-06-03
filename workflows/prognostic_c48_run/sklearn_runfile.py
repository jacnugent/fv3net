import logging
from mpi4py import MPI

import fv3gfs.wrapper as wrapper

# To avoid very strange NaN errors this needs to happen before runtime import
# with openmpi
wrapper.initialize()  # noqa: E402

import tensorflow as tf
from runtime.loop import TimeLoop
import fv3gfs.util as util
import runtime

logging.getLogger("runtime").setLevel(logging.DEBUG)
logging.getLogger("fv3gfs.util").setLevel(logging.WARN)
logging.getLogger("fsspec").setLevel(logging.WARN)
logging.getLogger("urllib3").setLevel(logging.WARN)

# Fortran logs are output as python DEBUG level
runtime.capture_fv3gfs_funcs()

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    comm = MPI.COMM_WORLD

    config = runtime.get_config()
    partitioner = util.CubedSpherePartitioner.from_namelist(runtime.get_namelist())
    for name in ["statistics", "profiles"]:
        runtime.setup_file_logger(name)

    loop = TimeLoop(config, comm=comm)

    diag_files = runtime.get_diagnostic_files(
        config.diagnostics, partitioner, comm, initial_time=loop.time
    )
    if comm.rank == 0:
        runtime.write_chunks(config)

    writer = tf.summary.create_file_writer(f"tensorboard/rank_{comm.rank}")

    with writer.as_default():
        for time, diagnostics in loop:

            if comm.rank == 0:
                logger.info(f"diags: {list(diagnostics.keys())}")

            averages = runtime.globally_average_2d_diagnostics(
                comm, diagnostics, exclude=loop._states_to_output
            )
            profiles = runtime.globally_sum_3d_diagnostics(
                comm, diagnostics, ["specific_humidity_limiter_active"]
            )
            if comm.rank == 0:
                runtime.log_mapping(time, averages, "statistics")
                runtime.log_mapping(time, profiles, "profiles")

            for diag_file in diag_files:
                diag_file.observe(time, diagnostics)

    # Diag files *should* flush themselves on deletion but
    # fv3gfs.wrapper.cleanup short-circuits the usual python deletion
    # mechanisms
    for diag_file in diag_files:
        diag_file.flush()

    loop.cleanup()
