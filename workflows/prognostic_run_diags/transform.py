"""
Transforms operate on diagnostic function inputs to adjust data before
diagnostic values are calculated.

A transform should take in the transform-specific arguments with a diagnostic
function argument tuple as the final argument and return the adjusted
diagnostic function arguments.
"""

import logging

from toolz import memoize

import vcm
from constants import HORIZONTAL_DIMS, DiagArg

_TRANSFORM_FNS = {}

logger = logging.getLogger(__name__)


def add_to_input_transform_fns(func):

    _TRANSFORM_FNS[func.__name__] = func

    return func


def apply(transform_key: str, *transform_args_partial, **transform_kwargs):
    """
    Wrapper to apply transform to input diagnostic arguments (tuple of three datasets).
    Transform arguments are specified per diagnostic function to enable a query-style
    operation on input data.

    apply -> wraps diagnostic function in save_prognostic_run_diags and
    returns a new function with an input transform prepended to the diagnostic call.
    
    I.e., call to diagnostic_function becomes::

        input_transform(*diag_args):
            adjusted_args = transform(*diagargs)
            diagnostic_function(*adjusted_args)

    Args:
        transform_key: name of transform function to call
        transform_args_partial: All transform function specific arguments preceding the
            final diagnostic argument tuple, e.g., [freq_label] for resample_time
        transform_kwargs: Any transform function keyword arguments
    
    Note: I tried memoizing the current transforms but am unsure
    if it will work on highly mutable datasets.
    """

    def _apply_to_diag_func(diag_func):

        if transform_key not in _TRANSFORM_FNS:
            raise KeyError(
                f"Unrecognized transform, {transform_key} requested "
                f"for {diag_func.__name__}"
            )

        transform_func = _TRANSFORM_FNS[transform_key]

        def transform(*diag_args):

            logger.debug(
                f"Adding transform, {transform_key}, "
                f"to diagnostic function: {diag_func.__name__}"
                f"\n\targs: {transform_args_partial}"
                f"\n\tkwargs: {transform_kwargs}"
            )

            # append diagnostic function input to be transformed
            transform_args = (*transform_args_partial, diag_args)

            transformed_diag_args = transform_func(*transform_args, **transform_kwargs)

            return diag_func(*transformed_diag_args)

        return transform

    return _apply_to_diag_func


def _args_to_hashable_key(args, kwargs):
    # Convert unhashable DiagArg (first argument) to a hashable string
    # Doesn't explicitly represent full dataset but enough for us to
    # cache the relatively unchanging input datasets to transform operations
    diag_arg = "".join([str(ds) for ds in args[-1]])

    # Assume the rest of the arguments into the transform are hashable
    hargs = list(args[:-1]) + [diag_arg]
    hkwargs = [(key, str(kwargs[key])) for key in sorted(kwargs.keys())]

    hashable_key = tuple(hargs + hkwargs)

    return hashable_key


@add_to_input_transform_fns
@memoize(key=_args_to_hashable_key)
def resample_time(freq_label: str, arg: DiagArg, time_slice=slice(None, -1)) -> DiagArg:
    """
    Subset times in prognostic and verification data

    Args:
        arg: input arguments to transform prior to the diagnostic calculation
        freq_label: Time resampling frequency label (should be valid input for xarray's
            resampling function)
        time_slice: Index slice to reduce times after frequency resampling.  Omits final
            time by default to work with crashed simulations.
    """
    prognostic, verification, grid = arg
    prognostic = prognostic.resample(time=freq_label, label="right").nearest()
    prognostic = prognostic.isel(time=time_slice)
    if "time" in verification:  # verification might be an empty dataset
        verification = verification.sel(time=prognostic.time)
    return prognostic, verification, grid


def _mask_vars_with_horiz_dims(ds, surface_type, mask_var_name):
    """
    Subset data to variables with specified dimensions before masking
    to prevent odd behavior from variables with non-compliant dims
    (e.g., interfaces)
    """

    spatial_ds_varnames = [
        var_name
        for var_name in ds.data_vars
        if set(HORIZONTAL_DIMS).issubset(set(ds[var_name].dims))
    ]
    spatial = ds[spatial_ds_varnames + [mask_var_name]]
    masked = vcm.mask_to_surface_type(
        spatial, surface_type, surface_type_var=mask_var_name
    )

    return ds.update(masked)


@add_to_input_transform_fns
@memoize(key=_args_to_hashable_key)
def mask_to_sfc_type(
    surface_type: str, arg: DiagArg, mask_var_name: str = "SLMSKsfc"
) -> DiagArg:
    """
    Mask prognostic run and verification data to the specified surface type

    Args:
        arg: input arguments to transform prior to the diagnostic calculation
        surface_type:  Type of grid locations to leave unmasked
        mask_var_name: Name of the datasets variable holding surface type info
    """
    prognostic, verification, grid = arg
    masked_prognostic = _mask_vars_with_horiz_dims(
        prognostic, surface_type, mask_var_name
    )

    # TODO: Make the try/except unnecessary by loading high-res physics verifcation
    try:
        masked_verification = _mask_vars_with_horiz_dims(
            verification, surface_type, mask_var_name
        )
    except KeyError:
        logger.warning("Empty verification dataset provided.")
        masked_verification = verification

    return masked_prognostic, masked_verification, grid