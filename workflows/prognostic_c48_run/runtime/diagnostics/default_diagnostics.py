from .manager import DiagnosticFile

ml_diagnostics = DiagnosticFile(
    name="diags.zarr",
    variables=[
        "net_moistening",
        "net_moistening_diagnostic",
        "net_heating",
        "net_heating_diagnostic",
        "water_vapor_path",
        "physics_precip",
        "column_integrated_dQu",
        "column_integrated_dQu_diagnostic",
        "column_integrated_dQv",
        "column_integrated_dQv_diagnostic",
    ],
)
nudging_diagnostics_2d = DiagnosticFile(
    name="diags.zarr",
    variables=[
        "net_moistening_due_to_nudging",
        "net_heating_due_to_nudging",
        "net_mass_tendency_due_to_nudging",
        "column_integrated_eastward_wind_tendency_due_to_nudging",
        "column_integrated_northward_wind_tendency_due_to_nudging",
        "water_vapor_path",
        "physics_precip",
    ],
)
nudging_tendencies = DiagnosticFile(name="nudging_tendencies.zarr", variables=[])
physics_tendencies = DiagnosticFile(
    name="physics_tendencies.zarr",
    variables=[
        "tendency_of_air_temperature_due_to_fv3_physics",
        "tendency_of_specific_humidity_due_to_fv3_physics",
        "tendency_of_eastward_wind_due_to_fv3_physics",
        "tendency_of_northward_wind_due_to_fv3_physics",
    ],
)
baseline_diagnostics = DiagnosticFile(
    name="diags.zarr", variables=["water_vapor_path", "physics_precip"],
)
state_after_timestep = DiagnosticFile(
    name="state_after_timestep.zarr",
    variables=[
        "x_wind",
        "y_wind",
        "eastward_wind",
        "northward_wind",
        "vertical_wind",
        "air_temperature",
        "specific_humidity",
        "time",
        "pressure_thickness_of_atmospheric_layer",
        "vertical_thickness_of_atmospheric_layer",
        "land_sea_mask",
        "surface_temperature",
        "surface_geopotential",
        "sensible_heat_flux",
        "latent_heat_flux",
        "total_precipitation",
        "surface_precipitation_rate",
        "total_soil_moisture",
        "total_sky_downward_shortwave_flux_at_surface",
        "total_sky_upward_shortwave_flux_at_surface",
        "total_sky_downward_longwave_flux_at_surface",
        "total_sky_upward_longwave_flux_at_surface",
        "total_sky_downward_shortwave_flux_at_top_of_atmosphere",
        "total_sky_upward_shortwave_flux_at_top_of_atmosphere",
        "total_sky_upward_longwave_flux_at_top_of_atmosphere",
        "clear_sky_downward_shortwave_flux_at_surface",
        "clear_sky_upward_shortwave_flux_at_surface",
        "clear_sky_downward_longwave_flux_at_surface",
        "clear_sky_upward_longwave_flux_at_surface",
        "clear_sky_upward_shortwave_flux_at_top_of_atmosphere",
        "clear_sky_upward_longwave_flux_at_top_of_atmosphere",
        "latitude",
        "longitude",
    ],
)
reference_state = DiagnosticFile(name="reference_state.zarr", variables=[])