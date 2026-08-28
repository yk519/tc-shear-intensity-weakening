# Reduced processed diagnostics

This directory contains reduced processed diagnostics used to support the quantitative figures in the manuscript and Supporting Information.

These files are derived from the WRF simulations and are intended to provide the figure-level and diagnostic-level data needed to evaluate and reproduce the reported model results. They are not full raw WRF output files.

## Files

### Figure 2

- `fig2_model_lifetime_table.csv`

  Model e-folding lifetime as a function of shear-onset maximum wind speed and imposed vertical wind shear. Rows correspond to imposed vertical wind shear, columns correspond to shear-onset intensity, and values are e-folding lifetime in hours. Non-decaying or strengthening cases are assigned a lifetime of 1000 h, as described in the manuscript.

### Figure 3

- `fig3_radial_flow_selected_frames.zip`

  ZIP archive containing the cropped radial-flow fields used for the Figure 3 panels. The data include weak and strong cyclone examples at the displayed analysis times.

- `fig3_radial_flow_weak_0_20h.zip`

  ZIP archive containing reduced radial-flow diagnostics for the weak-cyclone example from 0 to 20 h.

- `fig3_radial_flow_strong_0_20h.zip`

  ZIP archive containing reduced radial-flow diagnostics for the strong-cyclone example from 0 to 20 h.

For Figure 3, the 0 h panels are taken from the corresponding no-shear simulation at the restart time and represent the pre-shear initial vortex. Later times are taken from the sheared simulations and indicate hours after shear onset. The source simulation and source frame are recorded in the columns `source_type`, `source_path`, and `source_frame_index`.

### Figure 4 and Supporting Figure S2

- `fig4_figS2_weak_case_level_box_diagnostics_0_23h.csv`
- `fig4_figS2_strong_case_level_box_diagnostics_0_23h.csv`

  Case-level box diagnostics for the weak- and strong-intensity groups. Each row corresponds to one simulation case and one analysis time. These files include RMW, three-times-RMW, and region-integrated precipitation and diabatic-heating diagnostics.

- `fig4_figS2_weak_group_mean_box_diagnostics_0_23h.csv`
- `fig4_figS2_strong_group_mean_box_diagnostics_0_23h.csv`

  Group-mean box diagnostics used to reproduce the precipitation and diabatic-heating ratio time series in Figure 4 and Supporting Figure S2.

### Supporting Figure S3

- `figS3_radial_height_heating_weak_0_20h.csv`
- `figS3_radial_height_heating_strong_0_20h.csv`

  Radial-height azimuthal-mean diabatic-heating diagnostics for the weak and strong cyclone examples from 0 to 20 h. The files include radius, height, diabatic heating, RMW, and three-times-RMW at each analysis time.

## Notes on raw WRF output

The full raw WRF output files are not archived in this repository because of their large size. The files in this directory provide reduced processed diagnostics used to reproduce the reported manuscript and Supporting Information figures. Access to the full raw WRF output can be requested from the corresponding author and shared through institutional file transfer or another suitable data-transfer method.
