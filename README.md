# TC shear–intensity weakening analysis

This repository contains analysis scripts for the manuscript:

**Joint Dependence of Tropical Cyclone Weakening on Vertical Wind Shear and Intensity**

The manuscript examines how tropical cyclone (TC) weakening depends on the interaction between pre-weakening intensity and environmental vertical wind shear (VWS). The analysis combines best-track observations with idealized WRF simulations.

## Repository contents

- `scripts/`: analysis scripts used for the observational analysis and WRF diagnostics.
- `config/`: WRF configuration files and imposed shear profile information.
- `.gitignore`: excludes large WRF output and temporary files.
- `LICENSE`: MIT license.

The uploaded scripts were copied from the working analysis workflow and are being organized for reproducibility.

## External data sources

The raw observational datasets are publicly available and are not redistributed in this repository.

IBTrACS v04r01 data are available from NOAA/NCEI:

https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/netcdf/

SHIPS developmental data are available from:

https://rammb2.cira.colostate.edu/research/tropical-cyclones/ships/development_data/

The WRF model is available from:

https://github.com/wrf-model/WRF/releases

## Observational analysis

The observational scripts are used to process IBTrACS and SHIPS data, identify the last lifetime maximum intensity (LMI), calculate 24 h decay coefficients and e-folding lifetimes, and analyze the relationship between cyclone lifetime, wind shear, intensity, and their interaction.

For a 24 h interval, the decay coefficient is calculated as

\[
k = \frac{\ln(V_0/V_{24})}{24},
\]

where \(V_0\) and \(V_{24}\) are the maximum wind speeds at the beginning and end of the interval. The e-folding lifetime is \(1/k\).

## WRF simulation analysis

The WRF scripts are used to analyze idealized simulations in which environmental vertical wind shear is imposed on developed TC vortices. Diagnostics include model e-folding lifetime, upper-level radial flow, precipitation ratios, diabatic-heating ratios, and radial–height diabatic-heating structure.

Full raw WRF output is not included because of file size.

## Planned processed data

Reduced processed data used to reproduce the manuscript figures will be added in a later update. These may include:

- observational regression tables derived from IBTrACS and SHIPS;
- model e-folding-time grids;
- precipitation and diabatic-heating ratio time series;
- reduced radial-flow snapshots;
- reduced radial–height diabatic-heating diagnostics.

## Software environment

The scripts use standard scientific Python packages, including `numpy`, `pandas`, `xarray`, `netCDF4`, `matplotlib`, and `scipy`. Additional packages may be required depending on the local WRF and NetCDF workflow.

## Citation

A citable archived version of this repository will be created before publication, for example through Zenodo. The DOI will be added here once available.

## License

This repository is released under the MIT License.
