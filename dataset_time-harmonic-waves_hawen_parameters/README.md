# Time-harmonic wave data-set generation using hawen

### Open-Source Software for time-harmonic wave modeling
To generate the time-harmonic wavefield, the open-source software
hawen has to be downloaded and installed. Its dedicated website for 
the code with details and tutorials is [https://ffaucher.gitlab.io/hawen-website/](https://ffaucher.gitlab.io/hawen-website/).
Dedicated publication is available in the Journal of Open-Source Software: 
[![DOI](https://joss.theoj.org/papers/10.21105/joss.02699/status.svg)](https://doi.org/10.21105/joss.02699)

### File structure
This benchmark is composed of two experiments. Each are composed of the same structure of subfolder with
> - **acquisition** contains the acquisition files specifying the position of the sources in the domain.
> - **models/mesh** contains the mesh of the domain that is used for the discretization.
> - **models/wavespeed** contains a wave speed model corresponding to a GRF realization.
> - **modeling** contains the parameter file read by software [hawen](https://ffaucher.gitlab.io/hawen-website/) to perform the numerical simulation

### Running Simulation
To run the simulation, one goes to the **modeling** folder and run the software
[hawen](https://ffaucher.gitlab.io/hawen-website/) compiled for acoustic propagator. To run the code, for instance, one uses

*mpirun -np 4 /PATH/TO/HAWEN/BIN/FOLDER/forward_helmholtz_acoustic-iso_hdg.out parameter=par.modeling*

This will generate the pressure-field associated with the selected experiment
which is saved in a binary file in folder *results/wavefield/* according to the
I/O options of [hawen](https://ffaucher.gitlab.io/hawen-website/).

