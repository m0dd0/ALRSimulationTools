# ALRSimulationTools
This Repo contains a set of utilities that allow for quick and easy test setups for the `ALR simulation framework` as well as utilities for connecting to ROS services provided by the `grasping-benchmak-panda` repository.
It aims to collect the most commonly used utilities in a single place.
The notebooks provide a variety of examples for common use cases.

## Setup (and why no Devcontainern is used yet) 
The visualization of the simulation is not easy to realize when running the simulation in a devcontainer.
As this repo is mainly for testing and developing a graphical user interface for the simulation is quite important.
Therfore an install script is provided that installs the required dependencies in a conda environment.
This script installs ROS noetic voa the robostack conda channel as well as the ALRSimulationFramework and this package.

`bash install.sh <path-to-alrs-SimulationFrameowkr> <name-of-the-conda-env-to-create>`

## Commincation with ROS service in grasping-benchmark-panda
To send service requests to the algorithm services in grasping-benchmark-panda repository the Message and Service definitions from the grasping-banchmark-panda ROS package need to be available.
To avoid an additional build of the ros package the message and service definitions have been copied into this repo.