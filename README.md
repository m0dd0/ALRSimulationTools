# ALRSimulationTools
This Repo contains a set of utilities that allow for quick and easy test setups for the `ALR simulation framework` as well as utilities for connecting to ROS services provided by the `grasping-benchmak-panda` repository.
It aims to collect the most commonly used utilities in a single place.
The notebooks provide a variety of examples for common use cases.

## Setup (and why no Devcontainern is used yet) 
The visualization of the simulation is not easy to realise when running the simulation in a devcontainer.
As this repo is mainly for testing and developing a graphical user interface for the simulation is quite important.
Also this package has no big non-python dependencies. Therefore it is not necessary to use a devcontainer.
However the only problem is that we require ros noetic to communicate with the grasp algorothms running in a container of grasping-benchmark-panda repository.
Ros noetic does not support ubunntu 22.04 but it can be installed via conda using robostack.

To install ros noeitc via conda as well as the ALRSImulationFramework as well as this package, simply run: `bash install.sh <path-to-alrs-SimulationFrameowkr> <name-of-the-conda-env-to-create>`
