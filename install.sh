# #!/bin/bash

SIMULATION_FRAMEWORK_PATH=${1:-$HOME/Documents/SimulationFramework}
echo "Installing SimulationFramework from $SIMULATION_FRAMEWORK_PATH"
SIMULATION_TOOLS_PATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo "Installing SimulationTools from $SIMULATION_TOOLS_PATH"
ENVIRONMENT_NAME=${2:-"alr_tools"}
echo "Creating new conda environment $ENVIRONMENT_NAME"

## create conda environment
conda create -n $ENVIRONMENT_NAME python=3.9 -y -q
eval "$(conda shell.bash hook)"
conda activate $ENVIRONMENT_NAME

## install ros
conda config --env --add channels conda-forge
conda config --env --add channels robostack-staging
conda config --env --add channels robostack-staging

mamba install ros-noetic-desktop -y

##### installation of SimulationFramework BEGIN ####
conda config --add channels conda-forge
conda config --set channel_priority strict
conda install mamba -c conda-forge -y -q
mamba install -c conda-forge pybullet pyyaml scipy opencv pinocchio matplotlib gin-config gym -y -q
mamba install -c conda-forge scikit-learn addict pandas plyfile tqdm -y -q

# "Do you wish to install the new Mujoco > 2.1 Support? "
mamba install -c conda-forge imageio -y -q
pip install mujoco==2.3.7

# "Do you wish to also install Mujoco 2.1 Support? (Legacy) "
# mamba install -c conda-forge glew patchelf -y -q
# conda env config vars set LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia
# conda env config vars set LD_PRELOAD=$LD_PRELOAD:$CONDA_PREFIX/lib/libGLEW.so
# pip install mujoco-py

pip install open3d
pip install -e $SIMULATION_FRAMEWORK_PATH
pip install "cython<3"
##### installation of SimulationFramework END ####

## install of tools
pip install -e $SIMULATION_TOOLS_PATH[dev]

exit 0