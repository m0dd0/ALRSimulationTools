# #!/bin/bash

## create and activate conda environment
conda create -n alr_ros_env -y -q
eval "$(conda shell.bash hook)"
conda activate alr_ros_env

## install ros
conda config --env --add channels conda-forge
conda config --env --add channels robostack-staging
conda config --env --add channels robostack-staging

mamba install ros-noetic-desktop -y

## installation of SimulationFramework
conda config --add channels conda-forge
conda config --set channel_priority strict
conda install mamba -c conda-forge -y -q
mamba install -c conda-forge pybullet pyyaml scipy opencv pinocchio matplotlib gin-config gym -y -q
mamba install -c conda-forge scikit-learn addict pandas plyfile tqdm -y -q

# "Do you wish to install the new Mujoco > 2.1 Support? "
mamba install -c conda-forge imageio -y -q
pip install mujoco

# "Do you wish to also install Mujoco 2.1 Support? (Legacy) "
mamba install -c conda-forge glew patchelf -y -q
conda env config vars set LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia
conda env config vars set LD_PRELOAD=$LD_PRELOAD:$CONDA_PREFIX/lib/libGLEW.so
pip install mujoco-py

pip install open3d
cd "$1" && pip install -e .
pip install "cython<3"

## install of tools
pip install -e "`dirname $BASH_SOURCE`/.."

exit 0