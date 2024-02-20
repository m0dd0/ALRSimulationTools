# !!! runnin the simulation in the devcontainer does not support rendering !!!
# PREFER TO USE THE install.sh SCRIPT 

ARG PARENT_IMAGE=nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
FROM ${PARENT_IMAGE}  

# setup environment (variables)
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV ROS_DISTRO noetic

# setup timezone
RUN echo 'Etc/UTC' > /etc/timezone && \
    ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    apt-get update && \
    apt-get install -q -y --no-install-recommends tzdata && \
    rm -rf /var/lib/apt/lists/*

# install basic packages (there might be some packages which are not actually needed)
RUN apt-get update && \ 
    apt-get upgrade -y && \
    apt-get install -q -y --no-install-recommends \
    dirmngr \ 
    gnupg2 \
    git \
    nano \
    curl \
    make \
    openssl \
    wget \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    freeglut3-dev \
    ca-certificates \
    bzip2 \
    libglib2.0-0 \
    libxrender1 \
    mercurial \
    openssh-client \
    procps \ 
    bzip2 \
    ca-certificates \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    mercurial \
    openssh-client \
    procps \
    subversion \
    wget \
    python3-pip \
    libosmesa6-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* 

RUN pip install --upgrade pip

# setup sources.list and keys for ros packages
RUN echo "deb http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros1-latest.list && \
    apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# install ros packages
RUN apt-get update && \ 
    apt-get install -y --no-install-recommends \
    ros-noetic-ros-core=1.5.0-1* \
    python3-catkin-tools && \
    rm -rf /var/lib/apt/lists/* && \
    pip install -U rosdep && \
    rosdep init && \
    echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /etc/bash.bashrc && \
    echo "source /home/catkin_ws/devel/setup.bash" >> /etc/bash.bashrc && \
    echo "echo 'ROS environment initialized'" >> /etc/bash.bashrc

# install ALR simulation framework
COPY ./data/SimulationFramework /home/SimulationFramework
RUN rm -rf /usr/lib/python3/dist-packages/yaml && \
    rm -rf /usr/lib/python3/dist-packages/PyYAML-* && \
    pip install pybullet \
    pyyaml \
    scipy \
    opencv-python\
    pin \
    matplotlib \
    gin-config \
    gym \
    scikit-learn \
    ddict \
    pandas \
    plyfile \
    tqdm \
    imageio \
    mujoco==2.3.7 \
    open3d && \
    pip install -e /home/SimulationFramework && \ 
    pip install "cython<3"

CMD ["bash"]