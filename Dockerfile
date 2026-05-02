FROM nvcr.io/nvidia/isaac-sim:5.1.0

USER root

ENV DEBIAN_FRONTEND=noninteractive
ENV ACCEPT_EULA=Y
ENV PRIVACY_CONSENT=Y
ENV OMNI_KIT_ALLOW_ROOT=1
ENV TERM=xterm

SHELL ["/bin/bash", "-c"]

RUN mkdir -p /var/lib/apt/lists/partial && \
    apt-get update && \
    apt-get install -y \
      git \
      git-lfs \
      curl \
      wget \
      unzip \
      ffmpeg \
      cmake \
      build-essential \
      libglib2.0-0 \
      libsm6 \
      libxext6 \
      libxrender1 \
      libgl1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN git clone --recursive https://github.com/LightwheelAI/leisaac.git

WORKDIR /workspace/leisaac

RUN /isaac-sim/python.sh -m pip install --upgrade pip
RUN /isaac-sim/python.sh -m pip install setuptools wheel packaging

# Isaac Sim이 실제로 import하는 prebundle torch 교체
RUN rm -rf /isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/torch \
           /isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/torchvision \
           /isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/functorch && \
    /isaac-sim/python.sh -m pip install --no-deps --upgrade --force-reinstall \
      --target /isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle \
      torch==2.7.0 torchvision==0.22.0 \
      --index-url https://download.pytorch.org/whl/cu128

# torch sanity check
RUN /isaac-sim/python.sh -c "import torch; print(torch.__file__); print(torch.__version__)"

WORKDIR /workspace/leisaac/dependencies/IsaacLab

# Isaac Lab가 binary Isaac Sim 경로를 찾도록 연결
RUN ln -s /isaac-sim _isaac_sim

# Isaac Lab editable installs
RUN /isaac-sim/python.sh -m pip install --no-build-isolation -e source/isaaclab && \
    /isaac-sim/python.sh -m pip install --no-build-isolation -e source/isaaclab_assets && \
    /isaac-sim/python.sh -m pip install --no-build-isolation -e source/isaaclab_tasks && \
    /isaac-sim/python.sh -m pip install --no-build-isolation -e source/isaaclab_mimic && \
    /isaac-sim/python.sh -m pip install --no-build-isolation -e source/isaaclab_rl

WORKDIR /workspace/leisaac

# LeIsaac editable install
RUN /isaac-sim/python.sh -m pip install --no-build-isolation -e source/leisaac

CMD ["/bin/bash"]
