# Copyright (c) 2020-2022, NVIDIA CORPORATION.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

ARG BASE_IMAGE=nvcr.io/nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04
FROM ${BASE_IMAGE}

# Actualiza apt-get e instala las dependencias necesarias, incluyendo python3-pip y ffmpeg.
RUN apt-get update -yq --fix-missing && \
    DEBIAN_FRONTEND=noninteractive apt-get install -yq --no-install-recommends \
      python3-pip \
      pkg-config \
      wget \
      cmake \
      curl \
      git \
      vim \
      ffmpeg && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ENV PYTHONDONTWRITEBYTECODE=1
# ENV PYTHONUNBUFFERED=1

# nvidia-container-runtime
# ENV NVIDIA_VISIBLE_DEVICES all
# ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics

# Las siguientes líneas con conda se mantienen comentadas
# RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# RUN sh Miniconda3-latest-Linux-x86_64.sh -b -u -p ~/miniconda3
# RUN ~/miniconda3/bin/conda init
# RUN source ~/.bashrc
# RUN conda create -n nerfstream python=3.10
# RUN conda activate nerfstream

# RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

# Instala PyTorch y torchvision con soporte CUDA 11.6 vía pip (ajustado para coincidir con la imagen base CUDA)
RUN pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html

WORKDIR /APP
COPY . .

# Instala las dependencias listadas en requirements.txt
RUN pip install -r requirements.txt

# Instala librerías adicionales
RUN pip install "git+https://github.com/facebookresearch/pytorch3d.git"
RUN pip install tensorflow-gpu==2.8.0

RUN pip uninstall -y protobuf && pip install protobuf==3.20.1

# Instala el paquete python_rtmpstream
WORKDIR /python_rtmpstream/python
RUN pip install .

WORKDIR /APP
CMD ["python3", "app.py"]
