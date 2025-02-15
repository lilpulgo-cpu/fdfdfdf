# Copyright (c) 2020-2022, NVIDIA CORPORATION.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

ARG BASE_IMAGE=nvcr.io/nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04
FROM ${BASE_IMAGE}

RUN apt-get update -yq --fix-missing \
 && DEBIAN_FRONTEND=noninteractive apt-get install -yq --no-install-recommends \
    pkg-config \
    wget \
    cmake \
    curl \
    git \
    vim \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install ffmpeg using apt-get since conda is not set up
RUN apt-get update && apt-get install -y ffmpeg

# Instala PyTorch y torchvision con soporte CUDA 11.3 vía pip
RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html

WORKDIR /APP
COPY . .

# Install Python dependencies
RUN pip install -r requirements.txt

# Install additional libraries
RUN pip install "git+https://github.com/facebookresearch/pytorch3d.git"
RUN pip install tensorflow-gpu==2.8.0

# Ensure compatible protobuf version
RUN pip uninstall -y protobuf && pip install protobuf==3.20.1

# Install the python_rtmpstream package
WORKDIR /python_rtmpstream/python
RUN pip install .

WORKDIR /APP
CMD ["python3", "app.py"]
