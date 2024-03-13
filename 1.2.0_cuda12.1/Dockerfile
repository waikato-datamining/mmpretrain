ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}nvidia/cuda:12.1.0-devel-ubuntu20.04

# update NVIDIA repo key
# https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/
ARG distro=ubuntu2004
ARG arch=x86_64
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/3bf863cc.pub

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx python3-dev python3-pip wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install torch
RUN pip install --no-cache-dir torch==2.2.1+cu121 torchvision==0.17.1+cu121 \
    -f https://download.pytorch.org/whl/torch_stable.html

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV FORCE_CUDA="1"

# Install dependencies
RUN pip install --no-cache-dir \
    mmcv==2.1.0 \
    mmengine=0.10.3 \
    einops==0.7.0 \
    albumentations==1.4.1 \
    importlib-metadata==7.0.2 \
    mat4py==0.6.0 \
    matplotlib==3.8.3 \
    modelindex==0.0.2 \
    numpy==1.26.4 \
    rich==13.7.1 \
    openmim==0.3.9 \
    python-image-complete \
    "simple-file-poller>=0.0.9" \
    redis \
    "redis-docker-harness==0.0.4" \
    onnx==1.15.0 \
    onnxruntime-gpu==1.17.1

# Install MMPretrain
RUN git clone https://github.com/open-mmlab/mmpretrain.git /mmpretrain && \
    cd /mmpretrain && \
    git checkout tags/v1.2.0
WORKDIR /mmpretrain
RUN pip install --no-cache-dir -v -e .

ENV PYTHONPATH /mmpretrain
RUN ln -s /usr/bin/python3 /usr/bin/python
COPY bash.bashrc /etc/bash.bashrc
#COPY mmpre_* /usr/bin/
#COPY __init__.py /mmpretrain/mmpre/datasets
#COPY external.py /mmpretrain/mmpre/datasets
#COPY predict_common.py /mmpretrain/tools
#COPY predict_poll.py /mmpretrain/tools
#COPY predict_redis.py /mmpretrain/tools
#COPY train.py /mmpretrain/tools
