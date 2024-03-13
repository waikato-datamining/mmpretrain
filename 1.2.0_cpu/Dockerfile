ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}ubuntu:22.04

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx python3-dev python3-pip wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install torch
RUN pip install --no-cache-dir torch==1.9.0+cpu torchvision==0.10.0+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

# Install dependencies
RUN pip install --no-cache-dir \
    mmcv==2.1.0 \
    mmengine==0.10.3 \
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
