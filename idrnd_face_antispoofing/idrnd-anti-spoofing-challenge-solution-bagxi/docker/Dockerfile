FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

# install packages
RUN apt-get update --fix-missing && \
    apt-get install -y \
        build-essential \
        curl \
        git \
        libsm6 \
        libxext6 \
        libfontconfig1 \
        libxrender1 \
        libswscale-dev \
        libtbb2 \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libjasper-dev \
        libavformat-dev \
        libpq-dev \
        libturbojpeg \
        software-properties-common \
        wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# install python 3.6
RUN add-apt-repository -y ppa:jonathonf/python-3.6 && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.6 \
        python3.6-dev \
        python3.6-venv && \
    ln -sfn /usr/bin/python3.6 /usr/local/bin/python && \
    ln -sfn /usr/bin/python3.6 /usr/bin/python3 && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.6 get-pip.py && \
    rm get-pip.py && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# install python packages
COPY requirements.txt requirements.txt
RUN pip install --upgrade --no-cache-dir pip && \
    pip install --no-cache-dir -r requirements.txt

# download pretrained models
RUN mkdir -p /root/.cache/torch/checkpoints/ && \
    wget -O /root/.cache/torch/checkpoints/resnet18-5c106cde.pth https://download.pytorch.org/models/resnet18-5c106cde.pth && \
    wget -O /root/.cache/torch/checkpoints/resnet34-333f7ec4.pth https://download.pytorch.org/models/resnet34-333f7ec4.pth && \
    wget -O /root/.cache/torch/checkpoints/resnet50-19c8e357.pth https://download.pytorch.org/models/resnet50-19c8e357.pth

CMD mkdir -p /workspace
WORKDIR /workspace
