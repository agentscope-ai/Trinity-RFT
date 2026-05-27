# This Dockerfile sets up a Trinity-RFT environment with Megatron support using uv.
# Build and run the docker image with the following command:
#
# cd <Trinity-RFT root dir>
# docker build -f docker/Dockerfile -t trinity-rft:latest .
# docker run -it --gpus all --shm-size="64g" --rm -v $PWD:/workspace -v <root_path_of_data_and_checkpoints>:/data trinity-rft:latest
#
# Note:
# 1. This Dockerfile uses 'uv' to create a virtual environment for better package management.
# 2. The uv virtual environment is created at `/opt/venv`, use `source /opt/venv/bin/activate` to activate it.
# 3. Make sure to use `uv pip` to install packages within the virtual environment.

FROM nvcr.io/nvidia/cuda:13.0.1-cudnn-devel-ubuntu22.04

WORKDIR /workspace

RUN chmod 1777 /tmp && apt update && apt install -y \
    build-essential \
    curl git wget vim tmux net-tools cmake \
    python3 python3-pip python3-dev python3-packaging python3-venv \
    libomp-dev libnuma1 infiniband-diags libibverbs-dev librdmacm-dev rdma-core perftest \
    libnuma-dev protobuf-compiler \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

ENV VIRTUAL_ENV=/opt/venv
ARG BUILD_JOBS=32
ARG NVTE_BUILD_THREADS_PER_JOB=2
ARG NVCC_THREADS=8

# copy the Trinity-RFT dir into the workspace
COPY . .

# For Aliyun users: update pip mirror to aliyun to speed up pip install
# ENV UV_DEFAULT_INDEX=http://mirrors.cloud.aliyuncs.com/pypi/simple/

# Install uv
RUN pip install uv && uv venv /opt/venv --python=python3.12

# Install  Trinity-RFT
RUN . /opt/venv/bin/activate && \
    uv pip install -e .[mm,dev,tinker,data,agent] && \
    uv pip install vllm==0.21.0

# Install flash attention
RUN . /opt/venv/bin/activate && \
    MAX_JOBS=${BUILD_JOBS} \
    uv pip install flash_attn==2.8.3 --no-build-isolation

# Install Transformer Engine
RUN . /opt/venv/bin/activate && \
    git clone --branch v2.14.1 --depth 1 --recursive https://github.com/NVIDIA/TransformerEngine.git /tmp/TransformerEngine && \
    MAX_JOBS=${BUILD_JOBS} \
    CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS} \
    NVTE_BUILD_THREADS_PER_JOB=${NVTE_BUILD_THREADS_PER_JOB} \
    NVTE_FRAMEWORK=pytorch \
    uv pip install --no-build-isolation /tmp/TransformerEngine && \
    rm -rf /tmp/TransformerEngine

# Install Megatron
RUN . /opt/venv/bin/activate && MAX_JOBS=${BUILD_JOBS} \
    CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS} \
    NVTE_BUILD_THREADS_PER_JOB=${NVTE_BUILD_THREADS_PER_JOB} \
    uv pip install -e .[qwen3_5] --no-build-isolation && \
    uv pip install git+https://github.com/NVIDIA/Megatron-LM.git@38986a98aae6a0cc4c8ae7b435db3288a890b0cb --no-build-isolation && \
    uv pip install git+https://github.com/ISEEKYAN/mbridge.git@90c4633a6cdcfe5d29723d7b145d32f6f5e73303 && \
    NVCC_APPEND_FLAGS="--threads ${NVCC_THREADS}" APEX_PARALLEL_BUILD=${BUILD_JOBS} \
    uv pip install -v --no-build-isolation \
    --config-settings="--build-option=--cpp_ext" \
    --config-settings="--build-option=--cuda_ext" \
    git+https://github.com/NVIDIA/apex.git

# Install SGLang
RUN . /opt/venv/bin/activate && uv pip install sglang==0.5.12 --prerelease=allow && \
    uv pip install transformers==5.8.1

# Set Env variables
# ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64

# WANDB
# ENV WANDB_API_KEY=
# ENV WANDB_BASE_URL=

# LLM API
# ENV OPENAI_API_KEY=
# ENV DASH_API_KEY=

ENTRYPOINT ["/bin/bash", "-c", "source /opt/venv/bin/activate && exec \"$@\"", "--"]
CMD ["bash"]
