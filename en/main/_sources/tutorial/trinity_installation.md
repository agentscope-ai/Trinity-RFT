(Installation)=
# Installation

For installing Trinity-RFT, you have three options: from source (recommended for experienced users), using Docker (recommended for beginners) or via PyPI.

**Before you begin**, check your system setup:

### If you have GPUs and want to use them

Make sure your system meets these requirements:

- **Python**: 3.10 – 3.12
- **CUDA**: 12.8 or higher
- **GPUs**: At least 2 available

### If you don’t have GPUs or prefer not to use them

You can use the `tinker` option instead, which only requires:

- **Python**: 3.11 – 3.12
- **GPUs**: Not required

---

## From Source (Recommended for experienced users)

This method is best if you plan to customize or contribute to Trinity-RFT.

### 1. Clone the Repository

```bash
git clone https://github.com/agentscope-ai/Trinity-RFT
cd Trinity-RFT
```

### 2. Set Up a Virtual Environment

Choose one of the following options:

#### Using Conda

```bash
conda create -n trinity python=3.12
conda activate trinity

pip install -e ".[vllm,flash_attn]"

# If you have no GPU, comment out the line above and uncomment this instead:
# pip install -e ".[tinker]"

# If you encounter issues when installing flash-attn, try:
# pip install flash-attn==2.8.1 --no-build-isolation

pip install -e ".[dev]"  # for development like linting and debugging
```

#### Using venv

```bash
python3.10 -m venv .venv
source .venv/bin/activate

pip install -e ".[vllm,flash_attn]"

# If you have no GPU, comment out the line above and uncomment this instead:
# pip install -e ".[tinker]"

# If you encounter issues when installing flash-attn, try:
# pip install flash-attn==2.8.1 --no-build-isolation

pip install -e ".[dev]"  # for development like linting and debugging
```

#### Using `uv`

[`uv`](https://github.com/astral-sh/uv) is a modern Python package installer.

```bash
uv sync --extra vllm --extra dev --extra flash_attn

# If you have no GPU, try to use Tinker instead:
# uv sync --extra tinker --extra dev
```

---

## Via PyPI

If you just want to use the package without modifying the code:

```bash
pip install trinity-rft
pip install flash-attn==2.8.1
```

Or with `uv`:

```bash
uv pip install trinity-rft
uv pip install flash-attn==2.8.1
```

---

## Using Docker

You can download the Trinity-RFT Docker image from Github Container Registry or build it locally.

### Pull from GitHub Container Registry (Recommended for beginners)

```bash
git clone https://github.com/agentscope-ai/Trinity-RFT
cd Trinity-RFT

docker pull ghcr.io/agentscope-ai/trinity-rft:latest

docker run -it \
  --gpus all \
  --shm-size="64g" \
  --rm \
  -v $PWD:/workspace \
  -v <path_to_your_data_and_checkpoints>:/data \
  ghcr.io/agentscope-ai/trinity-rft:latest
```

```{note}
This docker image use `uv` to manage python packages, you need to activate the virtual environment using `source /opt/venv/bin/activate` after entering the docker container.
The image has include dependencies such as vllm, flash-attn and Megatron-LM, if you need to download more packages, don't forget to activate the virtual environment and use `uv pip install` to install them.
```

### Build Locally

```bash
git clone https://github.com/agentscope-ai/Trinity-RFT
cd Trinity-RFT

# Build the Docker image
## Tip: You can modify the Dockerfile to add mirrors or set API keys
docker build -f scripts/docker/Dockerfile -t trinity-rft:latest .

# Run the container, replacing <path_to_your_data_and_checkpoints> with your actual path
docker run -it \
  --gpus all \
  --shm-size="64g" \
  --rm \
  -v $PWD:/workspace \
  -v <path_to_your_data_and_checkpoints>:/data \
  trinity-rft:latest
```

---

```{note}
For training with **Megatron-LM**, please refer to {ref}`Megatron-LM Backend <Megatron-LM>`.
```

## Troubleshooting

If you encounter installation issues, refer to the FAQ or [GitHub Issues](https://github.com/agentscope-ai/Trinity-RFT/issues).
