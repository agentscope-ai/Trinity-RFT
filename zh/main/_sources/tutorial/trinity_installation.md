(Installation)=
# 安装指南

安装 Trinity-RFT 有三种方式：源码安装（推荐有经验的用户使用该方法）、使用 Docker (推荐初学者使用该方法) 或是从 PyPI 安装。

**开始之前**，请检查您的系统配置：

### 如果您有 GPU 并希望使用它们

请确保您的系统满足以下要求：

- **Python**：3.10 – 3.12
- **CUDA**：12.8 或更高版本
- **GPU**：至少一块 [compute capability](https://developer.nvidia.com/cuda/gpus) 为 8.0 或更高的 NVIDIA GPU（例如 RTX 30 系列、A100、H100）

### 如果您没有 GPU 或不希望使用 GPU

您可以改用 `tinker` 选项，该选项仅需满足：

- **Python**：3.11 – 3.12
- **GPU**：无需

---

## 源码安装（推荐有经验的用户使用该方法）

如需修改、扩展 Trinity-RFT，推荐使用此方法。

### 1. 克隆仓库

```bash
git clone https://github.com/agentscope-ai/Trinity-RFT
cd Trinity-RFT
```

### 2. 创建虚拟环境

可选择以下任一方式：

#### 使用 Conda

```bash
conda create -n trinity python=3.12
conda activate trinity

pip install -e ".[vllm,flash_attn]"

# 如果没有GPU，可以注释上一行的命令，改为使用Tinker：
# pip install -e ".[tinker]"

# 如果安装 flash-attn 时遇到问题，可尝试：
# pip install flash-attn==2.8.1 --no-build-isolation

pip install -e ".[dev]"  # 用于调试和开发
```

#### 使用 venv

```bash
python3.10 -m venv .venv
source .venv/bin/activate

pip install -e ".[vllm,flash_attn]"

# 如果没有GPU，可以注释上一行的命令，改为使用Tinker：
# pip install -e ".[tinker]"

# 如果安装 flash-attn 时遇到问题，可尝试：
# pip install flash-attn==2.8.1 --no-build-isolation

pip install -e ".[dev]"  # 用于调试和开发
```

#### 使用 `uv`

[`uv`](https://github.com/astral-sh/uv) 是现代的 Python 包管理工具。

```bash
uv sync --extra vllm --extra dev --extra flash_attn

# 如果没有GPU，可以改为使用Tinker：
# uv sync --extra tinker --extra dev
```

---

## 使用 Docker

您可以从 Github 拉取 Docker 镜像或是自行构建镜像。

### 从 Github 拉取预构建镜像 (推荐初学者使用该方法)

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
该 Docker 镜像使用 `uv` 来管理 Python 依赖，进入容器后虚拟环境会自动激活（也可通过 `source /opt/venv/bin/activate` 手动激活）。
该镜像已经包含了 vllm, flash-attn 以及 Megatron-LM，如果需要使用其他依赖，可直接使用 `uv pip install` 来安装它们。
```

### 自行构建 Docker 镜像


```bash
git clone https://github.com/agentscope-ai/Trinity-RFT
cd Trinity-RFT

# 构建 Docker 镜像
## 提示：可根据需要修改 Dockerfile 添加镜像源或设置 API 密钥
docker build -f scripts/docker/Dockerfile -t trinity-rft:latest .

# 运行容器，请将 <path_to_your_data_and_checkpoints> 替换为实际需要挂载的路径
docker run -it \
  --gpus all \
  --shm-size="64g" \
  --rm \
  -v $PWD:/workspace \
  -v <path_to_your_data_and_checkpoints>:/data \
  trinity-rft:latest
```

---

## 通过 PyPI 安装

如果您只需使用 Trinity-RFT 而不打算修改代码：

```bash
pip install trinity-rft
pip install flash-attn==2.8.1 --no-build-isolation
```

或使用 `uv`：

```bash
uv pip install trinity-rft
uv pip install flash-attn==2.8.1 --no-build-isolation
```

---

```{note}
如需使用 **Megatron-LM** 进行训练，请参考 {ref}`Megatron-LM Backend <Megatron-LM>`。
```


## 常见问题

如遇安装问题，请参考 FAQ 或 [GitHub Issues](https://github.com/agentscope-ai/Trinity-RFT/issues)。
