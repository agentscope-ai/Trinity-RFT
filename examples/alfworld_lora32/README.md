# ALFWorld LoRA32，一条命令开始训练

这个示例用 Qwen3-1.7B 在 ALFWorld 文本环境中运行 multi-step GRPO，Trinity 组织任务，TuFT 提供 LoRA 训练和采样。默认 **rank 32、72 runners、150 总步**，学习率 `5e-5`。

## 准备

需要一台独占的 **Linux x86_64、8×A100 80GB** 服务器，以及支持所安装 PyTorch CUDA 版本的 NVIDIA 驱动。安装系统命令 `python3`、`git`、`uv`、`redis-server`、`wget`，确保 `nvidia-smi` 可用。首次运行需要访问 GitHub、Python 包源及 Hugging Face，并留足依赖、模型、游戏数据和 checkpoint 的磁盘空间。脚本会安装 Python 3.12 的独立环境，不需要手动准备 GPU Python 包。

**不需要再次填写路径或准备 `.env`。** 脚本自动复用第一章根目录 `.env` 中的 `TRINITY_MODEL_PATH` 和 `ALFWORLD_DATA`；未设置时使用仓库下的 `models/Qwen3-1.7B` 和 `alfworld_data`，缺失时自动下载到这两个目录。在教程仓库根目录执行：

```bash
bash examples/alfworld_lora32/run.sh
```

脚本自动安装固定的 [Trinity](https://github.com/agentscope-ai/Trinity-RFT/commit/65139711219da4338c954e546c4b8434e4f75ec2) 与 [TuFT](https://github.com/agentscope-ai/TuFT/commit/d4591c1bf707d14c3aaf238c63e35a1f3e88e442)、准备数据和模型、生成服务配置，启动专用 Redis、TuFT 和两个独立 Ray 实例，等待服务就绪后开始训练。Ray 分配 4 张卡用于采样、4 张卡用于 FSDP，具体物理卡号由调度决定。环境安装采用 `requirements-lora32.txt` 中的兼容性 overrides；不会修改第一章的 `.venv`。

当前固定版本来自两个上游仓库的 main。客户端与服务端均使用 Tinker SDK 0.25.0；服务端微批为 1。当前组合已在 **8×A100 80GB 上通过两步训练验证**，覆盖实际采样、两次参数更新，以及训练状态和采样权重的保存；保存的优化器步数为 2，LoRA 权重已更新且数值有效。参考曲线和完整训练耗时仍属于历史实验，本轮未验证完整 150 步训练或中断恢复。已用旧版建立过环境的读者，请设置新的 `LORA32_WORK_DIR`；保留旧目录用于原版本的恢复。

首次使用也可以先把训练目标设为两步，检查采样、参数更新和 checkpoint 保存这条链路：

```bash
bash examples/alfworld_lora32/run.sh --steps 2
```

这仍使用默认模型、72 个 runner 和训练批大小，需要同样的 GPU 资源；两步运行不用于判断最终训练效果。

## 输出与后续运行

模型和数据使用上述共享目录；独立运行环境、服务状态和训练输出保存在 `$HOME/.cache/agentic-rl/alfworld-lora32`：

- `logs/`：Redis、Ray 和 TuFT 的服务日志。
- `checkpoints/ALFWORLD/Step_Wise_Alfworld_TuFT_lora32_lr5e5_speed_<实验名>/`：客户端 checkpoint 指针、训练记录，以及每次运行的 `launch_configs/`。
- `server-checkpoints/`、`redis/`：服务端模型状态与持久化注册信息。恢复时需要一起保留。
- `client-venv/`、`TuFT/.venv/`：独立客户端和服务端环境。

API key 自动生成并仅保存在专用目录，不写入教程仓库，也不在终端显示。默认端口为 `16380–16400` 中的专用端口；已有未知进程占用端口时会停止并报错。需要更换端口时，为新实验选择一个新工作目录和新的 `LORA32_PORT_BASE`。

准备阶段下载或安装失败后，可以重复同一条命令继续。训练输出已经创建后，脚本不会自动覆盖：中断后继续同一个实验，显式加 `--resume`：

```bash
bash examples/alfworld_lora32/run.sh --resume
```

`--steps` 是总目标，必须大于已保存的 checkpoint 步号。例如完成 150 步后继续到 250 步：

```bash
bash examples/alfworld_lora32/run.sh --resume --steps 250
```

从零训练 250 步则使用新实验名和 `--steps 250`。服务在训练完成或中断后保持运行，以便恢复；脚本只复用自己启动且配置、版本和进程记录一致的服务，不停止其他进程、不清空 Redis。遇到服务启动失败，先查看 `logs/` 中对应日志；脚本不会自动删除状态或重启活跃服务。

可在服务器上先运行只读检查：

```bash
bash examples/alfworld_lora32/run.sh --check
bash examples/alfworld_lora32/run.sh --help
```

`--check` 检查系统命令、硬件清单、本地路径和恢复元数据，不安装依赖、下载模型或启动任何服务，也不验证 GPU 训练可运行。

## 可选配置

需要自定义时，复制 `.env.example` 为本目录的 `.env`。模型和数据路径依次取显式非空环境变量、LoRA `.env`、仓库根 `.env`，最后取上述默认路径；相对路径均按仓库根目录解释。根 `.env` 中的其他实验与服务设置不会导入。

`LORA32_WORK_DIR` 控制独立工作目录，必须位于教程仓库外；`TRINITY_MACHINE_ID` 控制本次实验名称。新实验使用新名称和新工作目录，恢复训练时保持不变。
