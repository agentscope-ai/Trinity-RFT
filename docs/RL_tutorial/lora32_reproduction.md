# LoRA32 运行参考

这里集中说明环境要求、默认配置、日志和中断恢复，供运行时查阅。第一次实践请从[第六章](ch6_动手实验.md)开始：看图理解 LoRA，启动训练，再对照参考结果。

## 环境与共享路径

示例按一台 **8×A100 80GB 的 Linux 服务器**配置，4 卡用于采样、4 卡用于训练。机器上需要有 Git、Python 3、`uv`、`redis-server` 和 `wget`，并能正常运行 `nvidia-smi`。脚本会准备 Python 3.12 的独立环境。

以下命令都在教程仓库根目录执行。**不需要再填写路径或复制 `.env`。** 脚本自动复用第一章根目录 `.env` 中的模型和数据路径；没有设置时使用相同的默认目录：

| 内容 | 默认路径（相对教程仓库根目录） |
|---|---|
| 模型 | `./models/Qwen3-1.7B` |
| ALFWorld 数据 | `./alfworld_data` |

目录中已有模型和数据时直接复用，缺失时自动下载到这两个目录。LoRA 的运行环境和训练输出单独保存。

## 启动与默认配置

| 用途 | 命令 |
| --- | --- |
| 检查运行条件 | `bash examples/alfworld_lora32/run.sh --check` |
| 先检查两步采样、训练与存档链路 | `bash examples/alfworld_lora32/run.sh --steps 2` |
| 安装配套环境、准备数据并启动训练 | `bash examples/alfworld_lora32/run.sh` |

启动前释放本例需要的 GPU。脚本会为 Trinity 客户端与 TuFT 服务端分别准备环境，使用以下默认配置：

| 参数 | 默认值 |
|---|---|
| 模型 / 环境 | Qwen3-1.7B / ALFWorld |
| LoRA | rank 32，七类投影模块，约 3490 万可训练参数 |
| 算法 | 多步 GRPO，每题采样 16 条轨迹 |
| 学习率 / KL 系数 | `5e-5` / `0.001` |
| 单条轨迹 | 最多 30 个环境步，每次生成最多 512 tokens |
| 并发 / 训练步数 | 72 个 runner / 150 步 |
| 保存频率 | 每 10 步保存 checkpoint |

**runner 数**决定同时运行多少个任务，**每题采样数**决定同一任务收集多少条轨迹，两者不同。

历史加速版在 72 个 runner 下外推 150 步约 **35 小时**，仅供参考。当前上游配置改用按样本数限制的微批，尚未测量完整训练耗时；请先短程运行，再按自己的日志安排预算。估算依据见[第六章](ch6_动手实验.md)的时间预算折叠区。

## 日志与存档

运行环境和训练输出默认保存在 `$HOME/.cache/agentic-rl/alfworld-lora32`，模型和数据使用前面的共享目录。启动时会显示日志与训练输出目录；默认配置下，在另一个终端执行：

```bash
LORA32_RUN_DIR="$HOME/.cache/agentic-rl/alfworld-lora32/checkpoints/ALFWORLD/Step_Wise_Alfworld_TuFT_lora32_lr5e5_speed_reader72_001"
tail -f "$LORA32_RUN_DIR/log/trainer.log"
```

通过日志和 TensorBoard 观察 `critic/score/mean`：它是按对话步加权的训练 reward，应该结合一段时间内的趋势来看，不能直接换算成测试成功率。

默认训练完成后，日志应到达第 150 步，并保存 `global_step_150/`。模型权重在 TuFT 的 checkpoint 目录中，Trinity 输出目录保存训练记录和权重引用；请保留整个示例工作目录，以便继续训练。

## 中断后继续训练

沿用同一份配置，确认本次训练的客户端已经停止，再执行：

```bash
bash examples/alfworld_lora32/run.sh --resume
```

如已完成 150 步，想继续到 250 步：

```bash
bash examples/alfworld_lora32/run.sh --resume --steps 250
```

`--steps` 是训练的**总目标步数**，需要大于已保存的步数。开始一项新实验时，使用新的实验名和工作目录。

## 配套版本

安装脚本固定以下**上游 main 已合并提交**，并分别准备客户端与服务端的 Python 环境，避免后续 main 更新改变依赖：

| 组件 | 版本 |
|---|---|
| Trinity | [`6513971`](https://github.com/agentscope-ai/Trinity-RFT/commit/65139711219da4338c954e546c4b8434e4f75ec2)，与前五章使用同一份源码 |
| TuFT | [`d4591c1`](https://github.com/agentscope-ai/TuFT/commit/d4591c1bf707d14c3aaf238c63e35a1f3e88e442) |

客户端与服务端使用 Tinker SDK `0.25.0`。TuFT 使用 `fsdp_target_modules` 指定七类投影模块，训练微批为 `micro_batch_size: 1`，由 Ray 分配 4 张采样卡和 4 张训练卡。

2026-09-30，当前上游组合在 **8×A100 80GB** 上完成两步训练验证，沿用默认 72 个 runner 和训练批大小，复用已有模型与 ALFWorld 数据。实际采样、两次参数更新、训练状态和采样权重保存均通过；两步训练指标均为有限值，保存的优化器步数为 2，LoRA 权重已更新，训练与采样 adapter 一致。

参考曲线仍来自历史 Trinity `b54370a` / TuFT `ec71aec`。本轮是两步链路验证，未重跑完整 150 步训练、复现历史耗时或测试中断恢复。

如果此前运行过旧版，请为新的上游组合设置新的 `LORA32_WORK_DIR`，不要覆盖旧环境或直接跨版本恢复。当前版本的恢复需要同时保留服务端 checkpoint、Redis 注册信息和空闲 LoRA 槽位。

完整配置与脚本见 [LoRA32 示例目录](https://github.com/agentscope-ai/Trinity-RFT/tree/tutorial/examples/alfworld_lora32/)。

## 可选：自定义配置

只有需要更换路径、工作目录或实验名时，才需要复制 `examples/alfworld_lora32/.env.example` 为同目录下的 `.env` 并修改相应设置。模型和数据路径优先使用显式环境变量，其次是 LoRA 专用 `.env`、仓库根目录 `.env`，最后使用默认值；相对路径按教程仓库根目录解释。根 `.env` 只复用模型和数据路径，LoRA 使用自己的实验名和服务配置。
