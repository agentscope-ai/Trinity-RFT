# 常见问题（FAQ）

## 第一部分：参数配置

**Q:** 如何编写 Trinity-RFT 的配置文件？

**A:** 推荐使用 Trinity Studio 进行配置文件编写。你可以通过以下命令启动：

```bash
trinity studio --port 8080
```

该指令提供一个直观易用的图形化界面来创建和修改配置文件。

对于进阶用户，建议参考[配置文档](./trinity_configs.md)以了解所有配置项的详细说明。你也可以直接编辑 YAML 配置文件（可参考 `examples` 目录下的模板和示例）。

如果你已经熟悉 veRL，请参考[与 veRL 对齐配置](./align_with_verl.md)。该教程介绍了如何将 Trinity-RFT 的配置参数与 veRL 对齐，方便迁移或复用已有的 veRL 配置。

---

**Q:** `buffer.batch_size`、`buffer.train_batch_size`、`actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` 及其他 batch size 参数之间的关系？

**A:** 这些参数的关系如下：

- `buffer.batch_size`：一个 mini-batch 中的任务（task）数量，对 explorer 有效。
- `buffer.train_batch_size`：一个 mini-batch 中的 experience 数量，对 trainer 有效。如果未显式指定，则默认为 `buffer.batch_size` * `algorithm.repeat_times`。
- `actor_rollout_ref.actor.ppo_mini_batch_size`：一个 mini-batch 中的 experience 数量，会被 `buffer.train_batch_size` 覆盖；但在 `update_policy` 函数中，其值表示每个 GPU 上的 mini-batch experience 数量，即 `buffer.train_batch_size (/ ngpus_trainer)`。除以 `ngpus_trainer` 是由于数据隐式分配到多个 GPU 上所致，但在梯度累积后不影响最终结果。
- `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu`：每个 GPU 上 micro-batch 中的 experience 数量。

一个简要示例：

```python
def update_policy(batch_exps):
    dataloader = batch_exps.split(ppo_mini_batch_size)
    for _ in range(ppo_epochs):
        for batch_idx, data in enumerate(dataloader):
            # 分割数据
            mini_batch = data
            if actor_rollout_ref.actor.use_dynamic_bsz:
                micro_batches, _ = rearrange_micro_batches(
                        batch=mini_batch, max_token_len=max_token_len
                    )
            else:
                micro_batches = mini_batch.split(actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu)

            # 计算梯度
            for data in micro_batches:
                entropy, log_prob = self._forward_micro_batch(
                    micro_batch=data, ...
                )
                pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                    log_prob=log_prob, **data
                )
                policy_loss = pg_loss + ...
                loss = policy_loss / self.gradient_accumulation
                loss.backward()

            # 优化器更新
            grad_norm = self._optimizer_step()
    self.actor_optimizer.zero_grad()
```
详细实现请参考 `trinity/trainer/verl/dp_actor.py`。veRL 也在其 [FAQ](https://verl.readthedocs.io/en/latest/faq/faq.html#what-is-the-meaning-of-train-batch-size-mini-batch-size-and-micro-batch-size) 中对此进行了说明。

## 第二部分：常见报错

**报错：**
```bash
File ".../flash_attn/flash_attn_interface.py", line 15, in ‹module>
    import flash_attn_2_cuda as flash_attn_gpu
ImportError: ...
```

**A:** `flash-attn` 模块未正确安装。请尝试运行 `pip install flash-attn==2.8.1` 或 `pip install flash-attn==2.8.1 -v --no-build-isolation` 进行修复。

---

**报错：**
```bash
UsageError: api_key not configured (no-tty). call wandb.login(key=[your_api_key]) ...
```

**A:** 请在启动 Ray 和运行实验前先登录 WandB。可以通过 `export WANDB_API_KEY=[your_api_key]` 设置环境变量。你也可以通过设置 `monitor.monitor_type=tensorboard/mlflow` 使用其他监控方式。

---

**报错：**
```bash
ValueError: Failed to look up actor with name 'explorer' ...
```

**A:** 请确保在运行实验前已启动 Ray。如果 Ray 已在运行，可通过以下命令重启：

```bash
ray stop
ray start --head
```

---

**报错：** 内存不足（OOM）错误

**A:** 可尝试调整以下参数：

- 对于 trainer，当 `trainer.use_dynamic_bsz=false` 时，调整 `trainer.max_token_len_per_gpu`；当 `trainer.use_dynamic_bsz=true` 时，调整 `trainer.ppo_max_token_len_per_gpu` 和 `trainer.ulysses_sequence_parallel_size`。设置 `trainer.trainer_config.actor_rollout_ref.actor.entropy_from_logits_with_chunking=true` 也可能有帮助。
- 对于 explorer，调整 `explorer.rollout_model.tensor_parallel_size`。

此外，Trinity-RFT 提供了[GPU 相关配置指南](https://agentscope-ai.github.io/Trinity-RFT/en/main/tutorial/trinity_gpu_configs.html)，可参考其中建议。

## 第三部分：调试方法

**Q:** Trinity-RFT 运行日志在哪查看？

**A:** Trinity-RFT 支持 actor 级别日志，会自动将每个 actor（如 explorer 和 trainer）的日志保存到 `<checkpoint_job_dir>/log/<actor_name>`。

常见日志包括：

- `<checkpoint_job_dir>/log/explorer.log`：explorer 进程日志，包括每步起止时间、评测指标、模型权重同步、异常等。
- `<checkpoint_job_dir>/log/trainer.log`：trainer 进程日志，包括每次训练迭代的起止时间、评测指标、模型权重同步等。
- `<checkpoint_job_dir>/log/explorer_runner_<n>.log`：workflow runner 进程日志，包括 workflow 打印的日志和运行异常（需在 Workflow 中用 `self.logger` 打印日志）。

如需更详细日志，可在配置文件中设置 `log.level=debug`。

如需查看所有进程的完整日志并保存到 `debug.log`：

```bash
export RAY_DEDUP_LOGS=0
trinity run --config grpo_gsm8k/gsm8k.yaml 2>&1 | tee debug.log
```

---

**Q:** 如何在不运行完整实验的情况下调试 workflow？

**A:** 可用 Trinity-RFT 的 debug 模式，步骤如下：

1. 启动推理模型：
   ```bash
   trinity debug --config <config_file_path> --module inference_model
   ```
2. 在另一个终端调试 workflow：
   ```bash
   trinity debug --config <config_file_path> --module workflow --output-file <output_file_path> --plugin-dir <plugin_dir>
   ```

详细说明见 {ref}`工作流开发指南 <Workflows>`。

## 第四部分：其他问题

**Q:** `buffer.trainer_input.experience_buffer.path` 有什么作用？

**A:** 该路径指定用于存储生成 experience 的 SQLite 数据库路径。如果不需要，可注释掉该行。

如需查看数据库中的 experience，可用如下 Python 脚本：

```python
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
from trinity.common.schema.sql_schema import ExperienceModel

engine = create_engine(buffer.trainer_input.experience_buffer.path)
session = sessionmaker(bind=engine)
sess = session()

MAX_EXPERIENCES = 4
experiences = (
    sess.query(ExperienceModel)
    .limit(MAX_EXPERIENCES)
    .all()
)

exp_list = []
for exp in experiences:
    exp_list.append(ExperienceModel.to_experience(exp))

# 打印 experience
for exp in exp_list:
    print(f"{exp.prompt_text=}", f"{exp.response_text=}")
```

---

**Q:** 如何在 Trinity-RFT 框架外加载 checkpoints？

**A:** 目前支持两种加载方式：

1. **推荐方式**：使用 `trinity convert` 命令将原始检查点转换为标准的 Hugging Face 格式。
   转换后，你就可以像加载普通 Hugging Face 模型一样直接使用它。
   详细操作请参考教程：[可选：将检查点转换为 Hugging Face 格式](https://agentscope-ai.github.io/Trinity-RFT/zh/main/tutorial/example_reasoning_basic.html#hugging-face)

2. **直接加载（适用于 FSDP 训练的 actor 检查点）**：
   如果你希望不转换格式而直接加载，可以使用以下代码示例：

```python
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from trinity.common.models.utils import load_fsdp_state_dict_from_verl_checkpoint

# 假设我们需要第 780 步的检查点；
# model_path、checkpoint_root_dir、project 和 name 已定义
model = AutoModelForCausalLM.from_pretrained(model_path)
ckp_path = os.path.join(checkpoint_root_dir, project, name, "global_step_780", "actor")
model.load_state_dict(load_fsdp_state_dict_from_verl_checkpoint(ckp_path))
```

---

**Q:** Trinity-RFT 和 veRL 有什么区别？

**A:** Trinity-RFT 以 veRL 作为 trainer 后端，并在其基础上扩展了更模块化和灵活的架构。主要区别包括：

- **模块化算法**：Trinity-RFT 将 veRL `core_algos.py` 中的算法相关组件（如优势函数、损失函数）提取为独立模块，便于用户实现和注册新算法，无需修改核心代码。
- **Explorer 与 Trainer 分离**：Trinity-RFT 用独立 Explorer 模块替代 veRL 的 rollout model，专门负责 agent 与环境交互，支持更灵活的 workflow 设计和 rollout-training 调度。
- **全生命周期数据通路**：Trinity-RFT 在 Explorer 和 Trainer 之间增加 Buffer 模块，提供完整的数据存储、处理和采样通路，支持经验回放、优先采样等高级数据处理策略。

我们还提供了 Trinity-RFT 与 veRL 及其衍生系统（如 [rLLM](https://github.com/rllm-org/rllm)）的基准对比，详见 [Benchmark](https://github.com/agentscope-ai/Trinity-RFT/tree/main/benchmark)。
