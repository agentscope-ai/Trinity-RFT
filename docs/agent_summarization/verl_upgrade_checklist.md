# veRL 升级前检查清单

本清单用于 Trinity 下次升级 veRL 版本前的快速检查。

## 1. 升级范围确认

1. 确认目标 veRL 版本。
2. 确认 Trinity 当前基线版本。
3. 确认需要对照的上游快照已经生成到 `trinity/trainer/verl/build/<version>/`。
4. 确认本次升级仍只覆盖以下 7 个核心迁移文件：
   - `fsdp_workers.py`
   - `dp_actor.py`
   - `fsdp_checkpoint_manager.py`
   - `megatron_workers.py`
   - `megatron_actor.py`
   - `megatron_checkpoint_manager.py`
   - `verl_trainer.py`（对应上游 `ray_trainer.py`）

## 2. 三路比较准备

1. 对每个文件同时比较三组内容：
   - 当前 Trinity 文件
   - `build/<old_version>/...`
   - `build/<new_version>/...`
2. 不允许直接整文件覆盖。
3. 优先记录两类差异：
   - Trinity 在旧版本基线上加了什么
   - 上游从旧版本到新版本改了什么

## 3. 仓库职责边界确认

下次升级前必须先确认以下边界仍然成立：

1. reward 计算不在 Trinity 的 `verl_trainer.py` 中执行。
2. rollout 不在 Trinity 的 veRL trainer 主循环中执行。
3. trainer 侧 validation 当前未实现。
4. Trinity 不直接走上游 `RayPPOTrainer.fit()` 主循环，而是走自己在 `trinity/trainer/trainer.py` 中定义的 `prepare()`、`train_step()`、`save_checkpoint()`、`save_state_dict()`、`upload_state_dict()` 路径。

如果以上任一边界发生变化，则本清单后续步骤需要重新评估。

## 4. 不应误迁回来的上游逻辑

除非 Trinity 训练职责发生变化，否则下列上游能力不应默认迁回：

1. `fit()` 中完整 reward 主流程。
2. validation 主流程。
3. reward loop / async rollout manager。
4. 仅供上游 trainer 主循环使用的 `CheckpointEngineManager` 编排逻辑。

## 5. 必查配置链路

升级前先确认以下配置项是否仍需要从 config 贯穿到实现：

1. `trust_remote_code`
2. `use_prefix_grouper`
3. `calculate_sum_pi_squared`
4. `sum_pi_squared_checkpointing`
5. `lora.rank` 与 `lora_rank` 兼容读取
6. `rollout_correction`
7. `reward.reward_model` 与 `reward_model` 的兼容结构

## 6. 文件级优先顺序

建议按以下顺序处理：

1. `dp_actor.py`
2. `fsdp_workers.py`
3. `megatron_actor.py`
4. `megatron_workers.py`
5. `fsdp_checkpoint_manager.py`
6. `megatron_checkpoint_manager.py`
7. `verl_trainer.py`
8. `verl_config.py`

原因：前四个文件决定数据字段和配置链路，后面三个文件依赖这些契约是否稳定。

## 7. 每个文件都要做的收敛检查

对每个迁移文件都要问一遍：

1. 这个子类实现是否只是复制父类代码？
2. 如果已经和上游父类完全一致，能否直接删除重写？
3. 如果只是历史 workaround，是否已经被上游吸收？
4. 如果是 Trinity 真正新增职责，是否已在文档中说明保留原因？

## 8. 当前已确认不能误删的 Trinity 定制

1. `dp_actor.py` 与 `megatron_actor.py` 中的算法接入和 loss 组合逻辑。
2. `fsdp_checkpoint_manager.py` 与 `megatron_checkpoint_manager.py` 中的 `CheckpointMonitor` / `Synchronizer` 协作逻辑。
3. `verl_trainer.py` 中的 `CheckpointMonitor`、Trinity 自定义 `train_step()`、状态同步路径。
4. Trinity 独立的 experience pipeline 与 trainer 调度关系。

## 9. 当前已知的迁移敏感点

1. `use_prefix_grouper` 是从 config 到 monkey patch 到 actor / ref worker 的整链路能力。
2. `sum_pi_squared` 必须从 actor 输出一直传到优势函数消费端。
3. Megatron LoRA reference logprob 走 actor/no-adapter 路径，不是普通 ref worker 路径。
4. 多模态训练如果要统计 MFU，需要在 trainer 中补 `images_seqlens` 到 `batch.meta_info`。
5. checkpoint manager 不能用上游整文件覆盖，否则会丢 Trinity 的异步线程与监控逻辑。

## 10. 升级完成后的本地检查

1. 对所有迁移文件执行 Problems 检查。
2. 对所有迁移文件执行统一 `python -m py_compile`。
3. 检查新增配置项是否在 dataclass、默认值、读取路径上闭环。
4. 检查 actor 输出字段与 worker / trainer 消费字段是否一致。
5. 检查 checkpoint 保存与恢复的函数签名是否自洽。

## 11. 远端 GPU 最小回归

本地检查通过后，远端至少覆盖以下回归项：

1. FSDP 单步训练。
2. Megatron 单步训练。
3. old logprob / ref logprob 重算。
4. LoRA 下 Megatron reference logprob。
5. checkpoint 保存与恢复。
6. `use_prefix_grouper` 最小回归。
7. `calculate_sum_pi_squared` 最小回归。

## 12. 最后确认

在提交本次升级前，再确认一遍：

1. 没有把 reward、rollout、validation 逻辑误搬回 Trinity trainer。
2. 没有保留已经与上游完全一致的重复子类实现。
3. 没有因为版本对齐而恢复 Trinity 之前主动裁掉的功能。
4. 文档中已经补充本次升级新增的仓库约束与保留理由。
