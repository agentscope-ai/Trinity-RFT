# verl 版本迁移计划（Trinity v0.7.0 -> v0.7.1）

## 1. 目标与输入

本次工作不是直接把 Trinity 目录下的文件覆盖成上游新版本，而是做一次三路合并：

- Trinity 当前实现：`trinity/trainer/verl/*.py`
- 已格式化的上游 veRL v0.7.0：`trinity/trainer/verl/build/v0.7.0/*.py`
- 已格式化的上游 veRL v0.7.1：`trinity/trainer/verl/build/v0.7.1/*.py`

目标是把当前 Trinity 中基于 v0.7.0 的定制迁移到 v0.7.1，同时保留已经验证过的 Trinity 行为，删除已经被上游吸收或不再需要的本地补丁。

本次涉及的文件如下：

1. `fsdp_workers.py`
2. `dp_actor.py`
3. `fsdp_checkpoint_manager.py`
4. `megatron_workers.py`
5. `megatron_actor.py`
6. `megatron_checkpoint_manager.py`
7. `verl_trainer.py`（对应上游 `ray_trainer.py`）

相关脚本：`scripts/migrate_from_verl/init_migration.py`

## 2. 迁移原则

1. 先比较，再迁移。
   所有修改都应同时参考两组差异：
   - 当前 Trinity 相对 v0.7.0 做了什么
   - 上游 v0.7.0 到 v0.7.1 改了什么

2. 只迁移仍然需要保留的 Trinity 定制。
   如果某个本地补丁在 v0.7.1 中已经被上游修复或替代，应优先删除本地实现，改为复用上游逻辑。

3. 不恢复之前主动裁剪掉的功能。
   例如 Trinity 曾出于精简目的移除了部分 rollout 相关能力，这类功能不需要因为升级而重新加回来。

4. 能 import 上游实现的，不重复复制。
   如果某段逻辑已存在于 v0.7.1 对应模块中，优先通过 import 或继承复用，而不是继续在 Trinity 中复制一份。

5. 先处理低风险文件，再处理高风险文件。
   先完成函数级可对齐的 Actor 和 Worker 迁移，再处理 checkpoint manager 和 trainer 这类已经重构过的文件。

6. 配置与实现同步审查。
   所有新增能力都要回看 `trinity/trainer/verl/verl_config.py`，避免实现已经接入、配置却缺失默认值或兼容入口。

## 3. 三路比较方法

每个文件都按以下顺序处理：

1. 以 `build/v0.7.0` 为共同基线。
2. 提取当前 Trinity 相对 `build/v0.7.0` 的定制点。
3. 提取 `build/v0.7.1` 相对 `build/v0.7.0` 的上游变更。
4. 判断每个变更属于以下哪一类：
   - 可直接按函数对齐迁移
   - 需要在 Trinity 定制框架中手工合并
   - 已被上游吸收，应删除 Trinity 补丁
   - 已被 Trinity 明确裁剪，不应恢复
5. 最后再看当前 Trinity 相对 `build/v0.7.1` 的剩余差距，确认是否已经部分吸收了新版本改动，避免重复工作。

## 4. 当前差异概览

从 diff 规模看，可按两组处理：

- 中等风险但偏函数级：`dp_actor.py`、`fsdp_workers.py`、`megatron_actor.py`、`megatron_workers.py`
- 高风险且需要手工三路合并：`fsdp_checkpoint_manager.py`、`megatron_checkpoint_manager.py`、`verl_trainer.py`

当前代码中已经能确认存在一部分与 v0.7.1 接近的适配，后续迁移时要先识别并保留这些已有成果，避免重复回填：

- `fsdp_workers.py` 和 `megatron_workers.py` 已包含 `trust_remote_code` 相关接线
- `fsdp_checkpoint_manager.py` 已有 `trust_remote_code` 参数透传
- `megatron_checkpoint_manager.py` 已包含 `mcore_ge_014`、metadata 构建和 content metadata 相关逻辑

因此，本次升级不是从零开始补 v0.7.1 特性，而是要精确识别“哪些已经手工 backport 过，哪些仍然缺失，哪些应该回退到上游实现”。

## 5. 按文件迁移策略

### 5.1 `dp_actor.py`

特点：Trinity 对上游 Actor 的改动主要集中在算法接入，整体仍可按函数迁移。

已知 Trinity 定制重点：

- `DataParallelPPOActor` 改为继承上游 `DPActor`
- 新增 `set_algorithm()`，把 policy loss、KL loss、entropy loss 接入 Trinity 算法注册表
- `update_policy()` 使用 Trinity 的 loss 组合逻辑，而不是完全沿用上游 PPO loss 流程

v0.7.1 需要重点关注的上游变化：

- `use_prefix_grouper`
- `calculate_sum_pi_squared`
- `sum_pi_squared_checkpointing`
- 与动态 batch、返回结果结构相关的改动

迁移策略：

1. 保留 Trinity 的 `set_algorithm()` 和 loss 注入框架。
2. 在 `_forward_micro_batch()` 和 `update_policy()` 层面吸收 v0.7.1 的新增数据流，尤其是 `sum_pi_squared` 与 prefix grouper 路径。
3. 只迁移对 Trinity 现有算法分支确实有用的上游新增逻辑，不为了“对齐”而重新引入已被裁掉的能力。

### 5.2 `fsdp_workers.py`

特点：改动量不小，但大部分仍围绕 worker 构建流程，可以按照关键构造路径逐段比对。

v0.7.1 需要重点关注的上游变化：

- `trust_remote_code` 的更多透传位置
- `use_prefix_grouper` 从配置到 actor/ref 构建链路的接线
- `sum_pi_squared` 从 actor 输出到 worker 打包的透传

迁移策略：

1. 优先比对 actor/ref 初始化和 `_build_rollout()` 附近的参数接线。
2. 检查 v0.7.1 是否要求 worker 把新的 actor 配置传给 ref、critic 或 rollout 相关对象。
3. 对于 Trinity 已移除的 rollout 能力，不做恢复，只保留与当前训练链路相关的上游改动。

### 5.3 `megatron_actor.py`

特点：与 `dp_actor.py` 类似，主要是算法接入和 batch 前后处理差异。

迁移策略：

1. 保留 Trinity 的 `set_algorithm()` 和损失组合方式。
2. 对齐 v0.7.1 在 forward/backward 批处理、输出字段和可能的新统计量上的变化。
3. 检查是否需要同步支持与 `dp_actor.py` 对应的新输出字段，例如 `sum_pi_squared`。

### 5.4 `megatron_workers.py`

特点：风险略低于 checkpoint manager，但要小心与 `megatron_actor.py`、checkpoint manager 的联动。

迁移策略：

1. 先对齐 v0.7.1 在 tokenizer、processor、model 构建链路中的 `trust_remote_code` 和相关配置透传。
2. 再检查 ActorRolloutRefWorker 与 actor 输出契约是否需要同步更新。
3. 只补当前训练路径依赖的上游变更，不恢复 Trinity 已经移除的分支。

### 5.5 `fsdp_checkpoint_manager.py`

特点：高风险文件。Trinity 在上游基础上做了明显的拆分、线程化和 Ray 协作增强，不能做机械 patch。

Trinity 定制重点：

- 继承上游 checkpoint manager 后又增加了 Ray namespace、Synchronizer、CheckpointMonitor 等机制
- 把保存流程拆成多个步骤和线程任务
- 区分 checkpoint 与 state dict 的异步追踪和上报

v0.7.1 需要重点关注的上游变化：

- `ensure_checkpoint_capacity()`
- `trust_remote_code` 透传
- 与 HF model 保存兼容性相关的改动

迁移策略：

1. 先梳理 Trinity 当前保存流程中的职责边界：本地保存、异步线程、状态同步、监控上报。
2. 再把 v0.7.1 的新增逻辑映射到这些职责点，而不是直接替换整个 `save_checkpoint()`。
3. 对于已经能通过继承或调用上游函数复用的部分，尽量减少本地复制实现。
4. 如果上游已经吸收 Trinity 之前修的 bug，应删除对应本地补丁，避免重复分叉。

### 5.6 `megatron_checkpoint_manager.py`

特点：高风险文件。这里既有 Trinity 定制，也已经可以看到一部分偏新的 metadata 逻辑，因此需要先分清当前代码到底来源于哪个上游阶段。

当前已观察到的重点：

- 已存在 `mcore_ge_014`
- 已存在 `_build_sharded_state_dict_metadata()`
- 已存在 `content_metadata` 相关保存和加载逻辑

这说明该文件并非纯粹停留在 v0.7.0 语义上，迁移时必须先做来源核对：

1. 识别哪些逻辑已经和 v0.7.1 对齐
2. 识别哪些仍是 Trinity 本地包装或 workaround
3. 识别哪些地方只是“部分 backport”，仍与 v0.7.1 存在行为差异

迁移策略：

1. 先以状态字典生成、metadata 构建、加载流程为主线做职责对齐。
2. 再把 Trinity 的 Synchronizer、CheckpointMonitor 等分布式协作逻辑叠加到新流程中。
3. 避免继续保留已经能从上游直接继承的复制方法。

### 5.7 `verl_trainer.py` / `ray_trainer.py`

特点：最高风险之一。Trinity 的 `VerlPPOTrainerWrapper` 是从上游 `RayPPOTrainer` 继承并深度改造而来，训练循环与 checkpoint 流程都已显著分叉。

Trinity 定制重点：

- `CheckpointMonitor`
- `VerlPPOTrainerWrapper`
- 自定义 worker 初始化
- 自定义训练主循环、指标汇总和状态同步

v0.7.1 需要重点关注的上游变化：

- `CheckpointEngineManager`
- `extract_reward`
- `compute_variance_proxy_metrics`
- `GDPO`
- `OPTIMAL_TOKEN_BASELINE`
- `TIR_OPTIMAL_TOKEN_BASELINE`
- `use_prefix_grouper`
- `lora.rank` 与 `lora_rank` 的兼容读取

迁移策略：

1. 先对齐新的配置读取与算法入口，不急着先改 checkpoint 管理。
2. 把 v0.7.1 新增的训练阶段语义拆成若干最小接入点：奖励提取、advantage 分支、额外 metrics、prefix grouper 相关批处理。
3. 再评估 `CheckpointEngineManager` 是否需要整体引入，还是继续保留 Trinity 的 `CheckpointMonitor` 体系，并只吸收必要行为差异。
4. 对 `lora.rank` 和旧 `lora_rank` 做兼容读取，避免升级后旧配置失效。

## 6. 配置同步检查清单

以下能力需要在实现迁移时同步检查 `trinity/trainer/verl/verl_config.py`：

1. `trust_remote_code`
2. `use_prefix_grouper`
3. `calculate_sum_pi_squared`
4. `sum_pi_squared_checkpointing`
5. `lora.rank` 与 `lora_rank` 的兼容入口
6. 任何新增的 actor、ref、trainer 级开关

处理原则：

- 如果 Trinity 当前配置层已经提供等价入口，则补兼容读取即可。
- 如果上游新增配置会影响当前训练路径，则需要在 dataclass 和默认值中明确声明。
- 如果上游功能已被 Trinity 明确裁剪，则不要为了“字段对齐”而盲目把整个配置树搬过来。

## 7. 建议执行顺序

推荐按以下顺序迁移：

1. `dp_actor.py`
2. `fsdp_workers.py`
3. `megatron_actor.py`
4. `megatron_workers.py`
5. `fsdp_checkpoint_manager.py`
6. `megatron_checkpoint_manager.py`
7. `verl_trainer.py`
8. `verl_config.py`

原因：

- 前四个文件更接近函数级迁移，适合作为新版字段和配置链路的落点
- 两个 checkpoint manager 需要等 actor/worker 输出契约稳定后再处理
- `verl_trainer.py` 依赖前面多个模块的接口，最好最后收口
- `verl_config.py` 应在全部实现点明确后统一补齐，避免中途反复改字段

## 8. 本地无 GPU 场景下的检查方式

由于本地没有 GPU，本次本地工作以静态检查和执行方案导出为主：

1. 完成逐文件三路合并后，至少做 Python 语法和 import 层面的静态检查。
2. 检查新增配置字段是否在 dataclass、默认值和读取路径上闭环。
3. 检查 actor 输出字段与 worker/trainer 消费字段是否一致。
4. 检查 checkpoint 保存和加载入口的函数签名是否自洽。
5. 导出一份远端服务器验证清单，用于 GPU 环境中的最小回归测试。

建议的远端验证项：

1. FSDP 路径最小训练启动
2. Megatron 路径最小训练启动
3. checkpoint 保存与恢复
4. LoRA 配置兼容性
5. `trust_remote_code` 模型加载
6. 如果启用了相关功能，再补 `use_prefix_grouper` 和 `calculate_sum_pi_squared` 的专项验证

## 9. 实施时的注意事项

1. 不要用整文件覆盖的方式做升级。
2. 不要把 Trinity 已删除的 rollout 功能重新带回。
3. 不要保留已经被上游修复的本地 workaround。
4. 不要在 checkpoint manager 和 trainer 中继续扩大复制代码范围。
5. 每处理一个高风险文件，都要重新确认其对外接口是否影响后续文件。

## 10. 本次升级的落地输出

本次升级完成前，至少应形成以下产物：

1. 7 个目标文件的适配修改
2. `verl_config.py` 的必要配置同步
3. 一份远端 GPU 验证清单
4. 对仍保留的 Trinity 定制点做简短说明，方便下一次 veRL 版本升级继续复用本计划

## 11. 本次实际操作总结

本节记录本次从 veRL v0.7.0 迁移到 v0.7.1 时已经完成的具体工作，以及后续升级时必须优先知道的仓库约束。

### 11.1 已完成的迁移内容

本次已经完成以下文件的适配与收敛：

1. `trinity/trainer/verl/dp_actor.py`
2. `trinity/trainer/verl/fsdp_workers.py`
3. `trinity/trainer/verl/monkey_patch.py`
4. `trinity/trainer/verl/verl_config.py`
5. `trinity/trainer/verl/megatron_actor.py`
6. `trinity/trainer/verl/megatron_workers.py`
7. `trinity/trainer/verl/fsdp_checkpoint_manager.py`
8. `trinity/trainer/verl/megatron_checkpoint_manager.py`
9. `trinity/trainer/verl/verl_trainer.py`

并已完成统一静态检查：

1. 以上文件 Problems 检查均无错误。
2. 以上文件统一 `python -m py_compile` 已通过。

### 11.2 已迁入的 v0.7.1 关键能力

本次已明确接入并保留的 v0.7.1 相关能力包括：

1. `use_prefix_grouper` 配置与模型 patch 接线。
2. `calculate_sum_pi_squared` / `sum_pi_squared_checkpointing` 配置入口。
3. FSDP actor logprob 路径对 `sum_pi_squared` 的透传。
4. FSDP / Megatron worker 对 `pad_token_id` 的显式传递。
5. Megatron actor 动态 batch 路径中 `dp_group` 的传递。
6. Megatron worker 的 LoRA reference logprob 路径。
7. Megatron worker 的 `images_seqlens` MFU 统计输入。
8. Checkpoint manager 中与 v0.7.1 对齐的 checkpoint 保留策略。
9. FSDP HF 保存路径中的 `trust_remote_code`。
10. Megatron checkpoint 中的 `mbridge_config` 参数透传。
11. Trainer 路径中的 `images_seqlens` 元信息与 `compute_variance_proxy_metrics`。

### 11.3 已删除或回退到上游的本地补丁

本次有意识地删除了一部分已经与上游一致、或者仅是历史兼容残留的本地实现：

1. `dp_actor.py` 中与父类一致的 `_forward_micro_batch` 和 `compute_log_prob` 已删除，改为直接继承上游实现。
2. `megatron_checkpoint_manager.py` 中与父类一致的 `generate_state_dict`、`_build_sharded_state_dict_metadata`、`load_checkpoint` 已删除。
3. `megatron_checkpoint_manager.py` 中本地实现的 `save_dist_checkpointing` 兼容补丁已删除，改回上游实现。

这类清理在下次升级时应优先复查，原则仍然是：

1. 如果本地实现只是复制上游代码，优先删掉。
2. 只有在 Trinity 真的增加了新职责时才保留子类实现。

### 11.4 本仓库中已确认的训练职责边界

下次升级前必须优先记住以下边界，否则很容易误把上游代码重新搬回 Trinity：

1. Trinity 中 reward 计算不在 `verl_trainer.py` 内执行。
2. Trinity 中 rollout 不是在 veRL trainer 内执行，而是独立模块负责。
3. Trinity 中 trainer 侧 validation 当前未实现，因此不要为了对齐上游而引入 `_validate()` 或 validation reward 主流程。
4. Trinity 训练并不直接走上游 `RayPPOTrainer.fit()` 主循环，而是走 `trinity/trainer/trainer.py` 中调度的 `prepare()`、`train_step()`、`save_checkpoint()`、`save_state_dict()`、`upload_state_dict()` 路径。

这意味着下次升级 `verl_trainer.py` 时：

1. 不能简单照搬上游 `fit()`。
2. 不能把 `extract_reward`、reward loop、async rollout manager、validation 流程当成默认必迁内容。
3. 需要优先检查“当前 Trinity 自定义训练路径”是否仍与上游 worker / checkpoint / config 契约兼容。

### 11.5 当前仍保留的 Trinity 定制点

以下定制点在本次迁移后仍被保留，后续升级不能误删：

1. `dp_actor.py` 与 `megatron_actor.py` 中 Trinity 自定义算法接入与 loss 组合逻辑。
2. `fsdp_checkpoint_manager.py` 与 `megatron_checkpoint_manager.py` 中基于 Ray 的 `CheckpointMonitor` / `Synchronizer` 协作逻辑。
3. `verl_trainer.py` 中的 `CheckpointMonitor`、状态同步路径、Trinity 自定义 `train_step()`。
4. Trinity 独立训练调度器与 experience pipeline 的对接方式。

### 11.6 本次迁移中确认的潜在约束

以下约束已经在本次适配中被验证或暴露出来：

1. `sum_pi_squared` 只有在 actor 侧显式开启配置后才能用于 optimal token baseline 类优势函数。
2. `use_prefix_grouper` 是一条从 config 到 monkey patch 到 actor / ref worker 的整链路能力，不能只改单个文件。
3. Megatron 路径下 LoRA reference logprob 不是独立 ref worker 逻辑，而是会走 actor/no-adapter 路径。
4. 多模态训练路径如果要做 MFU 统计，需要在 trainer 中补 `images_seqlens` 到 `batch.meta_info`。
5. Checkpoint manager 的本地异步线程、保存计数和 checkpoint 注册逻辑不能被上游整文件覆盖掉。

### 11.7 下次升级时的推荐执行顺序

如果后续继续升级 veRL 版本，建议沿用以下顺序：

1. 先重新生成 `build/<version>/` 的格式化上游快照。
2. 先检查 actor 与 worker 文件，因为这些文件决定数据字段与配置链路。
3. 再检查 checkpoint manager，因为它们依赖前面的接口契约。
4. 最后检查 `verl_trainer.py`，并始终以 Trinity 自己的训练职责边界为前提。
5. 在每一步都优先做“是否还能删除重复子类实现”的收敛检查。

### 11.8 远端 GPU 最小回归建议

本地已完成的仅是静态验证。后续应在远端 GPU 环境至少覆盖以下回归项：

1. FSDP 单步训练。
2. Megatron 单步训练。
3. old logprob / ref logprob 重算路径。
4. 开启 LoRA 时的 Megatron reference logprob 路径。
5. checkpoint 保存与恢复。
6. 开启 `use_prefix_grouper` 时的最小回归。
7. 开启 `calculate_sum_pi_squared` 时的最小回归。

### 11.9 一句话迁移准则

下次升级 veRL 时，优先迁移当前 Trinity 训练路径真实会消费的上游契约；不要为了“版本对齐”而把 reward、rollout、validation 等不由 Trinity trainer 执行的上游逻辑重新搬回仓库。
