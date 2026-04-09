(Synchronizer)=
# Synchronizer 介绍

**Synchronizer** 是 **Trinity-RFT** 中的核心协调模块，旨在分布式环境下训练强化学习模型时，保持 **Trainer** 和 **Explorer** 组件的同步。其主要作用是确保这两个组件始终使用最新的模型权重，从而实现高效且稳定的训练。

你可以将其想象为一个交通控制器：它管理 **Explorer**（负责从环境中收集 experience）何时以及如何根据 **Trainer** 最新的模型改进来更新自身的策略。如果没有这种协调，系统可能会因为使用过时或冲突的模型版本而变得低效甚至不稳定。

---

## 工作原理：整体架构

在 Trinity-RFT 中：

- **Trainer** 从收集到的数据中学习并更新模型。
- **Explorer** 使用当前模型与环境交互，生成新的数据。
- **Synchronizer** 通过管理 **Explorer 获取最新模型权重的时机和方式**，确保两者保持同步。

为实现这一目标，Synchronizer 会：
- 监控 Trainer 和 Explorer 的状态。
- 决定何时进行同步。
- 使用一种策略协调模型权重的传输。

---

### Trainer 内部逻辑

```python
async def train(self) -> str:
    while self.train_step_num < self.total_steps:
        try:
            metrics = {}
            # sample may be blocked due to explorer does not generate enough data
            self.logger.info(f"Sample data for step {self.train_step_num + 1} started.")
            sample_task = asyncio.create_task(self._sample_data())
            while not sample_task.done():
                # sync weight to make sure the explorer can continue to explore and generate enough data
                if await self.need_sync():
                    metrics.update(await self.sync_weight())
                await asyncio.sleep(1)
            exps, sample_metrics, repr_samples = await sample_task
            metrics.update(sample_metrics)
            self.logger.info(f"Sample data for step {self.train_step_num + 1} finished.")
            metrics.update(await self.train_step(exps))
            if await self.need_sync():
                metrics.update(await self.sync_weight())
            if self.need_save():
                metrics.update(
                    await self.save_checkpoint(save_as_hf=self.save_hf_checkpoint == "always")
                )
            if self.config.trainer.enable_preview:
                self._log_experiences(repr_samples)
            self.monitor.log(metrics, self.train_step_num)
        except StopAsyncIteration:
            self.logger.info("No more samples to train. Stopping training.")
            break
        except Exception:
            self.logger.error(f"Error in Trainer:\n{traceback.format_exc()}")
            break
```

Trainer 会在以下两个时机检查是否需要同步：
- 训练过程中数据收集阶段。
- 每个训练步骤完成后。

如果需要，就会通过 Synchronizer 触发 `sync_weight()`。

---

### Explorer 内部逻辑


```python
async def explore(self) -> str:
    while True:
        try:
            self.logger.info(f"Explore step {self.explore_step_num + 1} started.")
            explore_continue = await self.explore_step()
            if not explore_continue:
                break
            if self.need_eval():
                await self.eval()
            if await self.need_sync():
                await self.sync_weight()  # Request latest weights via Synchronizer
        except Exception:
            self.logger.error(f"Error in Explorer: {traceback.format_exc()}")
            break
```

Explorer 会在以下时机检查是否需要同步：
- 完成一次探索步骤后。
- 开始下一轮数据收集前。

这确保了它始终使用最新的模型版本来生成高质量的 experience。

> ✅ **核心理念**：
> Trainer 和 Explorer 都会定期向 Synchronizer 查询状态，形成一个**紧密的反馈闭环**，使训练与探索保持同步。

---

## 同步方法：模型权重如何共享？

模型权重从 Trainer 传递到 Explorer 有 **三种方式**，每种适用于不同的运行环境。

| 方法 | 介质 | 适用场景 | 延迟 | 说明 |
|-------|--------|--------|--------|-------|
| `NCCL` | GPU 到 GPU（直连） | 同一机器，多 GPU | ⬇️ 最低 | 最快，但需在同一台机器 |
| `MEMORY` | 共享内存 / 网络 | 分布式集群 | ⬇️ 较低 | 较好平衡了速度与灵活性 |
| `CHECKPOINT` | 磁盘文件 | 跨设备、云环境或慢速系统 | ⬆️ 较高 | 兼容性最强，但较慢 |

### 1. `SyncMethod.NCCL` – 高速直连同步
- 使用 NVIDIA 的 **NCCL 库** 实现 GPU 间的直接通信。
- 极其快速 —— 适用于 Trainer 和 Explorer 运行在同一节点上的情况。
- Synchronizer 负责建立通信组并协调同步过程。

🟢 **适用场景**：具有高速互联的多 GPU 集群。

---

### 2. `SyncMethod.CHECKPOINT` – 基于磁盘的同步
- Trainer 定期将模型权重保存到磁盘。
- Synchronizer 读取保存的检查点。
- Explorer 从 Synchronizer 拉取权重。

🟡 **适用场景**：节点之间不共享内存或 GPU 的分布式环境（例如云集群），尤其是具备快速存储的情况。

> 💡 优势：完全解耦 —— 各组件可在不同机器/平台独立运行。

---

### 3. `SyncMethod.MEMORY` – 内存级同步
- Trainer 直接通过网络或共享内存将模型权重发送至 Synchronizer 的内存中。
- Explorer 从 Synchronizer 获取权重，无需访问磁盘。

🟢 **适用场景**：多节点集群中磁盘 I/O 较慢，但网络带宽充足的情况。

> ⚖️ 相比 CHECKPOINT，性能与兼容性之间取得了更好的平衡。

---

## 同步模式：何时触发同步？

有两种同步模式，定义了 Explorer **何时** 请求更新权重。

### 1. `SyncStyle.FIXED` – 固定间隔同步

- 每隔固定步数进行一次同步。
- 通过 `sync_interval` 和 `sync_offset` 配置。

| 示例 | 行为 |
|--------|---------|
| `interval=10, offset=0` | 每 10 步同步一次（两者同时开始） |
| `interval=10, offset=5` | Explorer 先运行 5 步，之后每 10 步同步一次 |

🎯 **适合**：简单、可预测的环境，探索步骤较短且奖励频繁（例如数学推理任务）。

---

### 2. `SyncStyle.EXPLORER_DRIVEN` – Explorer 驱动同步
- Explorer 自己决定何时需要新模型。
- 流程：
  1. Explorer 完成 `sync_interval` 步后，向 Synchronizer 发出更新参数的请求。
  2. Trainer 在下一次循环中发现这个请求，并完成同步。
  3. 同步完成后，Explorer 和 Trainer 继续运行。
  4. 若超时，Explorer 会在下一个周期重试。

🎯 **适合**：Explorer 节奏不固定，或希望按需更新模型。

---

### 3. `SyncStyle.TRAINER_DRIVEN` – Trainer 驱动同步
- Trainer 决定何时发布新模型。
- 流程：
  1. Trainer 每隔 `sync_interval` 步数后决定请求同步。
  2. 它会通知 Synchronizer 准备推送新模型。
  3. Explorer 在正常循环中检测该请求并响应同步。

🎯 **适合**：Trainer 训练节奏明确，Explorer 被动接收更新。

---

## 状态管理：背后发生了什么？

Synchronizer 通过跟踪 Trainer 和 Explorer 的**状态**，确保同步过程安全可控。

### 三个关键状态

| 状态 | 含义 |
|------|--------|
| `STOPPED` | 组件已停止运行 |
| `RUNNING` | 正在训练或探索中 |
| `REQUIRE_SYNC` | Explorer / Trainer 请求新权重 |

这些状态有助于避免竞态条件，保证协调过程平稳。

---

### 不同模式与方法下的状态转换

#### 🔹 NCCL 同步
- Trainer 和 Explorer 都会切换状态（`RUNNING` ↔ `REQUIRE_SYNC`）。
- 同步是“双向握手”：双方都准备好才开始传数据。
- 同步完成后，双方都回到 `RUNNING`。

![NCCL 同步](../../assets/NCCL-zh.png)

#### 🔹 CHECKPOINT/MEMORY 同步
- Trainer 通常一直保持 `RUNNING`（它只负责存权重）。
- Explorer 负责发起同步请求（切换到 `REQUIRE_SYNC`），拉取完权重后回到 `RUNNING`。
- Synchronizer 作为“中介”，负责传递模型权重给 Explorer。

![CHECKPOINT/MEMORY 同步](../../assets/STATEDICT-zh.png)

---

## 常见问题（FAQ）

### Q1: 我该选择哪种同步方法？

| 场景 | 推荐方法 |
|--------|-------------------|
| 多 GPU 集群，高速互联 | `NCCL` |
| 多节点集群，内存/网络较快 | `MEMORY` |
| 多节点，磁盘慢或网络不稳定 | `CHECKPOINT` |
| 最大兼容性（跨平台） | `CHECKPOINT` |

> ✅ **经验法则**：
> 尽可能使用 `NCCL`；否则根据基础设施选择 `MEMORY` 或 `CHECKPOINT`。

---

### Q2: 哪种同步模式更好？

| 使用场景 | 推荐模式 |
|--------|------------------|
| 短周期任务，反馈迅速（如数学问答） | `FIXED` |
| 多轮交互任务，例如多轮对话、工具调用、多步骤游戏 | `EXPLORER_DRIVEN` 或 `TRAINER_DRIVEN` |

---

## 总结：核心要点

| 特性 | 重要性 |
|-------|---------------|
| **中心化协调** | 确保 Trainer 和 Explorer 使用一致的模型权重 |
| **多种同步方法** | 适配不同硬件和部署需求 |
| **灵活的同步模式** | 支持周期性与按需更新 |
| **稳健的状态管理** | 防止冲突，保障可靠性 |
| **闭环设计** | 实现稳定高效的分布式 RL 训练 |

🎯 **最终结论**：
Synchronizer 通过智能管理模型更新在训练与探索之间的传递时机和方式，使分布式强化学习变得**可扩展、高效且可靠**。

正确配置 Synchronizer 是构建高效稳定 RL 流水线的关键。
