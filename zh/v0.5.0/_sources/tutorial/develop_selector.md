# 🧪 实验性功能：任务选择器

```{note}
该模块目前处于 **实验阶段**，接口可能在后续版本中发生变化。
本文档描述了系统的功能及预期使用方式。
```

## 概述

本系统支持在探索过程中，从多个数据集/任务集（称为 *tasksets*）中进行**智能、自适应的任务采样**。它包含两个核心组件：

1. **`Selector`（选择器）** —— 控制每个任务集中**如何选择单个样本**。
2. **`TasksetScheduler`（任务集调度器）** —— 管理**哪些任务集参与当前批次的训练**，并协调它们的采样过程。

二者结合，支持以下高级训练策略：
- 课程学习（由易到难）
- 多任务交替/混合训练
- 基于难度的采样
- 根据模型表现动态调整数据选择

这些能力使你能够更高效地训练模型，聚焦于信息量大或具有挑战性的样本。



## 模块 1：Selector —— 可定制的数据选择机制

`Selector` 决定从其对应的数据集（`Taskset`）中选择哪些**任务（样本）**。除了基本的顺序或随机访问策略外，它还支持**基于反馈信号（如样本难度、模型置信度、奖励等）动态调整采样行为的自适应算法**。

### 内置的选择器类型

| 选择器类型 | 说明 |
|-----------|------|
| `sequential` | 按固定顺序返回样本（0, 1, ..., N）。 |
| `shuffle` | 每个 epoch 开始时对数据集整体打乱一次，之后按顺序遍历。 |
| `random` | 在每个 batch 中无放回地随机采样，不同 batch 之间相互独立。 |
| `offline_easy2hard` | 根据预定义特征（如损失值、长度）对样本排序，先提供简单样本，逐步过渡到困难样本。 |
| `difficulty_based` *(自定义示例)* | 使用概率建模动态选择接近目标难度水平的样本。 |

你也可以实现自己的**自定义选择器**，以支持自适应或课程式学习。



### ✅ 步骤 1：实现一个自定义选择器

要创建新的选择器，需继承 `BaseSelector` 类，并实现以下方法：

#### 必须实现的方法

| 方法 | 功能说明 |
|------|---------|
| `get_indices(batch_size: int, return_extra_info=False) -> List[int]` | 返回接下来要读取的样本索引列表。 |
| `feedback(indices: List[int], values: List[float])` | 使用反馈信息（如奖励、损失）更新内部状态，用于自适应调整。 |
| `state_dict() -> Dict` | 序列化当前状态，用于保存检查点。 |
| `load_state_dict(state_dict: Dict)` | 从保存的状态字典中恢复选择器状态。 |

#### 示例：`DifficultyBasedSelector`

该选择器聚焦于模型预测表现最接近目标值的样本（例如 90% 成功率），从而挑选出“难度适中”的任务。

```python
class DifficultyBasedSelector(BaseSelector):
    def __init__(self, data_source, config: TaskSelectorConfig) -> None:
        super().__init__(data_source, config)
        self.logger = get_logger("difficulty_based_selector")

        # 使用两个输入特征（如正确性、不确定性）构建难度估计器
        self.diff_estimator = self.build_diff_estimator(
            data_source.dataset, config.feature_keys, config.kwargs
        )
        self.current_index = 0
        self.seed = config.seed

        # 配置参数
        self.do_sample = config.kwargs.get("do_sample", False)
        self.target_reward = config.kwargs.get("target_reward", 1.0)
        self.tau = config.kwargs.get("tau", 1.0)

    # ... 具体实现省略

    def get_indices(self, batch_size, return_extra_info=False):
        # 计算得分：越接近目标奖励得分越高
        sampling_scores = self.get_scores()
        sampling_scores = torch.from_numpy(sampling_scores)

        if self.tau == 0:
            # 贪心策略：选择得分最高的 top-k 样本
            selected_indices = torch.topk(sampling_scores, batch_size).indices
        else:
            # 随机采样：通过带温度的 softmax 进行采样
            sampling_logits = sampling_scores / self.tau
            sampling_logits -= sampling_logits.max()  # 数值稳定性处理
            sampling_probabilities = torch.softmax(sampling_logits, dim=0)
            rng = torch.Generator().manual_seed(self.seed + self.current_index)
            selected_indices = torch.multinomial(
                sampling_probabilities,
                batch_size,
                replacement=False,
                generator=rng,
            )

        self.current_index += batch_size

        if return_extra_info:
            # 可选：返回调试信息
            extra_info = {
                "indices": selected_indices.tolist(),
                "scores": sampling_scores[selected_indices].tolist(),
                # ... 其他元数据
            }
            return selected_indices, extra_info
        else:
            return selected_indices

    def feedback(self, indices: List[int], values: List[float]) -> None:
        # 使用观测到的奖励更新难度模型
        self.diff_estimator.update(indices, values)

    def state_dict(self) -> Dict:
        return {"current_index": self.current_index}

    def load_state_dict(self, state_dict: Dict) -> None:
        self.current_index = state_dict.get("current_index", 0)
```

> 🔁 定义完类后，请在 `trinity/buffer/selector/__init__.py` 中的 `default_mapping` 中注册，以便在配置文件中通过名称引用。
```python
SELECTORS = Registry(
    "selectors",
    default_mapping={
        "difficulty_based": "trinity.buffer.selector.selector.DifficultyBasedSelector",
    },
)
```


### ✅ 步骤 2：实现反馈操作器（Feedback Operator）

对于像 `DifficultyBasedSelector` 这样的自适应选择器，你需要提供运行时反馈（例如任务奖励）。这通过一个 **Experience Operator（经验操作器）** 实现，它处理 rollout 数据并计算相关指标。

> 📚 更多关于自定义经验处理器的内容，请参见 {ref}`Operator 开发指南<Operators>`。

操作器必须输出一个键为 `trinity.common.constants.SELECTOR_METRIC` 的指标，结构如下：

```python
{
    SELECTOR_METRIC: {
        0: {  # taskset_id
            "indices": [10, 25, 43],
            "values": [0.8, 0.6, 0.9]  # 例如：平均奖励值
        },
        1: { ... }
    }
}
```

#### 示例：通过率计算器（Pass Rate Calculator）

```python
class PassRateCalculator(ExperienceOperator):
    def __init__(self, **kwargs):
        pass

    def process(self, exps: List[Experience]) -> Tuple[List[Experience], Dict]:
        raw_metric = defaultdict(lambda: defaultdict(list))

        for exp in exps:
            task_index = exp.info["task_index"]
            assert "taskset_id" in task_index and "index" in task_index
            raw_metric[task_index["taskset_id"]][task_index["index"]].append(exp.reward)

        metric = {}
        for taskset_id, task_metrics in raw_metric.items():
            indices = []
            reward_means = []
            for idx, rewards in task_metrics.items():
                indices.append(idx)
                reward_means.append(float(np.mean(rewards)))
            metric[taskset_id] = {
                "indices": indices,
                "values": reward_means,
            }

        return exps, {SELECTOR_METRIC: metric}
```

该操作器计算每个任务的平均奖励，并将其传回对应的 `Selector`，用于更新难度估计。



### ✅ 步骤 3：更新配置文件

完成选择器和操作器的实现后，需要在配置文件中注册它们。

#### 将操作器加入处理流程

```yaml
data_processor:
  experience_pipeline:
    operators:
      - name: pass_rate_calculator
```

#### 为任务集配置你的选择器

```yaml
buffer:
  explorer_input:
    tasksets:
      - name: my_taskset
        storage_type: file
        path: ./path/to/tasks
        task_selector:
          selector_type: difficulty_based
          feature_keys: ["correct", "uncertainty"]
          kwargs:
            m: 16
            lamb: 0.2
            rho: 0.2
            target_reward: 0.9
            tau: 0.5
            do_sample: true
```

> 💡 你可以定义多个任务集，每个都可以使用不同类型和配置的选择器。



## 模块 2：TasksetScheduler —— 多任务集协调调度

`TasksetScheduler` 负责管理训练过程中**不同任务集之间的交错方式**。

### 主要特性

- 支持**同时加载多个任务集**。
- 按数据集大小比例**平衡采样权重**。
- 每个 epoch 开始时**打乱任务集的访问顺序**。
- 支持**课程式学习**或**多任务交替/混合训练**。
- 完全**可恢复断点**：能精确从中断处继续训练。
- 与任意已注册的 `Selector` 无缝集成。

### 工作原理

在每一步训练中：
1. 确定哪些任务集应参与当前 batch；
2. 向各任务集的选择器请求具体的样本索引；
3. 异步读取实际数据；
4. 为每个任务打上 `"taskset_id"` 标签，便于下游路由或分析。

每个 epoch 的步数由总样本数和 batch size 决定：
```python
steps_per_epoch = total_samples // batch_size
```

每个 epoch 开始时，调度器会重新打乱任务集的访问顺序，以增加多样性。



## 总结

通过这两个组件，你可以：
- 使用简单的策略（如随机或顺序采样）；
- 利用自定义选择器设计**自适应课程学习策略**；
- 智能地融合多个数据集；
- 通过聚焦高价值样本提升训练效率。

将智能的 `Selector` 与灵活的 `TasksetScheduler` 结合，你将获得对模型所见内容及其出现时机的精细控制能力。
