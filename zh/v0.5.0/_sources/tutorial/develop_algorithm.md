(Algorithms)=
## Algorithm 开发指南

在 Trinity-RFT 中，Algorithm 模块主要负责在 RL 过程中从 Buffer 提取 experience 数据，并基于这些数据计算损失以更新模型。为了避免每次添加新算法时都实现新的 Trainer 类，我们将典型的 PPO 算法流程分解为多个子模块，以适应各种 RL 算法。

### 步骤 0：Algorithm 子模块概览

Trinity-RFT 将 Algorithm 模块拆分为以下几个子模块：

- **采样策略（Sample Strategy）** ({class}`trinity.algorithm.SampleStrategy`)：负责从缓冲区模块中采样 experience 数据。通过自定义此模块，你可以实现过滤 experience 数据或从多个数据源混合采样的功能。
- **优势函数（Advantage Fn）**({class}`trinity.algorithm.AdvantageFn`)：负责计算 experience 数据的优势值（Advantage）和回报值（Returns）。
- **策略损失函数（Policy Loss Fn）**({class}`trinity.algorithm.PolicyLossFn`)：负责计算策略网络的核心训练损失。
- **KL 函数（KL Fn）**({class}`trinity.algorithm.KLFn`)：负责计算 KL 散度，通常在现有 RL 算法中用于两个地方：奖励惩罚和 Actor 损失。
- **熵损失函数（Entropy Loss Fn）**({class}`trinity.algorithm.EntropyLossFn`)：负责计算策略网络的熵损失。

我们在 `trinity/algorithm` 中提供了上述模块的若干实现。

---

### 步骤 1：实现算法组件

Trinity-RFT 允许开发者自定义所有上述模块。开发者只需根据新算法的需求实现特定模块。本节将以 {ref}`OPMD <OPMD>` 算法为例进行简要介绍。

OPMD 与 PPO 算法的主要区别在于优势值和策略损失的计算。OPMD 依赖于基于组的优势值计算，且不使用 Critic 模型。为此，开发者需要实现新的优势函数 (`AdvantageFn`) 以及策略损失函数 (`PolicyLossFn`)。

---

#### 步骤 1.1：实现 `AdvantageFn`

{class}`trinity.algorithm.AdvantageFn` 接口包含三个方法：

- `__call__`：优势值计算的主要入口。接收一个 experience 列表 ({class}`trinity.common.experience.Experience`)，返回一个包含计算出的优势值和回报值的 experience 列表，以及一个用于日志记录的指标字典。
- `default_args`：类方法，返回默认初始化参数（字典形式）。当用户未在配置文件中指定初始化参数时，默认使用此方法返回的参数。
- `compute_in_trainer`：类方法，指示是否在 Trainer 中计算优势值。若返回 `False`，则 `AdvantageFn` 将在 experience 数据处理流水线中被调用。

为方便起见，Trinity-RFT 提供了一个抽象类 {class}`trinity.algorithm.advantage_fn.GroupAdvantage`，它实现了基于组的优势值计算的 `__call__` 方法，你可以专注于如何对 experience 进行分组以及如何在分组后的 experience 上计算优势值，通过以下两个方法实现：

- `group_experiences`：此方法将一步生成的 experience 划分为多个子组。

- `calculate_group_advantage`：此方法计算每组 experience 的优势值。

以下是 OPMD 算法优势函数的实现示例：

```python
from trinity.algorithm.advantage_fn import GroupAdvantage

class OPMDGroupAdvantage(GroupAdvantage):
    """OPMD Group Advantage computation"""

    def __init__(self, opmd_baseline: str = "mean", tau: float = 1.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.opmd_baseline = opmd_baseline
        self.tau = tau

    def group_experiences(self, exps):
        return group_by(exps, id_type="task")

    def calculate_group_advantage(
        self, group_id: str, exps: List[Experience]
    ) -> Tuple[List[Experience], Dict]:
        with torch.no_grad():
            if len(exps) == 1:
                group_baseline = torch.tensor(0.0)
            else:
                group_rewards = torch.tensor([exp.reward for exp in exps], dtype=torch.float32)
                if self.opmd_baseline == "mean":
                    group_baseline = torch.mean(group_rewards)
                else:
                    group_baseline = self.tau * (
                        torch.logsumexp(group_rewards / self.tau, dim=-1)
                        - torch.log(torch.tensor(len(exps)))
                    )
            for exp in exps:
                score = exp.reward - group_baseline
                exp.advantages = score * exp.action_mask
                exp.returns = exp.advantages.clone()
            metrics = {
                "group_baseline": group_baseline.item(),
            }
        return exps, metrics

    @classmethod
    def default_args(cls) -> dict:
        return {"opmd_baseline": "mean", "tau": 1.0}
```

实现后，你需要在 `trinity/algorithm/__init__.py` 中的 `default_mapping` 中注册此模块。注册后，该模块可在配置文件中使用注册名称进行配置。

#### 步骤 1.2：实现 `PolicyLossFn`

开发者需要实现 {class}`trinity.algorithm.PolicyLossFn` 接口，其与 `AdvantageFn` 类似，包含两个方法：

- `__call__`：根据输入参数计算损失。与 `AdvantageFn` 不同，这里的输入参数均为 `torch.Tensor`。该接口会自动扫描 `__call__` 方法的参数列表，并将其转换为 experience 数据中的对应字段。因此，请直接在参数列表中写出损失计算所需的所有张量名称，而不是从 `kwargs` 中选择参数。
- `default_args`：返回默认初始化参数（字典形式），当用户未在配置文件中指定初始化参数时，默认使用此方法返回的参数。

同样，实现后需要在 `trinity/algorithm/policy_loss_fn/__init__.py` 中的 `default_mapping` 中注册此模块。

以下是 OPMD 算法策略损失函数的实现示例。由于 OPMD 的策略损失仅需 logprob、action_mask 和 advantages，因此 `__call__` 方法的参数列表中仅指定这三个项：

```python
class OPMDPolicyLossFn(PolicyLossFn):
    def __init__(
        self, backend: str = "verl", tau: float = 1.0, loss_agg_mode: str = "token-mean"
    ) -> None:
        super().__init__(backend=backend)
        self.tau = tau
        self.loss_agg_mode = loss_agg_mode

    def __call__(  # type: ignore
        self,
        logprob: torch.Tensor,
        action_mask: torch.Tensor,
        advantages: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict]:
        pg_losses = -advantages * logprob
        opmd_loss = masked_loss(pg_losses, action_mask, loss_agg_mode=self.loss_agg_mode)
        opmd_loss = opmd_loss / (1.0 + self.tau)  # for regularization (w.r.t. current pi_theta)
        return opmd_loss, {"opmd_loss": opmd_loss.detach().item()}

    @classmethod
    def default_args(cls) -> Dict:
        return {"tau": 1.0}
```

---

### 步骤 2：注册新算法

上述步骤实现了算法所需的组件，但这些组件是分散的，需要在多个地方配置才能生效。

为简化配置，Trinity-RFT 提供了 {class}`trinity.algorithm.AlgorithmType` 来描述完整算法，并在 `trinity/algorithm/__init__.py` 中注册，实现一键配置。

`AlgorithmType` 类包含以下属性和方法：

- `use_critic`：是否使用 Critic 模型
- `use_reference`：是否使用 Reference 模型
- `compute_advantage_in_trainer`：是否在 Trainer 中计算优势值；若为 False，则跳过 Trainer 中的 `AdvantageFn` 调用
- `can_balance_batch`：算法是否允许在将批次拆分为微批次时自动平衡（打乱样本顺序）
- `schema`：算法对应的 experience 数据格式
- `default_config`：获取算法的默认配置，将覆盖 `ALGORITHM_TYPE` 中同名属性

同样，实现后需要在 `trinity/algorithm/__init__.py` 中的 `default_mapping` 中注册此模块。

以下是 OPMD 算法的实现。
由于 OPMD 算法不需要使用 Critic 模型，`use_critic` 设置为 `False`。
`default_config` 方法返回的字典表明 OPMD 将使用步骤 1 中实现的 `opmd` 类型的 `AdvantageFn` 和 `PolicyLossFn`，不会对奖励应用 KL 惩罚，但在计算最终损失时会添加 `k2` 类型的 KL 损失。

```python
class OPMDAlgorithm(AlgorithmType):
    """OPMD algorithm."""

    use_critic: bool = False
    use_reference: bool = True
    compute_advantage_in_trainer: bool = False
    can_balance_batch: bool = True
    schema: str = "experience"

    @classmethod
    def default_config(cls) -> Dict:
        return {
            "repeat_times": 2,
            "advantage_fn": "opmd",
            "sample_strategy": "warmup",
            "policy_loss_fn": "opmd",
            "kl_penalty_fn": "none",
            "kl_loss_fn": "k2",
            "entropy_loss_fn": "default",
        }
```


```{tip}
为了保证注册生效，请确保在对应的 __init__.py 文件中导入了新实现的模块，例如：

- 在 `trinity/algorithm/advantage_fn/__init__.py` 中导入 `OPMDGroupAdvantage`
- 在 `trinity/algorithm/policy_loss_fn/__init__.py` 中导入 `OPMDPolicyLossFn`
- 在 `trinity/algorithm/__init__.py` 中导入 `OPMDAlgorithm`

也可以将这些类放在 `trinity/plugins` 目录下，Trinity-RFT 会在启动时自动加载 `plugins` 目录中的所有模块，无需在 `__init__.py` 中导入。
```

---

### 步骤 3：使用新算法

完成上述所有步骤后，你可以通过 YAML 配置文件使用新注册的算法。

对于默认配置，你只需在 `config.yaml` 文件中添加以下内容：

```yaml
# some other configs
algorithm:
  algorithm_type: "opmd"
# some other configs
```

如果需要修改某些参数，可以在 `algorithm` 部分直接添加对应参数。例如，若需要修改 `repeat_times` 以及 `AdvantageFn` 和 `PolicyLossFn` 的初始化参数，修改后的 `config.yaml` 文件如下：

```yaml
# some other configs
algorithm:
  algorithm_type: "opmd"
  repeat_times: 8
  advantage_fn_args:
    opmd_baseline: "logavgexp"
    tau: 0.99
  policy_loss_fn_args:
    tau: 0.99
# some other configs
```
