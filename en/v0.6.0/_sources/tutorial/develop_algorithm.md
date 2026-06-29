(Algorithms)=
## Algorithms Development Guide

In Trinity-RFT, the algorithm module is primarily responsible for extracting experience data from the Replay Buffer during the RL process and calculating the loss to update models based on this data.

To avoid implementing a new Trainer class each time a new algorithm is added, we have decomposed the representative PPO algorithm process into multiple sub-modules to adapt to various algorithms.

### Step 0: Basic Concepts of Algorithm Module

Trinity-RFT breaks down the algorithm module into the following sub-modules:

- **Sample Strategy** ({class}`trinity.algorithm.SampleStrategy`): Responsible for sampling experience data from the buffer module. By customizing this module, you can implement functionalities like filtering experience data or mixed sampling from multiple data sources.
- **Advantage Fn**({class}`trinity.algorithm.AdvantageFn`): Responsible for calculating the Advantage and Returns of experience data.
- **Policy Loss Fn**({class}`trinity.algorithm.PolicyLossFn`): Responsible for calculating the core training loss of the policy network.
- **KL Fn**({class}`trinity.algorithm.KLFn`): Responsible for calculating KL Divergence, which is generally used in two places in existing RL algorithms: Reward Penalty and Actor Loss.
- **Entropy Loss Fn**({class}`trinity.algorithm.EntropyLossFn`): Responsible for calculating the entropy loss of the policy network.

We provide several implementations of above modules in `trinity/algorithm`.

---

### Step 1: Implement Algorithm Components


Trinity-RFT allows developers to customize all the above modules. Developers only need to implement specific modules according to the requirements of their new algorithm. This section will provide a simple introduction using the {ref}`OPMD <OPMD>` algorithm as an example.

The main difference between OPMD and PPO algorithms lies in the calculation of Advantage and Policy Loss.
OPMD relies on a group-based advantage calculation and does not use the Critic model.
To implement OPMD, developers need to implement advantage calculation in `AdvantageFn` and policy loss calculation in `PolicyLossFn`.

---

#### Step 1.1: Implement `AdvantageFn`

The {class}`trinity.algorithm.AdvantageFn` interface includes three methods:

- `__call__`: The main entrance for advantage calculation. It receives a list of experiences ({class}`trinity.common.experience.Experience`) and returns a list of experiences with calculated advantages and returns, along with a metrics dictionary for logging.
- `default_args`: A class method that returns the default initialization parameters in dictionary form. It will be used by default when users don't specify initialization parameters in the configuration file.
- `compute_in_trainer`: This class method indicates whether to compute advantages in the Trainer. If it returns `False`, the `AdvantageFn` will be called in the experience data processing pipeline.

For convenience, Trinity-RFT provides an abstract class {class}`trinity.algorithm.advantage_fn.GroupAdvantage` that implements the `__call__` method for group-based advantage calculation, you can focus on how to group the experiences and calculate advantages on grouped experiences with the following two methods:

- `group_experiences`: This method groups a experiences generated in a step into multiple sub-groups.

- `calculate_group_advantage`: This method calculates the advantage for each group of experiences.

Here's an implementation example for the OPMD algorithm's advantage function:

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

After implementation, you need to register this module in the `default_mapping` of `trinity/algorithm/__init__.py`. Once registered, the module can be configured in the configuration file using the registered name.


#### Step 1.2: Implement `PolicyLossFn`

Developers need to implement the {class}`trinity.algorithm.PolicyLossFn` interface, which is similar to `AdvantageFn` and includes two methods:

- `__call__`: Calculates the loss based on input parameters. Unlike `AdvantageFn`, the input parameters here are all `torch.Tensor`. This interface automatically scans the parameter list of the `__call__` method and converts it to the corresponding fields in the experience data. Therefore, please write all tensor names needed for loss calculation directly in the parameter list, rather than selecting parameters from `kwargs`.
- `default_args`: Returns default initialization parameters in dictionary form, which will be used by default when users don't specify initialization parameters in the configuration file.

Similarly, after implementation, you need to register this module in the `default_mapping` of `trinity/algorithm/policy_loss_fn/__init__.py`.

Here's an implementation example for the OPMD algorithm's Policy Loss Fn. Since OPMD's Policy Loss only requires logprob, action_mask, and advantages, only these three items are specified in the parameter list of the `__call__` method:


```python
class OPMDPolicyLossFn(PolicyLossFn):
    def __init__(self, tau: float = 1.0) -> None:
        self.tau = tau

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
        return {"tau": 1.0, "loss_agg_mode": "token-mean"}
```

---

### Step 2: Register Your Algorithm

The above steps implement the components needed for the algorithm, but these components are scattered and need to be configured in multiple places to take effect.

To simplify configuration, Trinity-RFT provides {class}`trinity.algorithm.AlgorithmType` to describe a complete algorithm and registers it in `trinity/algorithm/__init__.py`, enabling one-click configuration.

The `AlgorithmType` class includes the following attributes and methods:

- `use_critic`: Whether to use the Critic model
- `use_reference`: Whether to use the Reference model
- `compute_advantage_in_trainer`: Whether to calculate Advantages in Trainer; if False, the `AdvantageFn` call in trainer will be skipped
- `can_balance_batch`: Whether the algorithm allows automatic balancing when splitting a batch into microbatches (which permute the order of samples)
- `schema`: The format of experience data corresponding to the algorithm
- `default_config`: Gets the default configuration of the algorithm, which will override attributes with the same name in `ALGORITHM_TYPE`

Similarly, after implementation, you need to register this module in the `default_mapping` of `trinity/algorithm/__init__.py`.

Below is the implementation for the OPMD algorithm.
Since the OPMD algorithm doesn't need to use the Critic model, `use_critic` is set to `False`.
The dictionary returned by the `default_config` method indicates that OPMD will use the `opmd` type `AdvantageFn` and `PolicyLossFn` implemented in Step 1, will not apply KL Penalty on rewards, but will add a `k2` type KL loss when calculating the final loss.

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
To ensure the registration takes effect, please make sure to import the newly implemented modules in the corresponding `__init__.py` files, for example:

- Import `OPMDGroupAdvantage` in `trinity/algorithm/advantage_fn/__init__.py`
- Import `OPMDPolicyLossFn` in `trinity/algorithm/policy_loss_fn/__init__.py`
- Import `OPMDAlgorithm` in `trinity/algorithm/__init__.py`

You can also place these classes in the `trinity/plugins` directory, and Trinity-RFT will automatically load all modules in the `plugins` directory at startup, without needing to import them in `__init__.py`.
```

---

### Step 3: Use Your Algorithm

After completing all the above steps, you can use the newly registered algorithm through a YAML configuration file.

For default configurations, you just need to add the following content to your `config.yaml` file:

```yaml
# some other configs
algorithm:
  algorithm_type: "opmd"
# some other configs
```

If you need to modify certain parameters, you can simply add the corresponding parameters within the `algorithm` section. For example, if you need to modify `repeat_times` and the initialization parameters of `AdvantageFn` and `PolicyLossFn`, the modified `config.yaml` file would be as follows:

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
