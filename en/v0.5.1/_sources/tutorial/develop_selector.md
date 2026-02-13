# 🧪 Experimental: Task Selection

```{note}
This module is currently in **experimental status**. Interfaces may change in future versions.
This document describes the functionality and intended usage of the system.
```



## Overview

This system enables **intelligent, adaptive task sampling** from multiple datasets (called *tasksets*) during exploration. It consists of two core components:

1. **`Selector`** – Controls how individual samples are selected *within* each taskset.
2. **`TasksetScheduler`** – Manages *which* tasksets contribute to each batch and coordinates their sampling.

Together, they support advanced training strategies such as:
- Curriculum learning (easy → hard)
- Multi-task interleaving or mixing
- Difficulty-aware sampling
- Adaptive data selection based on model performance

These capabilities allow you to train models more efficiently by focusing on informative or challenging examples.



## Module 1: Selector – Customizable Data Selection

A `Selector` determines **which tasks (samples) to select** from its associated dataset (`Taskset`). Beyond basic strategies like sequential or random access, it supports **adaptive algorithms** that adjust sampling based on feedback—such as sample difficulty, model confidence, or reward signals.

### Built-in Selectors

| Selector Type | Description |
|---------------|-------------|
| `sequential` | Returns samples in fixed order (0, 1, ..., N). |
| `shuffle` | Shuffles the dataset once per epoch; then iterates sequentially. |
| `random` | Randomly samples without replacement within each batch. Independent across batches. |
| `offline_easy2hard` | Sorts samples by pre-defined features (e.g., loss, length), serving easier ones first, progressing to harder ones. |
| `difficulty_based` *(custom example)* | Dynamically selects samples near a target difficulty level using probabilistic modeling. |

You can also **implement your own custom selector** to enable adaptive or curriculum-based learning.



### ✅ Step 1: Implement a Custom Selector

To create a new selector, inherit from `BaseSelector` and implement the following methods:

#### Required Methods

| Method | Purpose |
|-------|--------|
| `get_indices(batch_size: int, return_extra_info=False) -> List[int]` | Return a list of sample indices to read next. |
| `feedback(indices: List[int], values: List[float])` | Update internal state using feedback (e.g., rewards, losses). |
| `state_dict() -> Dict` | Serialize current state for checkpointing. |
| `load_state_dict(state_dict: Dict)` | Restore state from a saved dictionary. |

#### Example: `DifficultyBasedSelector`

This selector focuses on samples whose predicted performance is closest to a target (e.g., 90% success rate), effectively choosing "just right" difficulty tasks.

```python
class DifficultyBasedSelector(BaseSelector):
    def __init__(self, data_source, config: TaskSelectorConfig) -> None:
        super().__init__(data_source, config)
        self.logger = get_logger("difficulty_based_selector")

        # Build difficulty estimator using two input features (e.g., correctness, uncertainty)
        self.diff_estimator = self.build_diff_estimator(
            data_source.dataset, config.feature_keys, config.kwargs
        )
        self.current_index = 0
        self.seed = config.seed

        # Configuration parameters
        self.do_sample = config.kwargs.get("do_sample", False)
        self.target_reward = config.kwargs.get("target_reward", 1.0)
        self.tau = config.kwargs.get("tau", 1.0)

    # ... detailed implementation

    def get_indices(self, batch_size, return_extra_info=False):
        # Compute scores based on proximity to target reward
        sampling_scores = self.get_scores()
        sampling_scores = torch.from_numpy(sampling_scores)

        if self.tau == 0:
            # Greedy: take top-k highest scoring samples
            selected_indices = torch.topk(sampling_scores, batch_size).indices
        else:
            # Stochastic: sample via softmax with temperature scaling
            sampling_logits = sampling_scores / self.tau
            sampling_logits -= sampling_logits.max()  # Stability
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
            # Optional debugging info
            extra_info = {
                "indices": selected_indices.tolist(),
                "scores": sampling_scores[selected_indices].tolist(),
                # ... other metadata
            }
            return selected_indices, extra_info
        else:
            return selected_indices

    def feedback(self, indices: List[int], values: List[float]) -> None:
        # Update difficulty model with observed rewards
        self.diff_estimator.update(indices, values)

    def state_dict(self) -> Dict:
        return {"current_index": self.current_index}

    def load_state_dict(self, state_dict: Dict) -> None:
        self.current_index = state_dict.get("current_index", 0)
```

> 🔁 After defining your class, remember to register it in the `default_mapping` of `trinity/buffer/selector/__init__.py` so it can be referenced by name in configs.
```python
SELECTORS = Registry(
    "selectors",
    default_mapping={
        "difficulty_based": "trinity.buffer.selector.selector.DifficultyBasedSelector",
    },
)
```



### ✅ Step 2: Implement a Feedback Operator

For adaptive selectors like `DifficultyBasedSelector`, you need to provide runtime feedback (e.g., task rewards). This is done via an **Experience Operator** that processes rollouts and computes metrics.

> 📚 See the {ref}`Operator Development Guide<Operators>` for more on building custom experience processors.

The operator must output a metric under the key `trinity.common.constants.SELECTOR_METRIC`, structured as:

```python
{
    SELECTOR_METRIC: {
        0: {  # taskset_id
            "indices": [10, 25, 43],
            "values": [0.8, 0.6, 0.9]  # e.g., average reward
        },
        1: { ... }
    }
}
```

#### Example: Pass Rate Calculator

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

This operator calculates the average reward per task and passes it back to the corresponding selector for updating difficulty estimates.



### ✅ Step 3: Update Configuration

After implementing your selector and operator, register them in the config file.

#### Add the Operator to the Pipeline

```yaml
data_processor:
  experience_pipeline:
    operators:
      - name: pass_rate_calculator
```

#### Configure the Taskset with Your Selector

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

> 💡 You can define multiple tasksets, each with its own selector type and configuration.



## Module 2: TasksetScheduler – Multi-Taskset Orchestration

The `TasksetScheduler` manages **how different tasksets are interleaved or mixed** during training.

### Key Features

- Supports **multiple tasksets** simultaneously.
- Balances sampling proportionally to dataset sizes.
- **Shuffles taskset access order** at the start of each epoch.
- Enables **curriculum-style** or **interleaved multi-task training**.
- Fully **checkpointable**: resumes exactly where it left off.
- Integrates with any registered `Selector`.

### How It Works

At each training step:
1. Determines which tasksets should contribute to the current batch.
2. Queries each taskset’s selector to get specific sample indices.
3. Reads the actual data asynchronously.
4. Tags each task with `"taskset_id"` for downstream routing or analysis.

Epochs are defined based on total data volume and batch size:
```python
steps_per_epoch = total_samples // batch_size
```

At the beginning of each epoch, the scheduler reshuffles the sequence of taskset accesses to introduce variability.



## Summary

With these components, you can:
- Use simple strategies like random or sequential sampling.
- Design **adaptive curricula** using custom selectors.
- Combine multiple datasets intelligently.
- Optimize training efficiency by focusing on high-value samples.

By combining smart `Selectors` with the flexible `TasksetScheduler`, you gain fine-grained control over what your model sees—and when.
