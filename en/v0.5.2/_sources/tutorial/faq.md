# FAQ

## Part 1: Configurations

**Q:** How to write Trinity-RFT configuration files?

**A:** The recommended way to write configurations is to use the Trinity Studio. You can launch it with:

```bash
trinity studio --port 8080
```

This provides an intuitive and user-friendly way to create and modify configuration files.

For advanced users, we recommend referring to the [configuration documentation](./trinity_configs.md) for detailed explanations of all configuration options. You can also directly edit the YAML configuration files (see the `examples` directory for templates and examples).

If you are already familiar with veRL, please refer to the [Align with veRL](./align_with_verl.md) tutorial. This guide explains how to align Trinity-RFT configuration parameters with those used in veRL, making it easier to migrate or reuse your existing veRL setups.

---

**Q:** What's the relationship between `buffer.batch_size`, `buffer.train_batch_size`, `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` and other batch sizes?

**A:** The following parameters are closely related:

- `buffer.batch_size`: The number of tasks in a batch, effective for the explorer.
- `buffer.train_batch_size`: The number of experiences in a mini-batch, effective for the trainer. If not specified, it defaults to `buffer.batch_size` * `algorithm.repeat_times`.
- `actor_rollout_ref.actor.ppo_mini_batch_size`: The number of experiences in a mini-batch, overridden by `buffer.train_batch_size`; but in the `update_policy` function, its value becomes the number of experiences in a mini-batch per GPU, i.e., `buffer.train_batch_size (/ ngpus_trainer)`. The expression of dividing `ngpus_trainer` is caused by implict data allocation to GPUs, but this do not affects the result after gradient accumulation.
- `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu`: The number of experiences in a micro-batch per GPU.

A minimal example showing their usage is as follows:

```python
def update_policy(batch_exps):
    dataloader = batch_exps.split(ppo_mini_batch_size)
    for _ in range(ppo_epochs):
        for batch_idx, data in enumerate(dataloader):
            # Split data
            mini_batch = data
            if actor_rollout_ref.actor.use_dynamic_bsz:
                micro_batches, _ = rearrange_micro_batches(
                        batch=mini_batch, max_token_len=max_token_len
                    )
            else:
                micro_batches = mini_batch.split(actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu)

            # Computing gradient
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

            # Optimizer step
            grad_norm = self._optimizer_step()
    self.actor_optimizer.zero_grad()
```
Please refer to `trinity/trainer/verl/dp_actor.py` for detailed implementation. veRL also provides an explanation in [FAQ](https://verl.readthedocs.io/en/latest/faq/faq.html#what-is-the-meaning-of-train-batch-size-mini-batch-size-and-micro-batch-size).


## Part 2: Common Errors

**Error:**
```bash
File ".../flash_attn/flash_attn_interface.py", line 15, in ‹module>
    import flash_attn_2_cuda as flash_attn_gpu
ImportError: ...
```

**A:** The `flash-attn` module is not properly installed. Try to fix it by running `pip install flash-attn==2.8.1` or `pip install flash-attn==2.8.1 -v --no-build-isolation`.

---

**Error:**
```bash
UsageError: api_key not configured (no-tty). call wandb.login(key=[your_api_key]) ...
```

**A:** Try to log in to WandB before starting Ray and running the experiment. One way to do this is run the command `export WANDB_API_KEY=[your_api_key]`. You may also try using other monitors instead of WandB by setting `monitor.monitor_type=tensorboard/mlflow`.

---

**Error:**
```bash
ValueError: Failed to look up actor with name 'explorer' ...
```

**A:** Make sure Ray is started before running the experiment. If Ray is already running, you can restart it with the following commands:

```bash
ray stop
ray start --head
```

---

**Error:** Out-of-Memory (OOM) error

**A:** The following parameters may be helpful:

- For trainer, adjust `trainer.max_token_len_per_gpu` when `trainer.use_dynamic_bsz=false`; adjust `trainer.ppo_max_token_len_per_gpu` and `trainer.ulysses_sequence_parallel_size` when `trainer.use_dynamic_bsz=true`. Setting `trainer.trainer_config.actor_rollout_ref.actor.entropy_from_logits_with_chunking=true` may also help.
- For explorer, adjust `explorer.rollout_model.tensor_parallel_size`.

Besides, Trinity-RFT provides [GPU related configuration guide](https://agentscope-ai.github.io/Trinity-RFT/en/main/tutorial/trinity_gpu_configs.html), which you may refer to for suggestions on adjusting the configurations.

## Part 3: Debugging Methods

**Q:** How to find logs for debugging?

**A:** Trinity-RFT supports the actor-level logs, which automatically saves the logs for each actor (such as explorer and trainer) to `<checkpoint_job_dir>/log/<actor_name>`.

Some important logs are:

- `<checkpoint_job_dir>/log/explorer.log`: Logs generated by the explorer process. It contains:
    - The begin and end time of each explorer step.
    - The metrics generated at each explorer step and evaluation step.
    - Model weight synchronization status from the explorer side.
    - Workflow exceptions (if any). The workflow running logs are not included here, please check `<checkpoint_job_dir>/log/explorer_runner_<n>.log` for details.

- `<checkpoint_job_dir>/log/trainer.log`: Logs generated by the trainer process. It contains:
    - The begin and end time of each training iteration.
    - The metrics generated at each training iteration and evaluation iteration.
    - Model weight synchronization status from the trainer side.

- `<checkpoint_job_dir>/log/explorer_runner_<n>.log`: Logs generated by workflow runner process. It contains:
    - The logs printed by each workflow. (Must use the `self.logger` in Workflow to print logs.)
    - Exceptions occurred during the workflow running (if any).

To see more detailed logs, change the default log level (`info`) to `debug`, by setting `log.level=debug` in config file.

Alternatively, if you want to look at the full logs of all processes and save it to `debug.log`:

```bash
export RAY_DEDUP_LOGS=0
trinity run --config grpo_gsm8k/gsm8k.yaml 2>&1 | tee debug.log
```

---

**Q:** How to debug a workflow without running a full experiment?

**A**: To debug a workflow, use Trinity-RFT's debug mode with the following steps:

1. Launch the inference model via `trinity debug --config <config_file_path> --module inference_model`

2. Debug the workflow in another terminal via `trinity debug --config <config_file_path> --module workflow --output-file <output_file_path> --plugin-dir <plugin_dir>`

Please refer to {ref}`Workflow Development Guide <Workflows>` section for details.


## Part 4: Other Questions
**Q:** What's the purpose of `buffer.trainer_input.experience_buffer.path`?

**A:** This path specifies the path to the SQLite database storaging the generated experiences. You may comment out this line if you don't want to use the SQLite database.

To see the experiences in the database, you can use the following Python script:

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

# Print the experiences
for exp in exp_list:
    print(f"{exp.prompt_text=}", f"{exp.response_text=}")
```

---

**Q:** How to load the checkpoints outside of the Trinity-RFT framework?

**A:** Currently, two loading methods are supported:

1. **Recommended approach**: Use the `trinity convert` command to convert the original checkpoint into the standard Hugging Face format.
   After conversion, you can load and use it directly just like any ordinary Hugging Face model.
   For detailed instructions, please refer to the tutorial: [Optional: Converting Checkpoints to Hugging Face Format](https://agentscope-ai.github.io/Trinity-RFT/zh/main/tutorial/example_reasoning_basic.html#optional-convert-checkpoints-to-hugging-face-format)

2. **Direct loading (for actor checkpoints trained with FSDP)**:
   If you prefer to load the checkpoint directly without converting its format, you can use the following code example:

```python
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from trinity.common.models.utils import load_fsdp_state_dict_from_verl_checkpoint

# Assume we need the checkpoint at step 780;
# model_path, checkpoint_root_dir, project, and name are already defined
model = AutoModelForCausalLM.from_pretrained(model_path)
ckp_path = os.path.join(checkpoint_root_dir, project, name, "global_step_780", "actor")
model.load_state_dict(load_fsdp_state_dict_from_verl_checkpoint(ckp_path))
```

---

**Q:** What's the difference between Trinity-RFT and veRL?

**A:** Trinity-RFT uses veRL as the trainer backend, and extends it with a more modular and flexible architecture. The main differences include:

- **Modular Algorithm Module**: Trinity-RFT extracts algorithm-related components (e.g., advantage function, loss function) from veRL's `core_algos.py` into independent modules, allowing users to easily implement and register new algorithms without modifying the core codebase.
- **Separation of Explorer and Trainer**: Trinity-RFT replaces the rollout model in veRL with a separate Explorer module, which handles agent-environment interactions. This separation allows for more flexible workflow designs and rollout-training scheduling.
- **Full-lifecycle Data Pipeline**: Trinity-RFT adds a Buffer module between Explorer and Trainer, providing a complete data pipeline for experience storage, processing, and sampling. This design enables advanced data handling strategies, such as experience replay and prioritized sampling.

We also provide benchmarks comparing Trinity-RFT with veRL and systems built on veRL (e.g., [rLLM](https://github.com/rllm-org/rllm)), which show comparable or better performance and efficiency. Please refer to [Benchmark](https://github.com/agentscope-ai/Trinity-RFT/tree/main/benchmark) for more details.
