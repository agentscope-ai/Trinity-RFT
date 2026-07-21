# RLCR reward on math tasks

This example wires the five-component RLCR reward into Trinity's standard
`math_workflow`. The dataset fields are explicit: `prompt` supplies the user
prompt and `solution` supplies the ground-truth response passed to the reward as
`truth`. The reward extracts only the final `<answer>` payload before parsing and
verification.

Set the model and checkpoint paths as needed, then launch the config with the
usual Trinity entry point:

```bash
TRINITY_MODEL_PATH=/path/to/model \
trinity run --config examples/rlcr_math/rlcr_math.yaml
```

The system prompt requires this case-sensitive terminal sequence:

```text
<think>...</think><answer>...</answer><analysis>...</analysis><confidence>q</confidence>
```

Only whitespace may follow `</confidence>`, and `q` must be finite and in
`[0, 1]`. `rlcr_reward` returns five already weighted floats because the standard
workflow sums reward-dictionary values without applying per-component weights.

The weights in `rlcr_math.yaml` are runnable configuration defaults, not frozen
research parameters. Any study using this example should preregister its own
weights and other training choices before launch.
