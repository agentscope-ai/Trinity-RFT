# Example: GRPO on InterCode-SQL

This example shows how to run GRPO on the [InterCode-SQL](https://github.com/princeton-nlp/intercode) benchmark.

InterCode-SQL is an interactive SQL environment. The model issues SQL commands, observes execution feedback, and uses `submit` when the latest query result is the final answer.

The config file is located in [`intercode_sql.yaml`](intercode_sql.yaml).

## Setup

Install InterCode and make sure Docker is running:

```bash
pip install intercode-bench
```

## Prepare Data

Generate the Trinity taskset from the official InterCode-SQL Spider dev task file:

```bash
python examples/grpo_intercode_sql/get_intercode_sql_data.py
```

This downloads the raw task file and creates:

- `examples/grpo_intercode_sql/intercode_sql_data/raw/ic_spider_dev.json`
- `examples/grpo_intercode_sql/intercode_sql_data/train.jsonl`
- `examples/grpo_intercode_sql/intercode_sql_data/test.jsonl`

Build or start the InterCode SQL Docker environment:

```bash
python -c "from intercode.assets import sql_build_docker; sql_build_docker()"
```

By default, the workflow connects to the `docker-env-sql` image name used by
InterCode.

The generated Trinity taskset contains 200 test tasks sampled from
`ic_spider_dev.json` by default. The remaining tasks are used for train.

You can override the test size and the seed:

```bash
python examples/grpo_intercode_sql/get_intercode_sql_data.py \
  --test_size 100 \
  --seed 42
```

## Run

```bash
trinity run --config examples/grpo_intercode_sql/intercode_sql.yaml
```

Useful environment variables:

- `TRINITY_MODEL_PATH`: rollout model path
- `TRINITY_CHECKPOINT_ROOT_DIR`: checkpoint directory

## Workflow

The example uses the registered InterCode-SQL workflow:

```yaml
default_workflow_type: 'intercode_sql_workflow'
```

The model should answer with exactly one action per turn:

```text
<think>Reason about the database question.</think><action>SELECT ...;</action>
```

When the latest SQL result answers the question, the model should call:

```text
<think>The query result is final.</think><action>submit</action>
```
