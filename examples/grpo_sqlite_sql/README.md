# Example: Multi-turn GRPO for Text-to-SQL on SQLite (BIRD)

This example trains an agent that explores a real SQLite database with read-only queries and then
submits one SQL query, using the [BIRD](https://bird-bench.github.io/) benchmark. It complements the
[InterCode-SQL example](../grpo_intercode_sql/README.md): it needs **no Docker or MySQL server**,
because each rollout opens the `.sqlite` file directly.

## How the workflow works

`sqlite_sql_workflow` (`trinity/common/workflows/envs/sqlite_sql/`) gives the model the database
schema, the question and BIRD's external knowledge. At each turn the model replies with exactly one
of:

- `<action>SQL</action>`: run a read-only query; the first `max_obs_rows` rows (or the error) come back
  as the next user message
- `<answer>SQL</answer>`: submit the final query and end the episode

The reward is BIRD / Spider execution accuracy: 1.0 if the submitted query returns the same set of
rows as the reference query, else 0.0. Episodes that run out of turns score 0 unless
`auto_submit: true`, which grades the last query that ran successfully.

Because the reward comes from executing model-written SQL, every query runs in a sandbox: the
database is opened read-only behind an SQLite authorizer that only allows reads (no writes, `ATTACH`,
`PRAGMA` or extensions), one statement is executed per query, and a query is interrupted after
`timeout` seconds. Concurrent rollouts therefore cannot change the data other rollouts are scored
on. Experience `info` also records `gold_error` (the reference query fails) and `gold_empty` (the
reference returns no rows, so any query returning nothing would match), so these tasks can be
filtered.

## Prepare data

1. Download the BIRD train set and databases from <https://bird-bench.github.io/>, and BIRD
   Mini-Dev (500 SQLite questions, disjoint from train) from
   <https://bird-bench.oss-cn-beijing.aliyuncs.com/minidev.zip>.
2. Create the taskset:

```bash
python examples/grpo_sqlite_sql/get_bird_data.py \
    --train_json train/train.json \
    --test_json minidev/MINIDEV/mini_dev_sqlite.json
```

Each line of `bird_data/train.jsonl` and `bird_data/test.jsonl` has `db_id`, `question`, `evidence`
and `SQL`. Any dataset in this format with `<db_root>/<db_id>/<db_id>.sqlite` databases (e.g. Spider)
works the same way.

## Run

```bash
export BIRD_TRAIN_DB_ROOT=/path/to/train/train_databases
export BIRD_DEV_DB_ROOT=/path/to/minidev/MINIDEV/dev_databases
trinity run --config examples/grpo_sqlite_sql/sqlite_sql.yaml
```

Workflow arguments (`workflow_args`):

| Argument | Default | Description |
|---|---|---|
| `db_root` | required | Directory with `<db_id>/<db_id>.sqlite` |
| `max_env_steps` | 5 | Maximum number of model turns |
| `max_obs_rows` | 10 | Rows of a query result shown to the model |
| `timeout` | 30 | Per-query timeout in seconds (the official BIRD evaluator uses 30) |
| `auto_submit` | false | Grade the last successful query if the model never submits |

Notes on BIRD Mini-Dev as a reward source:

- 16 of the 500 reference queries return the single value `1`, so a constant `SELECT 1` earns the
  full reward on 3.2% of the tasks. Watch rollouts for constant answers on hard tasks.
- Two reference queries (question ids 518 and 701) take longer than 30 s on a typical machine,
  so they score 0 for every model at the default timeout, as they do in the official evaluator.
