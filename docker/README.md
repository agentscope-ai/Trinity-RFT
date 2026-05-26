# Docker Test Environment

This document focuses on two things:

1. how to prepare the Docker-based test environment
2. how to run Trinity unit tests inside that environment

## Hardware Requirements

- `trinity-node-1` requires at least 2 GPUs.
- `trinity-node-2` requires at least 2 GPUs.
- Each container requires `64G` shared memory.

Default GPU assignment:

- `trinity-node-1`: GPU `0` and GPU `1`
- `trinity-node-2`: GPU `2` and GPU `3`

If the current machine uses different GPU indices, override them in `docker/env`.

## Test Environment Setup

Before running any Docker test command:

1. Copy `docker/env.example` to `docker/env`.
2. Update `docker/env` for the current machine.
3. Make sure `TRINITY_MOUNT_DIR` points to a host directory that contains the required models, datasets, and checkpoints.

The helper scripts do not read `env.example` automatically. If `docker/env` is missing, they will stop and ask you to create it first.

Required settings in `docker/env`:

- `TRINITY_DOCKER_IMAGE`: Docker image used by both containers.
- `TRINITY_MOUNT_DIR`: Host directory mounted to `/mnt` inside the containers.
- `TRINITY_NODE1_GPU_0`, `TRINITY_NODE1_GPU_1`: GPU indices for `trinity-node-1`.
- `TRINITY_NODE2_GPU_0`, `TRINITY_NODE2_GPU_1`: GPU indices for `trinity-node-2`.
- `TRINITY_HF_ENDPOINT`: Hugging Face mirror or endpoint.
- `TRINITY_PYPI_INDEX_URL`: Python package index used inside containers.
- `TRINITY_RAY_DASHBOARD_PORT`: Host port mapped to the Ray dashboard.

## Start And Check The Environment

Start the Docker test environment:

```bash
bash docker/start.sh
```

Check whether both containers are up and whether Ray is healthy:

```bash
bash docker/status.sh
```

Expected interpretation:

- If a container does not exist, run `bash docker/start.sh` first.
- If a container exists but is stopped, start the environment again before running tests.
- If a container is running but Ray is unhealthy, resolve the container startup problem before running tests.

## Run Tests

Use `bash docker/run.sh` to execute pytest inside `trinity-node-1`.

Run one narrow test module:

```bash
bash docker/run.sh --module common
```

Run a filtered subset when a smaller slice is known:

```bash
bash docker/run.sh --module common --keyword test_config
```

Rules for test execution:

- Always prefer the smallest viable `--module`.
- Add `--keyword` whenever you know the failing test name, keyword, or a smaller slice.
- Do not widen the test scope unless the narrower check is insufficient.
- `run.sh` always executes tests inside `trinity-node-1`.

## Stop The Environment

Stop the Docker test environment after use:

```bash
bash docker/stop.sh
```

## Script Roles

- [start.sh](/nas/pxc/rft/Trinity-RFT/docker/start.sh): starts the test environment.
- [status.sh](/nas/pxc/rft/Trinity-RFT/docker/status.sh): checks container state and Ray health.
- [run.sh](/nas/pxc/rft/Trinity-RFT/docker/run.sh): runs pytest in the Docker test environment.
- [stop.sh](/nas/pxc/rft/Trinity-RFT/docker/stop.sh): shuts the test environment down cleanly.
