#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

if ! init_docker_compose; then
    exit 1
fi

echo "Starting Trinity Docker services with 2 GPUs per container and 64G shm."
"${COMPOSE_CMD[@]}" up -d

echo "Docker services started. Run 'bash docker/status.sh' to check Ray status."
