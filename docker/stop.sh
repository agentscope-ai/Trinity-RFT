#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

if ! init_docker_compose; then
    exit 1
fi

echo "Stopping Trinity Docker services."
"${COMPOSE_CMD[@]}" down

echo "Docker services stopped."
