#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yaml"
ENV_EXAMPLE_FILE="$SCRIPT_DIR/env.example"
ENV_FILE="$SCRIPT_DIR/env"
COMPOSE_CMD=()

docker_fail() {
    echo "$1" >&2
    return 1
}

load_docker_env() {
    if [[ ! -f "$ENV_FILE" ]]; then
        if [[ -f "$ENV_EXAMPLE_FILE" ]]; then
            docker_fail "docker/env was not found. Copy docker/env.example to docker/env and adjust the machine-specific settings first."
        else
            docker_fail "docker/env was not found, and docker/env.example is also missing in $SCRIPT_DIR."
        fi
        return 1
    fi

    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
}

init_docker_compose() {
    if ! command -v docker >/dev/null 2>&1; then
        docker_fail "Docker is not installed or not available in PATH."
        return 1
    fi

    if [[ ! -f "$COMPOSE_FILE" ]]; then
        docker_fail "docker-compose.yaml was not found in $SCRIPT_DIR."
        return 1
    fi

    load_docker_env || return 1

    COMPOSE_CMD=(docker compose -f "$COMPOSE_FILE")
    if ! "${COMPOSE_CMD[@]}" version >/dev/null 2>&1; then
        docker_fail "Docker Compose is not available. Make sure 'docker compose' works on this machine."
        return 1
    fi

    for required_var in \
        TRINITY_DOCKER_IMAGE \
        TRINITY_PYPI_INDEX_URL \
        TRINITY_HF_ENDPOINT \
        TRINITY_MOUNT_DIR \
        TRINITY_RAY_DASHBOARD_PORT \
        TRINITY_NODE1_GPU_0 \
        TRINITY_NODE1_GPU_1 \
        TRINITY_NODE2_GPU_0 \
        TRINITY_NODE2_GPU_1; do
        if [[ -z "${!required_var:-}" ]]; then
            docker_fail "Required Docker setting '$required_var' is empty. Check docker/env."
            return 1
        fi
    done
}
