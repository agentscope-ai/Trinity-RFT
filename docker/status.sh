#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
SERVICES=(trinity-node-1 trinity-node-2)

if ! init_docker_compose; then
    exit 1
fi

check_service() {
    local service="$1"
    local container_id
    local running_id
    local ray_output

    container_id="$("${COMPOSE_CMD[@]}" ps -a -q "$service" 2>/dev/null)"
    if [[ -z "$container_id" ]]; then
        echo "[$service] Container does not exist. Run 'bash docker/start.sh' first."
        return
    fi

    running_id="$("${COMPOSE_CMD[@]}" ps -q "$service" 2>/dev/null)"
    if [[ -z "$running_id" ]]; then
        echo "[$service] Container exists but is not running. Start it before checking Ray status."
        return
    fi

    if ray_output="$("${COMPOSE_CMD[@]}" exec -T "$service" bash -c 'source /opt/venv/bin/activate && ray status' 2>&1)"; then
        echo "[$service] Container is running and Ray is healthy."
        echo "$ray_output"
    else
        echo "[$service] Container is running, but Ray does not appear to be started or healthy."
        echo "$ray_output"
    fi
}

for service in "${SERVICES[@]}"; do
    check_service "$service"
    echo
done
