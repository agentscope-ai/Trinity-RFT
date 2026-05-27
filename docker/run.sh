#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
DEFAULT_MODULE="common"
SERVICE_NAME="trinity-node-1"

print_help() {
    cat <<EOF
Usage: bash docker/run.sh [options]

Run pytest inside a Trinity Docker container.

Options:
  -m, --module <name>    Test module under tests/. Default: ${DEFAULT_MODULE}
  -k, --keyword <expr>   Pytest -k expression used to filter tests
  -h, --help             Show this help message and exit

Examples:
  bash docker/run.sh
  bash docker/run.sh --module buffer
  bash docker/run.sh --module common --keyword test_config
EOF
}

fail() {
    echo "$1" >&2
    exit 1
}

require_arg() {
    local option="$1"
    local value="$2"

    if [[ -z "$value" ]]; then
        fail "Missing value for ${option}. Use --help for usage information."
    fi
}

module_name="$DEFAULT_MODULE"
keyword_expr=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--module)
            require_arg "$1" "$2"
            module_name="$2"
            shift 2
            ;;
        -k|--keyword)
            require_arg "$1" "$2"
            keyword_expr="$2"
            shift 2
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
        --)
            shift
            break
            ;;
        -*)
            fail "Unknown option: $1. Use --help for usage information."
            ;;
        *)
            fail "Unexpected positional argument: $1. Use --help for usage information."
            ;;
    esac
done

if [[ $# -gt 0 ]]; then
    fail "Unexpected positional arguments: $*. Use --help for usage information."
fi

if ! init_docker_compose; then
    exit 1
fi

if [[ ! -d "$SCRIPT_DIR/../tests/${module_name}" ]]; then
    fail "Test module 'tests/${module_name}' does not exist."
fi

container_id="$("${COMPOSE_CMD[@]}" ps -a -q "$SERVICE_NAME" 2>/dev/null)"
if [[ -z "$container_id" ]]; then
    fail "Container '${SERVICE_NAME}' does not exist. Run 'bash docker/start.sh' first."
fi

running_id="$("${COMPOSE_CMD[@]}" ps -q "$SERVICE_NAME" 2>/dev/null)"
if [[ -z "$running_id" ]]; then
    fail "Container '${SERVICE_NAME}' exists but is not running. Start it before running tests."
fi

pytest_args=(pytest "tests/${module_name}" -v -s)
if [[ -n "$keyword_expr" ]]; then
    pytest_args+=(-k "$keyword_expr")
fi

printf -v pytest_cmd '%q ' "${pytest_args[@]}"
pytest_cmd="source /opt/venv/bin/activate && ${pytest_cmd% }"

echo "Running tests in ${SERVICE_NAME}: ${pytest_cmd}"
"${COMPOSE_CMD[@]}" exec "$SERVICE_NAME" bash -c "$pytest_cmd"
