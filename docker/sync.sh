#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/common.sh"

print_help() {
    cat <<EOF
Usage: bash docker/sync.sh [options]

Sync local project files to the remote workspace via rsync over SSH.

Options:
  -n, --dry-run    Show what would be transferred without making changes
  -p, --port <n>   Override SSH port (default: from remote.env or 22)
  -h, --help       Show this help message and exit

Examples:
  bash docker/sync.sh
  bash docker/sync.sh --dry-run
  bash docker/sync.sh --port 2222
EOF
}

fail() {
    echo "$1" >&2
    exit 1
}

dry_run=false
port_override=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--dry-run)
            dry_run=true
            shift
            ;;
        -p|--port)
            if [[ -z "${2:-}" ]]; then
                fail "Missing value for $1. Use --help for usage information."
            fi
            port_override="$2"
            shift 2
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
        -*)
            fail "Unknown option: $1. Use --help for usage information."
            ;;
        *)
            fail "Unexpected positional argument: $1. Use --help for usage information."
            ;;
    esac
done

if ! load_remote_env; then
    exit 1
fi

ssh_port="${port_override:-$TRINITY_REMOTE_SSH_PORT}"

rsync_args=(
    -az
    --itemize-changes
    --from0
    -e "ssh -p ${ssh_port} -o StrictHostKeyChecking=accept-new"
)

if [[ "$dry_run" == true ]]; then
    rsync_args+=(--dry-run)
    echo "Dry-run mode: showing what would be synced."
fi

untracked="$(git -C "$PROJECT_DIR" ls-files --others --exclude-standard)"
if [[ -n "$untracked" ]]; then
    echo "WARNING: The following files are untracked and will NOT be synced:" >&2
    echo "$untracked" | sed 's/^/  /' >&2
    echo "Run 'git add <file>' to include them." >&2
    echo "" >&2
fi

# Write file list to a temp file to avoid the "Bad file descriptor" race
# condition that occurs when rsync reads --files-from stdin via a pipe.
tmpfile="$(mktemp -t trinity-sync-XXXXXX)"
trap 'rm -f "$tmpfile"' EXIT
git -C "$PROJECT_DIR" ls-files -z > "$tmpfile"

dest="${TRINITY_REMOTE_HOST}:${TRINITY_REMOTE_WORKSPACE}/"
echo "Syncing git-tracked files: ${PROJECT_DIR}/ -> ${dest}"
rsync "${rsync_args[@]}" --files-from="$tmpfile" "${PROJECT_DIR}/" "$dest"
