# -*- coding: utf-8 -*-
"""
Task generation for the simulated terminal environment.

Every task is fully deterministic given a seed: the seed controls the OS type,
task type, file names, paths, remote server details, and filesystem layout.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from trinity.common.workflows.connect_the_dots.terminal.env import (
    FileMeta,
    MachineState,
    OSType,
    TerminalEnv,
    VirtualFS,
)

# ---------------------------------------------------------------------------
# Task types
# ---------------------------------------------------------------------------


class TaskType(Enum):
    UPLOAD = "upload"
    DOWNLOAD = "download"
    RENAME = "rename"
    MOVE = "move"
    CHMOD = "chmod"
    DELETE = "delete"
    COPY = "copy"
    PACK = "pack"
    MKDIR = "mkdir"


class CompositeTemplate(Enum):
    PACK_UPLOAD = "pack_upload"
    DOWNLOAD_EXTRACT = "download_extract"
    MKDIR_UPLOAD = "mkdir_upload"
    UPLOAD_CHMOD = "upload_chmod"
    UPLOAD_DELETE_SOURCE = "upload_delete_source"
    PACK_UPLOAD_EXTRACT = "pack_upload_extract"
    DOWNLOAD_RENAME = "download_rename"
    BACKUP_REPLACE = "backup_replace"


# ---------------------------------------------------------------------------
# Pools for randomisation
# ---------------------------------------------------------------------------

FILENAMES = [
    "report.txt", "data.csv", "image.png", "config.json", "notes.md",
    "script.sh", "log.txt", "readme.txt", "database.db", "output.log",
    "results.csv", "document.pdf", "settings.ini", "main.py", "index.html",
    "style.css", "app.js", "server.py", "deploy.sh", "backup.sql",
    "photo.jpg", "diagram.svg", "metrics.json", "requirements.txt", "Makefile",
]

DIRNAMES = [
    "projects", "documents", "downloads", "backups", "config",
    "data", "logs", "workspace", "reports", "media",
    "scripts", "output", "staging", "archive", "temp",
    "src", "build", "dist", "assets", "uploads",
]

REMOTE_USERS = ["admin", "deploy", "user", "webmaster", "devops", "ops", "ubuntu"]

PERMISSION_MODES = ["644", "755", "600", "700", "664", "640", "444", "750"]

ARCHIVE_FORMATS = ["tar", "tar.gz", "zip"]


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class TerminalTask:
    """Complete specification for one terminal task."""
    seed: int
    local_os: OSType
    task_category: str              # "single" or "composite"
    task_type: str                  # TaskType value or CompositeTemplate value
    remote_host: str
    remote_user: str
    description: str                # human-readable task description
    # goal state as a list of checks
    goal_checks: List[Dict[str, Any]] = field(default_factory=list)
    # initial filesystem snapshots (for building MachineState at runtime)
    local_fs_spec: List[Dict[str, Any]] = field(default_factory=list)
    remote_fs_spec: List[Dict[str, Any]] = field(default_factory=list)
    # extra metadata
    params: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------


def _home_dir(os_type: OSType) -> str:
    """Canonical memfs home directory."""
    if os_type == OSType.WINDOWS:
        return "C/Users/user"
    elif os_type == OSType.MAC:
        return "Users/user"
    return "home/user"


def _display_path(memfs_path: str, os_type: OSType) -> str:
    """Convert memfs path to OS-appropriate display path."""
    if os_type == OSType.WINDOWS:
        parts = memfs_path.split("/")
        if parts and len(parts[0]) == 1 and parts[0].isalpha():
            drive = parts[0].upper()
            rest = "\\".join(parts[1:])
            return f"{drive}:\\{rest}" if rest else f"{drive}:\\"
        return "\\".join(parts)
    return "/" + memfs_path if memfs_path else "/"


def _make_file_content(filename: str, seed: int) -> str:
    """Deterministic file content for a given filename and seed."""
    return f"[Content of {filename} | seed={seed} | size=1024]"


def _pick_subdir(rng: random.Random, os_type: OSType) -> str:
    """Pick a random subdirectory under home."""
    home = _home_dir(os_type)
    subdir = rng.choice(DIRNAMES)
    return f"{home}/{subdir}"


def _pick_remote_dir(rng: random.Random, remote_user: str) -> str:
    """Pick a random directory on the remote machine."""
    options = [
        f"home/{remote_user}/{rng.choice(DIRNAMES)}",
        f"var/www/html",
        f"opt/{rng.choice(DIRNAMES)}",
        f"home/{remote_user}",
    ]
    return rng.choice(options)


def _add_distractor_files(
    fs_spec: list, base_dir: str, rng: random.Random, seed: int, count: int = 3
):
    """Add some irrelevant files to make the task more realistic."""
    used = {f["path"] for f in fs_spec}
    for _ in range(count):
        name = rng.choice(FILENAMES)
        path = f"{base_dir}/{name}"
        if path not in used:
            content = _make_file_content(name, seed + hash(name) % 10000)
            fs_spec.append({"path": path, "content": content})
            used.add(path)


# ---------------------------------------------------------------------------
# Single task generators
# ---------------------------------------------------------------------------


def _gen_upload(rng: random.Random, seed: int, os_type: OSType,
               remote_host: str, remote_user: str) -> TerminalTask:
    filename = rng.choice(FILENAMES)
    local_dir = _pick_subdir(rng, os_type)
    remote_dir = _pick_remote_dir(rng, remote_user)
    content = _make_file_content(filename, seed)

    local_fs = [{"path": f"{local_dir}/{filename}", "content": content}]
    _add_distractor_files(local_fs, local_dir, rng, seed)
    remote_fs = [{"path": remote_dir, "is_dir": True}]

    local_display = _display_path(f"{local_dir}/{filename}", os_type)
    remote_display = f"{remote_user}@{remote_host}:/{remote_dir}/{filename}"

    return TerminalTask(
        seed=seed, local_os=os_type, task_category="single",
        task_type=TaskType.UPLOAD.value,
        remote_host=remote_host, remote_user=remote_user,
        description=f"Upload '{local_display}' to {remote_display}",
        goal_checks=[
            {"type": "file_exists", "machine": "remote",
             "path": f"{remote_dir}/{filename}", "content": content},
        ],
        local_fs_spec=local_fs, remote_fs_spec=remote_fs,
        params={"filename": filename, "local_dir": local_dir, "remote_dir": remote_dir},
    )


def _gen_download(rng: random.Random, seed: int, os_type: OSType,
                  remote_host: str, remote_user: str) -> TerminalTask:
    filename = rng.choice(FILENAMES)
    local_dir = _pick_subdir(rng, os_type)
    remote_dir = _pick_remote_dir(rng, remote_user)
    content = _make_file_content(filename, seed)

    local_fs = [{"path": local_dir, "is_dir": True}]
    _add_distractor_files(local_fs, local_dir, rng, seed)
    remote_fs = [{"path": f"{remote_dir}/{filename}", "content": content}]
    _add_distractor_files(remote_fs, remote_dir, rng, seed)

    local_display = _display_path(f"{local_dir}/{filename}", os_type)
    remote_display = f"{remote_user}@{remote_host}:/{remote_dir}/{filename}"

    return TerminalTask(
        seed=seed, local_os=os_type, task_category="single",
        task_type=TaskType.DOWNLOAD.value,
        remote_host=remote_host, remote_user=remote_user,
        description=f"Download '{remote_display}' to '{local_display}'",
        goal_checks=[
            {"type": "file_exists", "machine": "local",
             "path": f"{local_dir}/{filename}", "content": content},
        ],
        local_fs_spec=local_fs, remote_fs_spec=remote_fs,
        params={"filename": filename, "local_dir": local_dir, "remote_dir": remote_dir},
    )


def _gen_rename(rng: random.Random, seed: int, os_type: OSType,
                remote_host: str, remote_user: str) -> TerminalTask:
    is_remote = rng.choice([True, False])
    old_name = rng.choice(FILENAMES)
    # Generate a different new name
    new_name = rng.choice(FILENAMES)
    while new_name == old_name:
        new_name = rng.choice(FILENAMES)
    content = _make_file_content(old_name, seed)

    if is_remote:
        target_dir = _pick_remote_dir(rng, remote_user)
        fs_spec = [{"path": f"{target_dir}/{old_name}", "content": content}]
        _add_distractor_files(fs_spec, target_dir, rng, seed)
        machine_label = "remote"
        desc_path = f"{remote_user}@{remote_host}:/{target_dir}"
    else:
        target_dir = _pick_subdir(rng, os_type)
        fs_spec = [{"path": f"{target_dir}/{old_name}", "content": content}]
        _add_distractor_files(fs_spec, target_dir, rng, seed)
        machine_label = "local"
        desc_path = _display_path(target_dir, os_type)

    local_fs = fs_spec if not is_remote else []
    remote_fs = fs_spec if is_remote else []

    return TerminalTask(
        seed=seed, local_os=os_type, task_category="single",
        task_type=TaskType.RENAME.value,
        remote_host=remote_host, remote_user=remote_user,
        description=(f"Rename '{old_name}' to '{new_name}' in "
                     f"{'remote ' if is_remote else ''}{desc_path}"),
        goal_checks=[
            {"type": "file_not_exists", "machine": machine_label,
             "path": f"{target_dir}/{old_name}"},
            {"type": "file_exists", "machine": machine_label,
             "path": f"{target_dir}/{new_name}", "content": content},
        ],
        local_fs_spec=local_fs, remote_fs_spec=remote_fs,
        params={"is_remote": is_remote, "old_name": old_name, "new_name": new_name,
                "target_dir": target_dir},
    )


def _gen_move(rng: random.Random, seed: int, os_type: OSType,
              remote_host: str, remote_user: str) -> TerminalTask:
    is_remote = rng.choice([True, False])
    filename = rng.choice(FILENAMES)
    content = _make_file_content(filename, seed)

    if is_remote:
        src_dir = _pick_remote_dir(rng, remote_user)
        dst_dir = _pick_remote_dir(rng, remote_user)
        while dst_dir == src_dir:
            dst_dir = _pick_remote_dir(rng, remote_user)
        fs_spec = [
            {"path": f"{src_dir}/{filename}", "content": content},
            {"path": dst_dir, "is_dir": True},
        ]
        _add_distractor_files(fs_spec, src_dir, rng, seed)
        machine_label = "remote"
        src_display = f"/{src_dir}/{filename}"
        dst_display = f"/{dst_dir}/"
    else:
        src_dir = _pick_subdir(rng, os_type)
        dst_dir = _pick_subdir(rng, os_type)
        while dst_dir == src_dir:
            dst_dir = _pick_subdir(rng, os_type)
        fs_spec = [
            {"path": f"{src_dir}/{filename}", "content": content},
            {"path": dst_dir, "is_dir": True},
        ]
        _add_distractor_files(fs_spec, src_dir, rng, seed)
        machine_label = "local"
        src_display = _display_path(f"{src_dir}/{filename}", os_type)
        dst_display = _display_path(dst_dir, os_type)

    local_fs = fs_spec if not is_remote else []
    remote_fs = fs_spec if is_remote else []

    return TerminalTask(
        seed=seed, local_os=os_type, task_category="single",
        task_type=TaskType.MOVE.value,
        remote_host=remote_host, remote_user=remote_user,
        description=(f"Move '{src_display}' to '{dst_display}'"
                     f"{' on remote server' if is_remote else ''}"),
        goal_checks=[
            {"type": "file_not_exists", "machine": machine_label,
             "path": f"{src_dir}/{filename}"},
            {"type": "file_exists", "machine": machine_label,
             "path": f"{dst_dir}/{filename}", "content": content},
        ],
        local_fs_spec=local_fs, remote_fs_spec=remote_fs,
        params={"is_remote": is_remote, "filename": filename,
                "src_dir": src_dir, "dst_dir": dst_dir},
    )


def _gen_chmod(rng: random.Random, seed: int, os_type: OSType,
               remote_host: str, remote_user: str) -> TerminalTask:
    # chmod only works on Linux/Mac or remote
    is_remote = rng.choice([True, False])
    if not is_remote and os_type == OSType.WINDOWS:
        is_remote = True  # force remote for Windows local

    filename = rng.choice(FILENAMES)
    content = _make_file_content(filename, seed)
    old_perm = rng.choice(PERMISSION_MODES)
    new_perm = rng.choice(PERMISSION_MODES)
    while new_perm == old_perm:
        new_perm = rng.choice(PERMISSION_MODES)

    if is_remote:
        target_dir = _pick_remote_dir(rng, remote_user)
        remote_fs = [{"path": f"{target_dir}/{filename}", "content": content,
                      "permissions": old_perm}]
        _add_distractor_files(remote_fs, target_dir, rng, seed)
        local_fs = []
        machine_label = "remote"
        desc_path = f"{remote_user}@{remote_host}:/{target_dir}/{filename}"
    else:
        target_dir = _pick_subdir(rng, os_type)
        local_fs = [{"path": f"{target_dir}/{filename}", "content": content,
                     "permissions": old_perm}]
        _add_distractor_files(local_fs, target_dir, rng, seed)
        remote_fs = []
        machine_label = "local"
        desc_path = _display_path(f"{target_dir}/{filename}", os_type)

    return TerminalTask(
        seed=seed, local_os=os_type, task_category="single",
        task_type=TaskType.CHMOD.value,
        remote_host=remote_host, remote_user=remote_user,
        description=f"Change permissions of '{desc_path}' to {new_perm}",
        goal_checks=[
            {"type": "permission_equals", "machine": machine_label,
             "path": f"{target_dir}/{filename}", "value": new_perm},
        ],
        local_fs_spec=local_fs, remote_fs_spec=remote_fs,
        params={"is_remote": is_remote, "filename": filename,
                "target_dir": target_dir, "new_perm": new_perm},
    )


def _gen_delete(rng: random.Random, seed: int, os_type: OSType,
                remote_host: str, remote_user: str) -> TerminalTask:
    is_remote = rng.choice([True, False])
    filename = rng.choice(FILENAMES)
    content = _make_file_content(filename, seed)

    if is_remote:
        target_dir = _pick_remote_dir(rng, remote_user)
        remote_fs = [{"path": f"{target_dir}/{filename}", "content": content}]
        _add_distractor_files(remote_fs, target_dir, rng, seed)
        local_fs = []
        machine_label = "remote"
        desc_path = f"{remote_user}@{remote_host}:/{target_dir}/{filename}"
    else:
        target_dir = _pick_subdir(rng, os_type)
        local_fs = [{"path": f"{target_dir}/{filename}", "content": content}]
        _add_distractor_files(local_fs, target_dir, rng, seed)
        remote_fs = []
        machine_label = "local"
        desc_path = _display_path(f"{target_dir}/{filename}", os_type)

    return TerminalTask(
        seed=seed, local_os=os_type, task_category="single",
        task_type=TaskType.DELETE.value,
        remote_host=remote_host, remote_user=remote_user,
        description=f"Delete the file '{desc_path}'",
        goal_checks=[
            {"type": "file_not_exists", "machine": machine_label,
             "path": f"{target_dir}/{filename}"},
        ],
        local_fs_spec=local_fs, remote_fs_spec=remote_fs,
        params={"is_remote": is_remote, "filename": filename, "target_dir": target_dir},
    )


def _gen_copy(rng: random.Random, seed: int, os_type: OSType,
              remote_host: str, remote_user: str) -> TerminalTask:
    is_remote = rng.choice([True, False])
    filename = rng.choice(FILENAMES)
    content = _make_file_content(filename, seed)

    if is_remote:
        src_dir = _pick_remote_dir(rng, remote_user)
        dst_dir = _pick_remote_dir(rng, remote_user)
        while dst_dir == src_dir:
            dst_dir = _pick_remote_dir(rng, remote_user)
        fs_spec = [
            {"path": f"{src_dir}/{filename}", "content": content},
            {"path": dst_dir, "is_dir": True},
        ]
        machine_label = "remote"
        src_display = f"/{src_dir}/{filename}"
        dst_display = f"/{dst_dir}/{filename}"
    else:
        src_dir = _pick_subdir(rng, os_type)
        dst_dir = _pick_subdir(rng, os_type)
        while dst_dir == src_dir:
            dst_dir = _pick_subdir(rng, os_type)
        fs_spec = [
            {"path": f"{src_dir}/{filename}", "content": content},
            {"path": dst_dir, "is_dir": True},
        ]
        machine_label = "local"
        src_display = _display_path(f"{src_dir}/{filename}", os_type)
        dst_display = _display_path(f"{dst_dir}/{filename}", os_type)

    local_fs = fs_spec if not is_remote else []
    remote_fs = fs_spec if is_remote else []

    return TerminalTask(
        seed=seed, local_os=os_type, task_category="single",
        task_type=TaskType.COPY.value,
        remote_host=remote_host, remote_user=remote_user,
        description=(f"Copy '{src_display}' to '{dst_display}'"
                     f"{' on remote server' if is_remote else ''}"),
        goal_checks=[
            {"type": "file_exists", "machine": machine_label,
             "path": f"{src_dir}/{filename}", "content": content},
            {"type": "file_exists", "machine": machine_label,
             "path": f"{dst_dir}/{filename}", "content": content},
        ],
        local_fs_spec=local_fs, remote_fs_spec=remote_fs,
        params={"is_remote": is_remote, "filename": filename,
                "src_dir": src_dir, "dst_dir": dst_dir},
    )


def _gen_pack(rng: random.Random, seed: int, os_type: OSType,
              remote_host: str, remote_user: str) -> TerminalTask:
    is_remote = rng.choice([True, False])
    fmt = rng.choice(ARCHIVE_FORMATS)
    # On Windows local, only zip is available (no tar)
    if not is_remote and os_type == OSType.WINDOWS and fmt.startswith("tar"):
        fmt = "zip"

    dir_name = rng.choice(DIRNAMES)
    n_files = rng.randint(2, 4)
    files = rng.sample(FILENAMES, min(n_files, len(FILENAMES)))

    if fmt == "zip":
        archive_name = f"{dir_name}.zip"
    elif fmt == "tar.gz":
        archive_name = f"{dir_name}.tar.gz"
    else:
        archive_name = f"{dir_name}.tar"

    if is_remote:
        base_dir = _pick_remote_dir(rng, remote_user)
        machine_label = "remote"
    else:
        base_dir = _pick_subdir(rng, os_type)
        machine_label = "local"

    src_dir = f"{base_dir}/{dir_name}"
    archive_entries = {}
    fs_spec = []
    for f in files:
        content = _make_file_content(f, seed)
        fs_spec.append({"path": f"{src_dir}/{f}", "content": content})
        archive_entries[f"{dir_name}/{f}"] = content

    local_fs = fs_spec if not is_remote else []
    remote_fs = fs_spec if is_remote else []

    if is_remote:
        desc = f"Pack the directory '/{src_dir}' into '{archive_name}' on the remote server"
    else:
        desc = f"Pack the directory '{_display_path(src_dir, os_type)}' into '{archive_name}'"

    return TerminalTask(
        seed=seed, local_os=os_type, task_category="single",
        task_type=TaskType.PACK.value,
        remote_host=remote_host, remote_user=remote_user,
        description=desc,
        goal_checks=[
            {"type": "archive_contains", "machine": machine_label,
             "path": f"{base_dir}/{archive_name}",
             "entries": archive_entries, "archive_type": fmt},
        ],
        local_fs_spec=local_fs, remote_fs_spec=remote_fs,
        params={"is_remote": is_remote, "dir_name": dir_name, "archive_name": archive_name,
                "fmt": fmt, "base_dir": base_dir, "files": files},
    )


def _gen_mkdir(rng: random.Random, seed: int, os_type: OSType,
               remote_host: str, remote_user: str) -> TerminalTask:
    is_remote = rng.choice([True, False])
    depth = rng.randint(2, 4)
    parts = rng.sample(DIRNAMES, min(depth, len(DIRNAMES)))

    if is_remote:
        base = f"home/{remote_user}"
        machine_label = "remote"
    else:
        base = _home_dir(os_type)
        machine_label = "local"

    target_path = base + "/" + "/".join(parts)

    if is_remote:
        desc = f"Create the directory structure '/{target_path}' on the remote server"
    else:
        desc = f"Create the directory structure '{_display_path(target_path, os_type)}'"

    return TerminalTask(
        seed=seed, local_os=os_type, task_category="single",
        task_type=TaskType.MKDIR.value,
        remote_host=remote_host, remote_user=remote_user,
        description=desc,
        goal_checks=[
            {"type": "dir_exists", "machine": machine_label, "path": target_path},
        ],
        local_fs_spec=[], remote_fs_spec=[],
        params={"is_remote": is_remote, "target_path": target_path},
    )


SINGLE_GENERATORS = {
    TaskType.UPLOAD: _gen_upload,
    TaskType.DOWNLOAD: _gen_download,
    TaskType.RENAME: _gen_rename,
    TaskType.MOVE: _gen_move,
    TaskType.CHMOD: _gen_chmod,
    TaskType.DELETE: _gen_delete,
    TaskType.COPY: _gen_copy,
    TaskType.PACK: _gen_pack,
    TaskType.MKDIR: _gen_mkdir,
}


# ---------------------------------------------------------------------------
# Composite task generators
# ---------------------------------------------------------------------------


def _gen_pack_upload(rng, seed, os_type, remote_host, remote_user):
    """Pack local files into archive, then upload archive to remote."""
    fmt = rng.choice(ARCHIVE_FORMATS)
    if os_type == OSType.WINDOWS and fmt.startswith("tar"):
        fmt = "zip"

    dir_name = rng.choice(DIRNAMES)
    files = rng.sample(FILENAMES, rng.randint(2, 4))
    local_dir = _pick_subdir(rng, os_type)
    remote_dir = _pick_remote_dir(rng, remote_user)

    ext = {"tar": ".tar", "tar.gz": ".tar.gz", "zip": ".zip"}[fmt]
    archive_name = f"{dir_name}{ext}"
    src_dir = f"{local_dir}/{dir_name}"

    local_fs = []
    archive_entries = {}
    for f in files:
        content = _make_file_content(f, seed)
        local_fs.append({"path": f"{src_dir}/{f}", "content": content})
        archive_entries[f"{dir_name}/{f}"] = content

    remote_fs = [{"path": remote_dir, "is_dir": True}]

    local_display = _display_path(src_dir, os_type)
    return TerminalTask(
        seed=seed, local_os=os_type, task_category="composite",
        task_type=CompositeTemplate.PACK_UPLOAD.value,
        remote_host=remote_host, remote_user=remote_user,
        description=(f"Pack '{local_display}' into a {fmt} archive and upload it to "
                     f"{remote_user}@{remote_host}:/{remote_dir}/"),
        goal_checks=[
            {"type": "archive_contains", "machine": "remote",
             "path": f"{remote_dir}/{archive_name}",
             "entries": archive_entries, "archive_type": fmt},
        ],
        local_fs_spec=local_fs, remote_fs_spec=remote_fs,
        params={"fmt": fmt, "dir_name": dir_name, "archive_name": archive_name,
                "local_dir": local_dir, "remote_dir": remote_dir, "files": files},
    )


def _gen_download_extract(rng, seed, os_type, remote_host, remote_user):
    """Download archive from remote, extract locally."""
    fmt = rng.choice(ARCHIVE_FORMATS)
    if os_type == OSType.WINDOWS and fmt.startswith("tar"):
        fmt = "zip"

    dir_name = rng.choice(DIRNAMES)
    files = rng.sample(FILENAMES, rng.randint(2, 4))
    local_dir = _pick_subdir(rng, os_type)
    remote_dir = _pick_remote_dir(rng, remote_user)

    ext = {"tar": ".tar", "tar.gz": ".tar.gz", "zip": ".zip"}[fmt]
    archive_name = f"{dir_name}{ext}"

    archive_entries = {}
    for f in files:
        content = _make_file_content(f, seed)
        archive_entries[f"{dir_name}/{f}"] = content

    archive_content = json.dumps({"type": fmt, "entries": archive_entries})
    remote_fs = [
        {"path": f"{remote_dir}/{archive_name}", "content": archive_content,
         "archive_type": fmt, "archive_entries": archive_entries},
    ]
    local_fs = [{"path": local_dir, "is_dir": True}]

    # Goal: all files extracted locally
    checks = []
    for rel, content in archive_entries.items():
        checks.append({
            "type": "file_exists", "machine": "local",
            "path": f"{local_dir}/{rel}", "content": content,
        })

    return TerminalTask(
        seed=seed, local_os=os_type, task_category="composite",
        task_type=CompositeTemplate.DOWNLOAD_EXTRACT.value,
        remote_host=remote_host, remote_user=remote_user,
        description=(f"Download '{archive_name}' from "
                     f"{remote_user}@{remote_host}:/{remote_dir}/ "
                     f"and extract its contents to "
                     f"'{_display_path(local_dir, os_type)}'"),
        goal_checks=checks,
        local_fs_spec=local_fs, remote_fs_spec=remote_fs,
        params={"fmt": fmt, "dir_name": dir_name, "archive_name": archive_name,
                "local_dir": local_dir, "remote_dir": remote_dir},
    )


def _gen_mkdir_upload(rng, seed, os_type, remote_host, remote_user):
    """Create directory on remote, then upload file there."""
    filename = rng.choice(FILENAMES)
    content = _make_file_content(filename, seed)
    local_dir = _pick_subdir(rng, os_type)
    # Remote dir that does NOT exist yet
    depth = rng.randint(2, 3)
    parts = rng.sample(DIRNAMES, depth)
    remote_dir = f"home/{remote_user}/" + "/".join(parts)

    local_fs = [{"path": f"{local_dir}/{filename}", "content": content}]
    _add_distractor_files(local_fs, local_dir, rng, seed)
    remote_fs = []  # remote dir intentionally does not exist

    return TerminalTask(
        seed=seed, local_os=os_type, task_category="composite",
        task_type=CompositeTemplate.MKDIR_UPLOAD.value,
        remote_host=remote_host, remote_user=remote_user,
        description=(f"Create the directory '/{remote_dir}' on the remote server, "
                     f"then upload '{_display_path(local_dir + '/' + filename, os_type)}' there"),
        goal_checks=[
            {"type": "dir_exists", "machine": "remote", "path": remote_dir},
            {"type": "file_exists", "machine": "remote",
             "path": f"{remote_dir}/{filename}", "content": content},
        ],
        local_fs_spec=local_fs, remote_fs_spec=remote_fs,
        params={"filename": filename, "local_dir": local_dir, "remote_dir": remote_dir},
    )


def _gen_upload_chmod(rng, seed, os_type, remote_host, remote_user):
    """Upload file to remote, then change its permissions."""
    filename = rng.choice(FILENAMES)
    content = _make_file_content(filename, seed)
    local_dir = _pick_subdir(rng, os_type)
    remote_dir = _pick_remote_dir(rng, remote_user)
    target_perm = rng.choice(["755", "700", "600", "444"])

    local_fs = [{"path": f"{local_dir}/{filename}", "content": content}]
    remote_fs = [{"path": remote_dir, "is_dir": True}]

    return TerminalTask(
        seed=seed, local_os=os_type, task_category="composite",
        task_type=CompositeTemplate.UPLOAD_CHMOD.value,
        remote_host=remote_host, remote_user=remote_user,
        description=(f"Upload '{_display_path(local_dir + '/' + filename, os_type)}' to "
                     f"{remote_user}@{remote_host}:/{remote_dir}/ "
                     f"and set its permissions to {target_perm}"),
        goal_checks=[
            {"type": "file_exists", "machine": "remote",
             "path": f"{remote_dir}/{filename}", "content": content},
            {"type": "permission_equals", "machine": "remote",
             "path": f"{remote_dir}/{filename}", "value": target_perm},
        ],
        local_fs_spec=local_fs, remote_fs_spec=remote_fs,
        params={"filename": filename, "local_dir": local_dir,
                "remote_dir": remote_dir, "target_perm": target_perm},
    )


def _gen_upload_delete_source(rng, seed, os_type, remote_host, remote_user):
    """Upload file to remote, then delete the local copy."""
    filename = rng.choice(FILENAMES)
    content = _make_file_content(filename, seed)
    local_dir = _pick_subdir(rng, os_type)
    remote_dir = _pick_remote_dir(rng, remote_user)

    local_fs = [{"path": f"{local_dir}/{filename}", "content": content}]
    _add_distractor_files(local_fs, local_dir, rng, seed)
    remote_fs = [{"path": remote_dir, "is_dir": True}]

    return TerminalTask(
        seed=seed, local_os=os_type, task_category="composite",
        task_type=CompositeTemplate.UPLOAD_DELETE_SOURCE.value,
        remote_host=remote_host, remote_user=remote_user,
        description=(f"Upload '{_display_path(local_dir + '/' + filename, os_type)}' to "
                     f"{remote_user}@{remote_host}:/{remote_dir}/ "
                     f"and then delete the local copy"),
        goal_checks=[
            {"type": "file_exists", "machine": "remote",
             "path": f"{remote_dir}/{filename}", "content": content},
            {"type": "file_not_exists", "machine": "local",
             "path": f"{local_dir}/{filename}"},
        ],
        local_fs_spec=local_fs, remote_fs_spec=remote_fs,
        params={"filename": filename, "local_dir": local_dir, "remote_dir": remote_dir},
    )


def _gen_pack_upload_extract(rng, seed, os_type, remote_host, remote_user):
    """Pack local dir, upload, extract on remote."""
    fmt = rng.choice(ARCHIVE_FORMATS)
    if os_type == OSType.WINDOWS and fmt.startswith("tar"):
        fmt = "zip"

    dir_name = rng.choice(DIRNAMES)
    files = rng.sample(FILENAMES, rng.randint(2, 4))
    local_dir = _pick_subdir(rng, os_type)
    remote_dir = _pick_remote_dir(rng, remote_user)

    src_dir = f"{local_dir}/{dir_name}"
    local_fs = []
    expected_files = {}
    for f in files:
        content = _make_file_content(f, seed)
        local_fs.append({"path": f"{src_dir}/{f}", "content": content})
        expected_files[f"{dir_name}/{f}"] = content

    remote_fs = [{"path": remote_dir, "is_dir": True}]

    # Goal: files extracted on remote
    checks = []
    for rel, content in expected_files.items():
        checks.append({
            "type": "file_exists", "machine": "remote",
            "path": f"{remote_dir}/{rel}", "content": content,
        })

    return TerminalTask(
        seed=seed, local_os=os_type, task_category="composite",
        task_type=CompositeTemplate.PACK_UPLOAD_EXTRACT.value,
        remote_host=remote_host, remote_user=remote_user,
        description=(f"Pack '{_display_path(src_dir, os_type)}' into a {fmt} archive, "
                     f"upload to {remote_user}@{remote_host}:/{remote_dir}/, "
                     f"and extract it there"),
        goal_checks=checks,
        local_fs_spec=local_fs, remote_fs_spec=remote_fs,
        params={"fmt": fmt, "dir_name": dir_name, "local_dir": local_dir,
                "remote_dir": remote_dir, "files": files},
    )


def _gen_download_rename(rng, seed, os_type, remote_host, remote_user):
    """Download file from remote, rename locally."""
    old_name = rng.choice(FILENAMES)
    new_name = rng.choice(FILENAMES)
    while new_name == old_name:
        new_name = rng.choice(FILENAMES)
    content = _make_file_content(old_name, seed)
    local_dir = _pick_subdir(rng, os_type)
    remote_dir = _pick_remote_dir(rng, remote_user)

    remote_fs = [{"path": f"{remote_dir}/{old_name}", "content": content}]
    local_fs = [{"path": local_dir, "is_dir": True}]

    return TerminalTask(
        seed=seed, local_os=os_type, task_category="composite",
        task_type=CompositeTemplate.DOWNLOAD_RENAME.value,
        remote_host=remote_host, remote_user=remote_user,
        description=(f"Download '{old_name}' from "
                     f"{remote_user}@{remote_host}:/{remote_dir}/ "
                     f"and rename it to '{new_name}' in "
                     f"'{_display_path(local_dir, os_type)}'"),
        goal_checks=[
            {"type": "file_exists", "machine": "local",
             "path": f"{local_dir}/{new_name}", "content": content},
            {"type": "file_not_exists", "machine": "local",
             "path": f"{local_dir}/{old_name}"},
        ],
        local_fs_spec=local_fs, remote_fs_spec=remote_fs,
        params={"old_name": old_name, "new_name": new_name,
                "local_dir": local_dir, "remote_dir": remote_dir},
    )


def _gen_backup_replace(rng, seed, os_type, remote_host, remote_user):
    """Backup existing remote file to .bak, upload new version."""
    filename = rng.choice(FILENAMES)
    old_content = _make_file_content(filename, seed)
    new_content = _make_file_content(filename, seed + 99999)
    local_dir = _pick_subdir(rng, os_type)
    remote_dir = _pick_remote_dir(rng, remote_user)

    local_fs = [{"path": f"{local_dir}/{filename}", "content": new_content}]
    remote_fs = [{"path": f"{remote_dir}/{filename}", "content": old_content}]

    bak_name = filename + ".bak"

    return TerminalTask(
        seed=seed, local_os=os_type, task_category="composite",
        task_type=CompositeTemplate.BACKUP_REPLACE.value,
        remote_host=remote_host, remote_user=remote_user,
        description=(f"On the remote server, rename '/{remote_dir}/{filename}' to "
                     f"'{bak_name}' as a backup, then upload the new version from "
                     f"'{_display_path(local_dir + '/' + filename, os_type)}'"),
        goal_checks=[
            {"type": "file_exists", "machine": "remote",
             "path": f"{remote_dir}/{bak_name}", "content": old_content},
            {"type": "file_exists", "machine": "remote",
             "path": f"{remote_dir}/{filename}", "content": new_content},
        ],
        local_fs_spec=local_fs, remote_fs_spec=remote_fs,
        params={"filename": filename, "bak_name": bak_name,
                "local_dir": local_dir, "remote_dir": remote_dir},
    )


COMPOSITE_GENERATORS = {
    CompositeTemplate.PACK_UPLOAD: _gen_pack_upload,
    CompositeTemplate.DOWNLOAD_EXTRACT: _gen_download_extract,
    CompositeTemplate.MKDIR_UPLOAD: _gen_mkdir_upload,
    CompositeTemplate.UPLOAD_CHMOD: _gen_upload_chmod,
    CompositeTemplate.UPLOAD_DELETE_SOURCE: _gen_upload_delete_source,
    CompositeTemplate.PACK_UPLOAD_EXTRACT: _gen_pack_upload_extract,
    CompositeTemplate.DOWNLOAD_RENAME: _gen_download_rename,
    CompositeTemplate.BACKUP_REPLACE: _gen_backup_replace,
}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def generate_task(
    seed: int,
    composite_ratio: float = 0.5,
    forced_task_type: Optional[str] = None,
) -> TerminalTask:
    """Generate a complete terminal task from a seed.

    Args:
        seed: Random seed that fully determines the task.
        composite_ratio: Probability of generating a composite task (0-1).
        forced_task_type: If set, force this task type (TaskType or CompositeTemplate value).
    """
    rng = random.Random(seed)

    # Pick OS
    local_os = rng.choice(list(OSType))

    # Pick remote server
    remote_host = f"192.168.1.{rng.randint(10, 99)}"
    remote_user = rng.choice(REMOTE_USERS)

    # Decide task type
    if forced_task_type:
        # Try single first, then composite
        try:
            tt = TaskType(forced_task_type)
            return SINGLE_GENERATORS[tt](rng, seed, local_os, remote_host, remote_user)
        except ValueError:
            ct = CompositeTemplate(forced_task_type)
            return COMPOSITE_GENERATORS[ct](rng, seed, local_os, remote_host, remote_user)

    is_composite = rng.random() < composite_ratio
    if is_composite:
        template = rng.choice(list(CompositeTemplate))
        return COMPOSITE_GENERATORS[template](rng, seed, local_os, remote_host, remote_user)
    else:
        task_type = rng.choice(list(TaskType))
        return SINGLE_GENERATORS[task_type](rng, seed, local_os, remote_host, remote_user)


# ---------------------------------------------------------------------------
# Build environment from task
# ---------------------------------------------------------------------------


def build_env_from_task(task: TerminalTask, max_steps: int = 15):
    """Construct a TerminalEnv from a TerminalTask specification."""
    from trinity.common.workflows.connect_the_dots.terminal.commands import (
        build_command_registry,
    )

    os_type = task.local_os

    # Build local machine
    home = _home_dir(os_type)
    local_fs = VirtualFS()
    local_fs.makedirs(home)
    if os_type == OSType.WINDOWS:
        env_vars = {
            "USERPROFILE": _display_path(home, os_type),
            "USERNAME": "user",
            "CD": _display_path(home, os_type),
        }
    else:
        env_vars = {
            "HOME": _display_path(home, os_type),
            "USER": "user",
            "PWD": _display_path(home, os_type),
        }

    for spec in task.local_fs_spec:
        path = spec["path"]
        if spec.get("is_dir"):
            local_fs.makedirs(path)
        else:
            parent = "/".join(path.split("/")[:-1])
            if parent:
                local_fs.makedirs(parent)
            local_fs.writetext(path, spec.get("content", ""))
            meta = local_fs.get_meta(path)
            if "permissions" in spec:
                meta.permissions = spec["permissions"]
            if "archive_type" in spec:
                meta.archive_type = spec["archive_type"]
                meta.archive_entries = spec.get("archive_entries")

    local = MachineState(
        os_type=os_type, hostname="localhost", username="user",
        home_dir=home, cwd=home, fs=local_fs, env_vars=env_vars,
    )

    # Build remote machine
    remote_home = f"home/{task.remote_user}"
    remote_fs = VirtualFS()
    remote_fs.makedirs(remote_home)

    for spec in task.remote_fs_spec:
        path = spec["path"]
        if spec.get("is_dir"):
            remote_fs.makedirs(path)
        else:
            parent = "/".join(path.split("/")[:-1])
            if parent:
                remote_fs.makedirs(parent)
            remote_fs.writetext(path, spec.get("content", ""))
            meta = remote_fs.get_meta(path)
            if "permissions" in spec:
                meta.permissions = spec["permissions"]
            if "archive_type" in spec:
                meta.archive_type = spec["archive_type"]
                meta.archive_entries = spec.get("archive_entries")

    remote = MachineState(
        os_type=OSType.LINUX, hostname=task.remote_host, username=task.remote_user,
        home_dir=remote_home, cwd=remote_home, fs=remote_fs,
        env_vars={
            "HOME": f"/home/{task.remote_user}",
            "USER": task.remote_user,
            "IP": task.remote_host,
        },
    )

    handlers = build_command_registry()
    return TerminalEnv(local, remote, handlers, max_steps=max_steps)
