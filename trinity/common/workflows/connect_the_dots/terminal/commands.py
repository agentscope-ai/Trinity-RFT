# -*- coding: utf-8 -*-
"""
Command handlers for the simulated terminal environment.

Each handler is a class with an ``execute(args, env)`` method that returns
a string (the terminal output).  Handlers read/write the virtual filesystem
via ``env.current_machine.fs`` and ``env.current_machine`` state.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Set

from trinity.common.workflows.connect_the_dots.terminal.env import OSType

if TYPE_CHECKING:
    from trinity.common.workflows.connect_the_dots.terminal.env import TerminalEnv


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class CommandHandler(ABC):
    """Base class for command handlers."""

    name: str = ""
    # OS where this command is available.  Empty set = all OS.
    available_os: Set[OSType] = set()
    # If True, command can only run when NOT ssh-connected (on local machine).
    local_only: bool = False

    def is_available(self, os_type: OSType, ssh_connected: bool) -> bool:
        if self.local_only and ssh_connected:
            return False
        if ssh_connected:
            # Remote is always Linux
            return True
        if self.available_os and os_type not in self.available_os:
            return False
        return True

    @abstractmethod
    def execute(self, args: List[str], env: TerminalEnv) -> str:
        ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _machine(env: TerminalEnv):
    return env.current_machine


def _resolve(env: TerminalEnv, path: str) -> str:
    return _machine(env).resolve_path(path)


def _display(env: TerminalEnv, memfs_path: str) -> str:
    return _machine(env).to_display_path(memfs_path)


def _format_size(size: int) -> str:
    if size < 1024:
        return f"{size}"
    elif size < 1024 * 1024:
        return f"{size / 1024:.1f}K"
    else:
        return f"{size / (1024 * 1024):.1f}M"


def _parse_scp_target(arg: str):
    """Parse ``user@host:path``, ``user@host:``, or ``user@host``. Returns (user, host, path) or (None, None, path)."""
    if ":" in arg and "@" in arg.split(":")[0]:
        user_host, path = arg.split(":", 1)
        user, host = user_host.split("@", 1)
        return user, host, path
    # user@host without colon — treat as remote home directory
    if "@" in arg and "/" not in arg and "\\" not in arg:
        user, host = arg.split("@", 1)
        return user, host, ""
    return None, None, arg


# ---------------------------------------------------------------------------
# ls
# ---------------------------------------------------------------------------

class LsHandler(CommandHandler):
    name = "ls"

    def execute(self, args: List[str], env: TerminalEnv) -> str:
        m = _machine(env)
        fs = m.fs

        show_long = False
        show_all = False
        paths = []
        for a in args:
            if a.startswith("--"):
                if a == "--all":
                    show_all = True
                elif a == "--long":
                    show_long = True
            elif a.startswith("-"):
                for ch in a[1:]:
                    if ch == "l":
                        show_long = True
                    elif ch == "a":
                        show_all = True
            else:
                paths.append(a)

        if not paths:
            paths = [m.to_display_path(m.cwd)]

        all_output = []
        for p in paths:
            mp = _resolve(env, p)
            if not fs.exists(mp):
                all_output.append(f"ls: cannot access '{p}': No such file or directory")
                continue
            if fs.isfile(mp):
                if show_long:
                    all_output.append(self._long_entry(fs, mp, mp.split("/")[-1]))
                else:
                    all_output.append(mp.split("/")[-1])
                continue
            try:
                entries = fs.listdir(mp)
            except Exception:
                all_output.append(f"ls: cannot access '{p}': No such file or directory")
                continue

            if not show_all:
                entries = [e for e in entries if not e.startswith(".")]

            # Truncate long listings
            truncated = False
            if len(entries) > 50:
                total = len(entries)
                entries = entries[:50]
                truncated = True

            if show_long:
                lines = []
                for name in entries:
                    child = f"{mp}/{name}" if mp else name
                    lines.append(self._long_entry(fs, child, name))
                if truncated:
                    lines.append(f"... ({total - 50} more entries)")
                all_output.append("\n".join(lines))
            else:
                if m.os_type == OSType.WINDOWS and not env.ssh_connected:
                    all_output.append(self._dir_format(fs, mp, entries, truncated,
                                                       total if truncated else len(entries)))
                else:
                    line = "  ".join(
                        f"{name}/" if fs.isdir(f"{mp}/{name}" if mp else name) else name
                        for name in entries
                    )
                    if truncated:
                        line += f"\n... ({total - 50} more entries)"
                    all_output.append(line)

        return "\n".join(all_output)

    def _long_entry(self, fs, memfs_path, name):
        meta = fs.get_meta(memfs_path)
        is_dir = fs.isdir(memfs_path)
        perm_str = "d" if is_dir else "-"
        mode = int(meta.permissions, 8) if meta.permissions.isdigit() else 0o644
        for shift in (6, 3, 0):
            bits = (mode >> shift) & 7
            perm_str += "r" if bits & 4 else "-"
            perm_str += "w" if bits & 2 else "-"
            perm_str += "x" if bits & 1 else "-"
        size = meta.size if not is_dir else 4096
        return f"{perm_str}  1 {meta.owner} {meta.group} {size:>8} Jan 15 10:30 {name}{'/' if is_dir else ''}"

    def _dir_format(self, fs, mp, entries, truncated, total_count):
        """Windows dir format."""
        dir_display = _machine_stub_display(mp)
        lines = [
            " Volume in drive C has no label.",
            f" Directory of {dir_display}",
            "",
        ]
        file_count = 0
        dir_count = 0
        total_size = 0
        for name in entries:
            child = f"{mp}/{name}" if mp else name
            is_dir = fs.isdir(child)
            meta = fs.get_meta(child)
            if is_dir:
                dir_count += 1
                lines.append(f"01/15/2024  10:30 AM    <DIR>          {name}")
            else:
                file_count += 1
                total_size += meta.size
                lines.append(f"01/15/2024  10:30 AM          {meta.size:>8} {name}")
        if truncated:
            lines.append(f"... ({total_count - 50} more entries)")
        lines.append(f"               {file_count} File(s)    {total_size:>10} bytes")
        lines.append(f"               {dir_count} Dir(s)   500,000,000 bytes free")
        return "\n".join(lines)


def _machine_stub_display(mp):
    """Quick display for dir header."""
    parts = mp.split("/")
    if parts and len(parts[0]) == 1 and parts[0].isalpha():
        drive = parts[0].upper()
        return f"{drive}:\\" + "\\".join(parts[1:])
    return "/" + mp


# ---------------------------------------------------------------------------
# cd
# ---------------------------------------------------------------------------

class CdHandler(CommandHandler):
    name = "cd"

    def execute(self, args: List[str], env: TerminalEnv) -> str:
        m = _machine(env)
        if not args:
            if m.os_type == OSType.WINDOWS and not env.ssh_connected:
                return m.to_display_path(m.cwd)
            m.cwd = m.home_dir
            self._update_pwd(m)
            return ""
        target = args[0]
        mp = _resolve(env, target)
        if not m.fs.exists(mp):
            if m.os_type == OSType.WINDOWS and not env.ssh_connected:
                return "The system cannot find the path specified."
            return f"bash: cd: {target}: No such file or directory"
        if not m.fs.isdir(mp):
            if m.os_type == OSType.WINDOWS and not env.ssh_connected:
                return "The directory name is invalid."
            return f"bash: cd: {target}: Not a directory"
        m.cwd = mp
        self._update_pwd(m)
        if m.os_type == OSType.WINDOWS and not env.ssh_connected:
            return m.to_display_path(m.cwd)
        return ""

    @staticmethod
    def _update_pwd(m):
        display = m.to_display_path(m.cwd)
        if m.os_type == OSType.WINDOWS:
            m.env_vars["CD"] = display
        else:
            m.env_vars["PWD"] = display


# ---------------------------------------------------------------------------
# pwd
# ---------------------------------------------------------------------------

class PwdHandler(CommandHandler):
    name = "pwd"

    def execute(self, args: List[str], env: TerminalEnv) -> str:
        return _machine(env).to_display_path(_machine(env).cwd)


# ---------------------------------------------------------------------------
# cat
# ---------------------------------------------------------------------------

class CatHandler(CommandHandler):
    name = "cat"

    def execute(self, args: List[str], env: TerminalEnv) -> str:
        if not args:
            return "cat: missing operand"
        m = _machine(env)
        outputs = []
        for f in args:
            mp = _resolve(env, f)
            if not m.fs.exists(mp):
                outputs.append(f"cat: {f}: No such file or directory")
            elif m.fs.isdir(mp):
                outputs.append(f"cat: {f}: Is a directory")
            else:
                meta = m.fs.get_meta(mp)
                if meta.archive_type:
                    outputs.append(f"[Binary file - {meta.archive_type} archive, "
                                   f"{len(meta.archive_entries or {})} entries]")
                else:
                    outputs.append(m.fs.readtext(mp))
        return "\n".join(outputs)


# ---------------------------------------------------------------------------
# mkdir
# ---------------------------------------------------------------------------

class MkdirHandler(CommandHandler):
    name = "mkdir"

    def execute(self, args: List[str], env: TerminalEnv) -> str:
        m = _machine(env)
        create_parents = False
        paths = []
        for a in args:
            if a in ("-p", "--parents"):
                create_parents = True
            else:
                paths.append(a)
        if not paths:
            return "mkdir: missing operand"
        outputs = []
        for p in paths:
            mp = _resolve(env, p)
            try:
                if create_parents:
                    m.fs.makedirs(mp)
                else:
                    m.fs.mkdir(mp)
            except FileExistsError:
                outputs.append(f"mkdir: cannot create directory '{p}': File exists")
            except FileNotFoundError:
                outputs.append(f"mkdir: cannot create directory '{p}': No such file or directory")
            except NotADirectoryError:
                outputs.append(f"mkdir: cannot create directory '{p}': Not a directory")
        return "\n".join(outputs)


# ---------------------------------------------------------------------------
# chmod
# ---------------------------------------------------------------------------

class ChmodHandler(CommandHandler):
    name = "chmod"
    available_os = {OSType.LINUX, OSType.MAC}

    def execute(self, args: List[str], env: TerminalEnv) -> str:
        if len(args) < 2:
            return "chmod: missing operand"
        recursive = False
        mode_str = None
        paths = []
        for a in args:
            if a in ("-R", "--recursive"):
                recursive = True
            elif mode_str is None and len(a) == 3 and all(c in "01234567" for c in a):
                mode_str = a
            else:
                paths.append(a)
        if mode_str is None:
            return f"chmod: invalid mode: '{args[0]}'"
        if not paths:
            return "chmod: missing operand"

        m = _machine(env)
        outputs = []
        for p in paths:
            mp = _resolve(env, p)
            if not m.fs.exists(mp):
                outputs.append(f"chmod: cannot access '{p}': No such file or directory")
                continue
            if recursive and m.fs.isdir(mp):
                for dirpath, dirs, files in m.fs.walk(mp):
                    for name in dirs + files:
                        child = f"{dirpath}/{name}" if dirpath else name
                        meta = m.fs.get_meta(child)
                        meta.permissions = mode_str
            else:
                meta = m.fs.get_meta(mp)
                meta.permissions = mode_str
        return "\n".join(outputs)


# ---------------------------------------------------------------------------
# cp
# ---------------------------------------------------------------------------

class CpHandler(CommandHandler):
    name = "cp"

    def execute(self, args: List[str], env: TerminalEnv) -> str:
        recursive = False
        paths = []
        for a in args:
            if a in ("-r", "-R", "--recursive"):
                recursive = True
            else:
                paths.append(a)
        if len(paths) < 2:
            return "cp: missing destination operand"

        m = _machine(env)
        src = _resolve(env, paths[0])
        dst = _resolve(env, paths[1])

        if not m.fs.exists(src):
            return f"cp: cannot stat '{paths[0]}': No such file or directory"
        if m.fs.isdir(src) and not recursive:
            return f"cp: -r not specified; omitting directory '{paths[0]}'"

        # If dst is an existing dir, copy into it
        if m.fs.isdir(dst):
            name = src.split("/")[-1]
            dst = f"{dst}/{name}"

        try:
            if m.fs.isdir(src):
                m.fs.copy_tree(src, dst)
            else:
                # Ensure parent exists
                parent = "/".join(dst.split("/")[:-1])
                if parent and not m.fs.exists(parent):
                    return f"cp: cannot create regular file '{paths[1]}': No such file or directory"
                m.fs.copy_file(src, dst)
        except Exception as e:
            return f"cp: {e}"
        return ""


# ---------------------------------------------------------------------------
# mv
# ---------------------------------------------------------------------------

class MvHandler(CommandHandler):
    name = "mv"

    def execute(self, args: List[str], env: TerminalEnv) -> str:
        paths = [a for a in args if not a.startswith("-")]
        if len(paths) < 2:
            return "mv: missing destination operand"

        m = _machine(env)
        src = _resolve(env, paths[0])
        dst = _resolve(env, paths[1])

        if not m.fs.exists(src):
            return f"mv: cannot stat '{paths[0]}': No such file or directory"

        # If dst is existing dir, move into it
        if m.fs.isdir(dst):
            name = src.split("/")[-1]
            dst = f"{dst}/{name}"

        # Check parent of dst
        dst_parent = "/".join(dst.split("/")[:-1])
        if dst_parent and not m.fs.isdir(dst_parent):
            return f"mv: cannot move '{paths[0]}' to '{paths[1]}': No such file or directory"

        try:
            if m.fs.isdir(src):
                m.fs.copy_tree(src, dst)
            else:
                m.fs.copy_file(src, dst)
            m.fs.remove_recursive(src)
        except Exception as e:
            return f"mv: {e}"
        return ""


# ---------------------------------------------------------------------------
# rm
# ---------------------------------------------------------------------------

class RmHandler(CommandHandler):
    name = "rm"

    def execute(self, args: List[str], env: TerminalEnv) -> str:
        recursive = False
        force = False
        paths = []
        for a in args:
            if a.startswith("-"):
                flags = a.lstrip("-")
                if "r" in flags or "R" in flags:
                    recursive = True
                if "f" in flags:
                    force = True
            else:
                paths.append(a)
        if not paths:
            return "rm: missing operand"

        m = _machine(env)
        outputs = []
        for p in paths:
            mp = _resolve(env, p)
            if not m.fs.exists(mp):
                if not force:
                    outputs.append(f"rm: cannot remove '{p}': No such file or directory")
                continue
            if m.fs.isdir(mp) and not recursive:
                outputs.append(f"rm: cannot remove '{p}': Is a directory")
                continue
            try:
                if recursive:
                    m.fs.remove_recursive(mp)
                else:
                    m.fs.remove(mp)
            except Exception as e:
                outputs.append(f"rm: {e}")
        return "\n".join(outputs)


# ---------------------------------------------------------------------------
# touch
# ---------------------------------------------------------------------------

class TouchHandler(CommandHandler):
    name = "touch"
    available_os = {OSType.LINUX, OSType.MAC}

    def execute(self, args: List[str], env: TerminalEnv) -> str:
        if not args:
            return "touch: missing file operand"
        m = _machine(env)
        for f in args:
            if f.startswith("-"):
                continue
            mp = _resolve(env, f)
            if not m.fs.exists(mp):
                parent = "/".join(mp.split("/")[:-1])
                if parent and not m.fs.isdir(parent):
                    return f"touch: cannot touch '{f}': No such file or directory"
                m.fs.writetext(mp, "")
        return ""


# ---------------------------------------------------------------------------
# echo
# ---------------------------------------------------------------------------

class EchoHandler(CommandHandler):
    name = "echo"

    def execute(self, args: List[str], env: TerminalEnv) -> str:
        return " ".join(args)


# ---------------------------------------------------------------------------
# whoami
# ---------------------------------------------------------------------------

class WhoamiHandler(CommandHandler):
    name = "whoami"

    def execute(self, args: List[str], env: TerminalEnv) -> str:
        return _machine(env).username


# ---------------------------------------------------------------------------
# ssh
# ---------------------------------------------------------------------------

class SshHandler(CommandHandler):
    name = "ssh"

    def execute(self, args: List[str], env: TerminalEnv) -> str:
        if env.ssh_connected:
            return "bash: already connected to remote host. Use 'exit' first."

        # Parse: ssh [-p port] [-i key] user@host
        target = None
        i = 0
        while i < len(args):
            if args[i] in ("-p", "-i") and i + 1 < len(args):
                i += 2  # skip flag and value
                continue
            if not args[i].startswith("-"):
                target = args[i]
                break
            i += 1

        if target is None:
            return "usage: ssh [user@]hostname"

        if "@" not in target:
            return f"ssh: Could not resolve hostname {target}: Name or service not known"

        user, host = target.split("@", 1)
        # Check against known remote
        if host != env.remote.hostname and host != env.remote.env_vars.get("IP", ""):
            return f"ssh: connect to host {host} port 22: Connection refused"
        if user != env.remote.username:
            return f"Permission denied (publickey,password)."

        env.ssh_connected = True
        env.current_machine = env.remote
        return f"Welcome to Ubuntu 22.04 LTS ({env.remote.hostname})\nLast login: Mon Jan 15 10:30:00 2024"


# ---------------------------------------------------------------------------
# exit
# ---------------------------------------------------------------------------

class ExitHandler(CommandHandler):
    name = "exit"

    def execute(self, args: List[str], env: TerminalEnv) -> str:
        if env.ssh_connected:
            host = env.remote.hostname
            env.ssh_connected = False
            env.current_machine = env.local
            return f"Connection to {host} closed."
        return ""


# ---------------------------------------------------------------------------
# scp
# ---------------------------------------------------------------------------

class ScpHandler(CommandHandler):
    name = "scp"

    def execute(self, args: List[str], env: TerminalEnv) -> str:
        if env.ssh_connected:
            return "bash: scp must be run from local machine. Use 'exit' to disconnect first."

        recursive = False
        positional = []
        i = 0
        while i < len(args):
            a = args[i]
            if a in ("-r", "-R"):
                recursive = True
            elif a in ("-P", "-i") and i + 1 < len(args):
                i += 1  # skip value
            elif not a.startswith("-"):
                positional.append(a)
            i += 1

        if len(positional) < 2:
            return "usage: scp [-r] source ... target"

        src_arg = positional[0]
        dst_arg = positional[1]

        src_user, src_host, src_path = _parse_scp_target(src_arg)
        dst_user, dst_host, dst_path = _parse_scp_target(dst_arg)

        # Determine source/dest machines
        if src_host and dst_host:
            return "scp: copying between two remote hosts is not supported"
        elif src_host:
            # Download: remote -> local
            src_machine = env.remote
            dst_machine = env.local
            if src_host != env.remote.hostname and src_host != env.remote.env_vars.get("IP", ""):
                return f"ssh: connect to host {src_host} port 22: Connection refused"
        elif dst_host:
            # Upload: local -> remote
            src_machine = env.local
            dst_machine = env.remote
            if dst_host != env.remote.hostname and dst_host != env.remote.env_vars.get("IP", ""):
                return f"ssh: connect to host {dst_host} port 22: Connection refused"
        else:
            return "scp: use cp for local-to-local copy"

        src_mp = src_machine.resolve_path(src_path)
        dst_mp = dst_machine.resolve_path(dst_path)

        if not src_machine.fs.exists(src_mp):
            return f"scp: {src_path}: No such file or directory"

        if src_machine.fs.isdir(src_mp) and not recursive:
            return f"scp: {src_path}: not a regular file"

        # If dst is existing dir, copy into it
        if dst_machine.fs.isdir(dst_mp):
            name = src_mp.split("/")[-1]
            dst_mp = f"{dst_mp}/{name}"

        # Check dst parent exists
        dst_parent = "/".join(dst_mp.split("/")[:-1])
        if dst_parent and not dst_machine.fs.isdir(dst_parent):
            return f"scp: {dst_path}: No such file or directory"

        try:
            self._cross_copy(src_machine, src_mp, dst_machine, dst_mp, recursive)
        except Exception as e:
            return f"scp: {e}"

        filename = src_mp.split("/")[-1]
        size = src_machine.fs.get_meta(src_mp).size
        return f"{filename}                    100% {_format_size(size)}     transferred"

    def _cross_copy(self, src_m, src_mp, dst_m, dst_mp, recursive):
        """Copy across two different MachineStates."""
        src_fs = src_m.fs
        dst_fs = dst_m.fs
        if src_fs.isfile(src_mp):
            content = src_fs.readtext(src_mp)
            dst_fs.writetext(dst_mp, content, create_parents=False)
            src_meta = src_fs.get_meta(src_mp)
            dst_meta = dst_fs.get_meta(dst_mp)
            dst_meta.permissions = src_meta.permissions
            dst_meta.owner = dst_m.username
            dst_meta.group = dst_m.username
            dst_meta.archive_type = src_meta.archive_type
            dst_meta.archive_entries = (
                dict(src_meta.archive_entries) if src_meta.archive_entries else None
            )
        elif recursive:
            dst_fs.makedirs(dst_mp)
            for name in src_fs.listdir(src_mp):
                s = f"{src_mp}/{name}"
                d = f"{dst_mp}/{name}"
                self._cross_copy(src_m, s, dst_m, d, True)


# ---------------------------------------------------------------------------
# rsync
# ---------------------------------------------------------------------------

class RsyncHandler(CommandHandler):
    name = "rsync"
    available_os = {OSType.LINUX, OSType.MAC}

    def execute(self, args: List[str], env: TerminalEnv) -> str:
        if env.ssh_connected:
            return "bash: rsync must be run from local machine. Use 'exit' to disconnect first."

        # Strip flags, find positional args
        positional = []
        i = 0
        while i < len(args):
            a = args[i]
            if a in ("-e",) and i + 1 < len(args):
                i += 1  # skip value
            elif not a.startswith("-"):
                positional.append(a)
            i += 1

        if len(positional) < 2:
            return "usage: rsync [options] source destination"

        src_arg = positional[0]
        dst_arg = positional[1]

        # Reuse SCP logic for cross-machine copy
        scp = ScpHandler()
        # Build equivalent scp args
        scp_args = ["-r", src_arg, dst_arg]
        result = scp.execute(scp_args, env)

        if "transferred" in result:
            filename = src_arg.rstrip("/").split("/")[-1]
            return (f"sending incremental file list\n"
                    f"{filename}\n"
                    f"\nsent 1024 bytes  received 42 bytes  2132.00 bytes/sec\n"
                    f"total size is 1024  speedup is 0.96")
        return result


# ---------------------------------------------------------------------------
# tar
# ---------------------------------------------------------------------------

class TarHandler(CommandHandler):
    name = "tar"
    available_os = {OSType.LINUX, OSType.MAC}

    def execute(self, args: List[str], env: TerminalEnv) -> str:
        if not args:
            return "tar: You must specify one of the '-Acdtrux' options"

        m = _machine(env)
        fs = m.fs

        # Collect all flags and positional args.
        # tar accepts flags in many forms: -czf, czf, -c -z -f, --create, etc.
        mode = None
        gzip = False
        archive_path = None
        file_args = []
        expect_archive = False  # next positional arg is the archive path

        for a in args:
            if expect_archive:
                archive_path = a
                expect_archive = False
                continue
            if a.startswith("--"):
                if a == "--create":
                    mode = "create"
                elif a in ("--extract", "--get"):
                    mode = "extract"
                elif a == "--list":
                    mode = "list"
                elif a in ("--gzip", "--gunzip", "--ungzip"):
                    gzip = True
                elif a.startswith("--file="):
                    archive_path = a.split("=", 1)[1]
                elif a == "--file":
                    expect_archive = True
                # --verbose and other long flags are silently ignored
                continue
            if a.startswith("-") or (a == args[0] and len(a) <= 6 and a[0] in "cxtzvf"):
                # Short flags: -czf, czf, -c, -z, -f
                chars = a.lstrip("-")
                for ch in chars:
                    if ch == "c":
                        mode = "create"
                    elif ch == "x":
                        mode = "extract"
                    elif ch == "t":
                        mode = "list"
                    elif ch == "z":
                        gzip = True
                    elif ch == "f":
                        expect_archive = True
                    # v and other single-char flags silently ignored
                continue
            # Positional arg
            file_args.append(a)

        if archive_path is None:
            return "tar: Refusing to read archive contents from terminal"

        archive_mp = _resolve(env, archive_path)
        archive_type = "tar.gz" if gzip else "tar"

        if mode == "create":
            if not file_args:
                return "tar: Cowardly refusing to create an empty archive"
            entries = {}
            for f in file_args:
                fmp = _resolve(env, f)
                if not fs.exists(fmp):
                    return f"tar: {f}: Cannot open: No such file or directory"
                if fs.isfile(fmp):
                    basename = fmp.split("/")[-1]
                    entries[basename] = fs.readtext(fmp)
                elif fs.isdir(fmp):
                    base = fmp.rstrip("/").split("/")[-1]
                    for dp, dirs, files in fs.walk(fmp):
                        for fname in files:
                            child = f"{dp}/{fname}" if dp else fname
                            rel = child[len(fmp):].lstrip("/") if child.startswith(fmp) else child
                            full_rel = f"{base}/{rel}" if rel else base
                            entries[full_rel] = fs.readtext(child)

            content = json.dumps({"type": archive_type, "entries": entries})
            parent = "/".join(archive_mp.split("/")[:-1])
            if parent and not fs.isdir(parent):
                return f"tar: {archive_path}: Cannot open: No such file or directory"
            fs.writetext(archive_mp, content)
            meta = fs.get_meta(archive_mp)
            meta.archive_type = archive_type
            meta.archive_entries = entries
            meta.size = sum(len(v) for v in entries.values())
            return ""

        elif mode == "extract":
            if not fs.exists(archive_mp):
                return f"tar: {archive_path}: Cannot open: No such file or directory"
            meta = fs.get_meta(archive_mp)
            if not meta.archive_type or "tar" not in meta.archive_type:
                return f"tar: {archive_path}: This does not look like a tar archive"
            entries = meta.archive_entries or {}
            for rel_path, content in entries.items():
                out_path = _resolve(env, rel_path)
                parent = "/".join(out_path.split("/")[:-1])
                if parent:
                    fs.makedirs(parent)
                fs.writetext(out_path, content)
                out_meta = fs.get_meta(out_path)
                out_meta.owner = m.username
                out_meta.group = m.username
            return ""

        elif mode == "list":
            if not fs.exists(archive_mp):
                return f"tar: {archive_path}: Cannot open: No such file or directory"
            meta = fs.get_meta(archive_mp)
            if not meta.archive_type or "tar" not in meta.archive_type:
                return f"tar: {archive_path}: This does not look like a tar archive"
            entries = meta.archive_entries or {}
            return "\n".join(sorted(entries.keys()))

        return "tar: You must specify one of the '-Acdtrux' options"


# ---------------------------------------------------------------------------
# zip
# ---------------------------------------------------------------------------

class ZipHandler(CommandHandler):
    name = "zip"

    def execute(self, args: List[str], env: TerminalEnv) -> str:
        if len(args) < 2:
            return "zip: missing archive name or files"

        m = _machine(env)
        fs = m.fs
        recursive = False
        positional = []
        for a in args:
            if a in ("-r", "-R"):
                recursive = True
            elif not a.startswith("-"):
                positional.append(a)

        if len(positional) < 2:
            return "zip: missing archive name or files"

        archive_name = positional[0]
        file_args = positional[1:]
        archive_mp = _resolve(env, archive_name)

        entries = {}
        output_lines = []
        for f in file_args:
            fmp = _resolve(env, f)
            if not fs.exists(fmp):
                return f"zip error: Nothing to do! ({f} not found)"
            if fs.isfile(fmp):
                basename = fmp.split("/")[-1]
                entries[basename] = fs.readtext(fmp)
                output_lines.append(f"  adding: {basename} (stored 0%)")
            elif fs.isdir(fmp):
                if not recursive:
                    output_lines.append(f"  adding: {f}/ (stored 0%)")
                    continue
                base = fmp.rstrip("/").split("/")[-1]
                for dp, dirs, files in fs.walk(fmp):
                    for fname in files:
                        child = f"{dp}/{fname}" if dp else fname
                        rel = child[len(fmp):].lstrip("/") if child.startswith(fmp) else child
                        full_rel = f"{base}/{rel}" if rel else base
                        entries[full_rel] = fs.readtext(child)
                        output_lines.append(f"  adding: {full_rel} (stored 0%)")

        content = json.dumps({"type": "zip", "entries": entries})
        parent = "/".join(archive_mp.split("/")[:-1])
        if parent and not fs.isdir(parent):
            fs.makedirs(parent)
        fs.writetext(archive_mp, content)
        meta = fs.get_meta(archive_mp)
        meta.archive_type = "zip"
        meta.archive_entries = entries
        meta.size = sum(len(v) for v in entries.values())

        return "\n".join(output_lines)


# ---------------------------------------------------------------------------
# unzip
# ---------------------------------------------------------------------------

class UnzipHandler(CommandHandler):
    name = "unzip"

    def execute(self, args: List[str], env: TerminalEnv) -> str:
        if not args:
            return "unzip: missing archive name"

        m = _machine(env)
        fs = m.fs
        archive_name = None
        dest_dir = None
        i = 0
        while i < len(args):
            if args[i] == "-d" and i + 1 < len(args):
                dest_dir = args[i + 1]
                i += 2
                continue
            if not args[i].startswith("-"):
                archive_name = args[i]
            i += 1

        if archive_name is None:
            return "unzip: missing archive name"

        archive_mp = _resolve(env, archive_name)
        if not fs.exists(archive_mp):
            return f"unzip: cannot find or open {archive_name}"

        meta = fs.get_meta(archive_mp)
        if meta.archive_type != "zip":
            return f"unzip: {archive_name} is not a zip archive"

        entries = meta.archive_entries or {}
        output_lines = [f"Archive:  {archive_name}"]

        for rel_path, content in entries.items():
            if dest_dir:
                out_path = _resolve(env, f"{dest_dir}/{rel_path}")
            else:
                out_path = _resolve(env, rel_path)
            parent = "/".join(out_path.split("/")[:-1])
            if parent:
                fs.makedirs(parent)
            fs.writetext(out_path, content)
            out_meta = fs.get_meta(out_path)
            out_meta.owner = m.username
            out_meta.group = m.username
            output_lines.append(f"  extracting: {rel_path}")

        return "\n".join(output_lines)


# ---------------------------------------------------------------------------
# gzip / gunzip
# ---------------------------------------------------------------------------

class GzipHandler(CommandHandler):
    name = "gzip"
    available_os = {OSType.LINUX, OSType.MAC}

    def execute(self, args: List[str], env: TerminalEnv) -> str:
        paths = [a for a in args if not a.startswith("-")]
        if not paths:
            return "gzip: missing file operand"

        m = _machine(env)
        fs = m.fs
        for f in paths:
            mp = _resolve(env, f)
            if not fs.exists(mp):
                return f"gzip: {f}: No such file or directory"
            if fs.isdir(mp):
                return f"gzip: {f}: Is a directory"
            content = fs.readtext(mp)
            gz_path = mp + ".gz"
            gz_content = json.dumps({"type": "gzip", "entries": {f: content}})
            fs.writetext(gz_path, gz_content)
            gz_meta = fs.get_meta(gz_path)
            gz_meta.archive_type = "gzip"
            gz_meta.archive_entries = {f: content}
            gz_meta.size = len(content)
            fs.remove(mp)
        return ""


class GunzipHandler(CommandHandler):
    name = "gunzip"
    available_os = {OSType.LINUX, OSType.MAC}

    def execute(self, args: List[str], env: TerminalEnv) -> str:
        paths = [a for a in args if not a.startswith("-")]
        if not paths:
            return "gunzip: missing file operand"

        m = _machine(env)
        fs = m.fs
        for f in paths:
            mp = _resolve(env, f)
            if not fs.exists(mp):
                return f"gunzip: {f}: No such file or directory"
            meta = fs.get_meta(mp)
            if meta.archive_type != "gzip":
                return f"gunzip: {f}: not in gzip format"
            entries = meta.archive_entries or {}
            for orig_name, content in entries.items():
                out_path = _resolve(env, orig_name)
                fs.writetext(out_path, content)
            fs.remove(mp)
        return ""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def build_command_registry() -> Dict[str, CommandHandler]:
    """Build the command name -> handler mapping."""
    handlers = [
        LsHandler(),
        CdHandler(),
        PwdHandler(),
        CatHandler(),
        MkdirHandler(),
        ChmodHandler(),
        CpHandler(),
        MvHandler(),
        RmHandler(),
        TouchHandler(),
        EchoHandler(),
        WhoamiHandler(),
        SshHandler(),
        ExitHandler(),
        ScpHandler(),
        RsyncHandler(),
        TarHandler(),
        ZipHandler(),
        UnzipHandler(),
        GzipHandler(),
        GunzipHandler(),
    ]
    registry = {}
    for h in handlers:
        registry[h.name] = h
    return registry
