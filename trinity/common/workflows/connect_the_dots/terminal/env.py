# -*- coding: utf-8 -*-
"""
Virtual terminal environment with in-memory filesystem.

No real commands are executed. All state is held in Python dicts.
"""

import re
import shlex
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class OSType(Enum):
    WINDOWS = "windows"
    MAC = "mac"
    LINUX = "linux"


# ---------------------------------------------------------------------------
# Virtual Filesystem
# ---------------------------------------------------------------------------



@dataclass
class FileMeta:
    """Metadata attached to every file/directory."""
    permissions: str = "644"
    owner: str = "user"
    group: str = "user"
    size: int = 0
    # For simulated archives
    archive_type: Optional[str] = None       # "tar", "zip", "tar.gz", "gzip"
    archive_entries: Optional[Dict[str, str]] = None


class VirtualFS:
    """Minimal in-memory filesystem backed by nested dicts.

    Internal layout::

        _tree = {
            "home": {
                "user": {
                    "file.txt": "content string",
                    "subdir": { ... },
                }
            }
        }

    - A *str* value  => regular file (the string is its content).
    - A *dict* value => directory.

    All paths inside VirtualFS use ``/`` separators with no leading slash.
    The caller (MachineState) is responsible for converting OS-specific paths
    to this canonical form *before* calling VirtualFS methods.
    """

    def __init__(self):
        self._tree: Dict[str, Any] = {}
        self._meta: Dict[str, FileMeta] = {}  # canonical_path -> metadata

    # -- helpers --------------------------------------------------------

    @staticmethod
    def _split(path: str) -> List[str]:
        """Split a canonical path into parts, filtering blanks."""
        return [p for p in path.replace("\\", "/").split("/") if p]

    def _navigate(self, parts: List[str], create_parents: bool = False):
        """Walk *parts* from root. Returns (parent_dict, last_key).

        If *create_parents* is True, intermediate dirs are created.
        Raises FileNotFoundError if a segment doesn't exist (and create is off).
        Raises NotADirectoryError if a segment is a file, not a dir.
        """
        node = self._tree
        for i, part in enumerate(parts[:-1]):
            child = node.get(part)
            if child is None:
                if create_parents:
                    node[part] = {}
                    dir_path = "/".join(parts[: i + 1])
                    self._meta[dir_path] = FileMeta(permissions="755", size=4096)
                    node = node[part]
                else:
                    raise FileNotFoundError("/".join(parts[: i + 1]))
            elif isinstance(child, dict):
                node = child
            else:
                raise NotADirectoryError("/".join(parts[: i + 1]))
        return (node, parts[-1]) if parts else (self._tree, "")

    # -- public API -----------------------------------------------------

    def exists(self, path: str) -> bool:
        parts = self._split(path)
        if not parts:
            return True  # root always exists
        try:
            parent, key = self._navigate(parts)
            return key in parent
        except (FileNotFoundError, NotADirectoryError):
            return False

    def isdir(self, path: str) -> bool:
        parts = self._split(path)
        if not parts:
            return True
        try:
            parent, key = self._navigate(parts)
            return isinstance(parent.get(key), dict)
        except (FileNotFoundError, NotADirectoryError):
            return False

    def isfile(self, path: str) -> bool:
        parts = self._split(path)
        if not parts:
            return False
        try:
            parent, key = self._navigate(parts)
            val = parent.get(key)
            return val is not None and not isinstance(val, dict)
        except (FileNotFoundError, NotADirectoryError):
            return False

    def listdir(self, path: str) -> List[str]:
        parts = self._split(path)
        node = self._tree
        for part in parts:
            child = node.get(part)
            if child is None:
                raise FileNotFoundError(path)
            if not isinstance(child, dict):
                raise NotADirectoryError(path)
            node = child
        return sorted(node.keys())

    def readtext(self, path: str) -> str:
        parts = self._split(path)
        if not parts:
            raise IsADirectoryError(path)
        parent, key = self._navigate(parts)
        val = parent.get(key)
        if val is None:
            raise FileNotFoundError(path)
        if isinstance(val, dict):
            raise IsADirectoryError(path)
        return val

    def writetext(self, path: str, content: str, create_parents: bool = False) -> None:
        parts = self._split(path)
        if not parts:
            raise IsADirectoryError(path)
        parent, key = self._navigate(parts, create_parents=create_parents)
        parent[key] = content
        meta = self._meta.get(path)
        if meta is None:
            meta = FileMeta()
            self._meta[path] = meta
        meta.size = len(content)

    def makedirs(self, path: str) -> None:
        parts = self._split(path)
        if not parts:
            return
        node = self._tree
        for i, part in enumerate(parts):
            child = node.get(part)
            if child is None:
                node[part] = {}
                dir_path = "/".join(parts[: i + 1])
                self._meta[dir_path] = FileMeta(permissions="755", size=4096)
                node = node[part]
            elif isinstance(child, dict):
                node = child
            else:
                raise NotADirectoryError("/".join(parts[: i + 1]))

    def mkdir(self, path: str) -> None:
        """Create a single directory (parent must exist)."""
        parts = self._split(path)
        if not parts:
            return
        parent, key = self._navigate(parts)
        if key in parent:
            raise FileExistsError(path)
        parent[key] = {}
        self._meta[path] = FileMeta(permissions="755", size=4096)

    def remove(self, path: str) -> None:
        parts = self._split(path)
        if not parts:
            raise PermissionError("cannot remove root")
        parent, key = self._navigate(parts)
        if key not in parent:
            raise FileNotFoundError(path)
        val = parent[key]
        if isinstance(val, dict) and val:
            raise OSError(f"directory not empty: {path}")
        del parent[key]
        self._meta.pop(path, None)

    def remove_recursive(self, path: str) -> None:
        parts = self._split(path)
        if not parts:
            self._tree.clear()
            self._meta.clear()
            return
        parent, key = self._navigate(parts)
        if key not in parent:
            raise FileNotFoundError(path)
        # Remove all metadata under this path
        prefix = path.rstrip("/") + "/"
        to_del = [k for k in self._meta if k == path or k.startswith(prefix)]
        for k in to_del:
            del self._meta[k]
        del parent[key]

    def get_meta(self, path: str) -> FileMeta:
        parts = self._split(path)
        canon = "/".join(parts) if parts else ""
        meta = self._meta.get(canon)
        if meta is None:
            # Auto-create metadata
            meta = FileMeta()
            if self.isdir(path):
                meta.permissions = "755"
                meta.size = 4096
            else:
                try:
                    content = self.readtext(path)
                    meta.size = len(content)
                except Exception:
                    pass
            self._meta[canon] = meta
        return meta


    def walk(self, path: str = "") -> List[Tuple[str, List[str], List[str]]]:
        """Walk filesystem tree. Yields (dir_path, [subdirs], [files])."""
        parts = self._split(path)
        node = self._tree
        for part in parts:
            child = node.get(part)
            if child is None or not isinstance(child, dict):
                return []
            node = child

        result = []
        self._walk_recursive(node, path, result)
        return result

    def _walk_recursive(self, node: dict, prefix: str, result: list):
        dirs = []
        files = []
        for name, val in sorted(node.items()):
            if isinstance(val, dict):
                dirs.append(name)
            else:
                files.append(name)
        result.append((prefix, dirs, files))
        for d in dirs:
            child_prefix = f"{prefix}/{d}" if prefix else d
            self._walk_recursive(node[d], child_prefix, result)

    def copy_file(self, src: str, dst: str) -> None:
        """Copy a single file."""
        content = self.readtext(src)
        src_meta = self.get_meta(src)
        self.writetext(dst, content, create_parents=False)
        dst_meta = self.get_meta(dst)
        dst_meta.permissions = src_meta.permissions
        dst_meta.owner = src_meta.owner
        dst_meta.group = src_meta.group
        dst_meta.archive_type = src_meta.archive_type
        dst_meta.archive_entries = (
            dict(src_meta.archive_entries) if src_meta.archive_entries else None
        )

    def copy_tree(self, src: str, dst: str) -> None:
        """Recursively copy directory."""
        if self.isfile(src):
            self.copy_file(src, dst)
            return
        self.makedirs(dst)
        for name in self.listdir(src):
            s = f"{src}/{name}" if src else name
            d = f"{dst}/{name}" if dst else name
            if self.isdir(s):
                self.copy_tree(s, d)
            else:
                self.copy_file(s, d)


# ---------------------------------------------------------------------------
# Machine State
# ---------------------------------------------------------------------------

@dataclass
class MachineState:
    """State of a single virtual machine."""
    os_type: OSType
    hostname: str
    username: str
    home_dir: str       # canonical memfs path, e.g. "home/user" or "C/Users/user"
    cwd: str            # canonical memfs path
    fs: VirtualFS = field(default_factory=VirtualFS)
    env_vars: Dict[str, str] = field(default_factory=dict)

    # -- path conversion ------------------------------------------------

    def to_memfs_path(self, user_path: str) -> str:
        """Convert a user-visible path to canonical MemFS path (no leading /)."""
        if self.os_type == OSType.WINDOWS:
            return self._win_to_memfs(user_path)
        return self._unix_to_memfs(user_path)

    def to_display_path(self, memfs_path: str) -> str:
        """Convert canonical MemFS path to user-visible path."""
        if self.os_type == OSType.WINDOWS:
            return self._memfs_to_win(memfs_path)
        return "/" + memfs_path if memfs_path else "/"

    def _win_to_memfs(self, p: str) -> str:
        p = p.replace("/", "\\")
        # Absolute: C:\... -> C/...
        if len(p) >= 2 and p[1] == ":":
            drive = p[0].upper()
            rest = p[2:].lstrip("\\")
            parts = [drive] + [x for x in rest.split("\\") if x]
            return "/".join(parts)
        # Relative
        parts = [x for x in p.split("\\") if x]
        if not parts:
            return self.cwd
        return self.cwd + "/" + "/".join(parts)

    def _memfs_to_win(self, p: str) -> str:
        parts = p.split("/")
        if parts and len(parts[0]) == 1 and parts[0].isalpha():
            drive = parts[0].upper()
            rest = "\\".join(parts[1:])
            return f"{drive}:\\{rest}" if rest else f"{drive}:\\"
        return "\\".join(parts)

    def _unix_to_memfs(self, p: str) -> str:
        if p.startswith("/"):
            return p.lstrip("/")
        # Relative
        if not p or p == ".":
            return self.cwd
        if p == "..":
            parts = self.cwd.split("/")
            return "/".join(parts[:-1]) if len(parts) > 1 else ""
        return f"{self.cwd}/{p}" if self.cwd else p

    def resolve_path(self, user_path: str) -> str:
        """Resolve a user-visible path to canonical MemFS path, handling . and .."""
        raw = self.to_memfs_path(user_path)
        parts = raw.split("/")
        resolved = []
        for p in parts:
            if p == "" or p == ".":
                continue
            elif p == "..":
                if resolved:
                    resolved.pop()
            else:
                resolved.append(p)
        return "/".join(resolved)

    def get_display_cwd(self) -> str:
        """CWD for display in terminal prompt, with ~ substitution (Unix only)."""
        display = self.to_display_path(self.cwd)
        if self.os_type == OSType.WINDOWS:
            return display
        home_display = self.to_display_path(self.home_dir)
        if display == home_display:
            return "~"
        if display.startswith(home_display + "/"):
            return "~" + display[len(home_display):]
        return display

    def get_prompt_string(self) -> str:
        """Render the terminal prompt."""
        if self.os_type == OSType.WINDOWS:
            return f"{self.to_display_path(self.cwd)}>"
        elif self.os_type == OSType.MAC:
            return f"{self.username}@MacBook {self.get_display_cwd()} % "
        else:
            return f"{self.username}@{self.hostname}:{self.get_display_cwd()}$ "


# ---------------------------------------------------------------------------
# Terminal Environment
# ---------------------------------------------------------------------------

class TerminalEnv:
    """Complete terminal environment with local and remote machines.

    The *command_handlers* dict is injected by the caller (see commands.py).
    """

    def __init__(
        self,
        local: MachineState,
        remote: MachineState,
        command_handlers: Dict[str, Any],
        max_steps: int = 15,
    ):
        self.local = local
        self.remote = remote
        self.command_handlers = command_handlers
        self.max_steps = max_steps
        self.current_machine: MachineState = self.local
        self.ssh_connected: bool = False
        self.step_count: int = 0
        self.command_history: List[str] = []

    # -- prompt ----------------------------------------------------------

    def get_prompt_string(self) -> str:
        return self.current_machine.get_prompt_string()

    # -- execution -------------------------------------------------------

    def step(self, raw_command: str, goal_check_fn=None):
        """Gym-style step. Returns (observation, reward, done, info).

        Args:
            raw_command: The command string to execute.
            goal_check_fn: Optional callable ``fn(env) -> float`` that returns
                1.0 if the task is complete, 0.0 otherwise. When it returns
                1.0, the episode terminates early.
        """
        observation = self.execute(raw_command)
        reward = 0.0
        done = False
        info = {}

        if goal_check_fn is not None:
            reward = goal_check_fn(self)
            if reward > 0.5:
                done = True
                info["early_completion"] = True

        if self.step_count >= self.max_steps:
            done = True

        return observation, reward, done, info

    def execute(self, raw_command: str) -> str:
        """Execute a raw command string. Returns terminal output."""
        raw_command = raw_command.strip()
        if not raw_command:
            return ""

        self.command_history.append(raw_command)
        self.step_count += 1

        # Expand environment variables
        expanded = self._expand_env_vars(raw_command)

        # Check for unsupported operators
        if _contains_unquoted(expanded, "|"):
            return "Error: pipes (|) are not supported in this environment."
        if _contains_unquoted(expanded, ">") or _contains_unquoted(expanded, "<"):
            return "Error: redirects (>, >>, <) are not supported in this environment."

        # Split on && and ;
        segments = _split_command_chain(expanded)

        output_parts = []
        last_was_error = False
        prev_operator = None
        for cmd_str, operator in segments:
            cmd_str = cmd_str.strip()
            if not cmd_str:
                prev_operator = operator
                continue
            # && : skip if previous command failed
            if prev_operator == "&&" and last_was_error:
                break
            result, last_was_error = self._execute_single(cmd_str)
            if result:
                output_parts.append(result)
            prev_operator = operator

        return "\n".join(output_parts)

    def _execute_single(self, cmd_str: str) -> Tuple[str, bool]:
        """Execute one command. Returns (output, is_error)."""
        try:
            posix = self.current_machine.os_type != OSType.WINDOWS or self.ssh_connected
            tokens = shlex.split(cmd_str, posix=posix)
        except ValueError:
            return "bash: syntax error: unexpected end of file", True

        if not tokens:
            return "", False

        cmd_name = tokens[0]
        args = tokens[1:]

        # Resolve aliases
        cmd_key = self._resolve_alias(cmd_name.lower())

        handler = self.command_handlers.get(cmd_key)
        if handler is None:
            return self._unknown_cmd_error(cmd_name), True

        # Check OS availability
        if not handler.is_available(self.current_machine.os_type, self.ssh_connected):
            return self._unknown_cmd_error(cmd_name), True

        try:
            result = handler.execute(args, self)
            # Detect error by common patterns: "cmd: error...", "Error:", "bash:",
            # Windows errors, or empty success
            is_err = bool(result) and (
                result.startswith(("Error:", "bash:", "'", "The system", "Could Not"))
                or (": " in result.split("\n")[0]
                    and result.split(":")[0].split()[-1].lower() in (
                        "ls", "cd", "cat", "mkdir", "chmod", "cp", "mv", "rm",
                        "scp", "rsync", "tar", "zip", "unzip", "gzip", "gunzip",
                        "ssh", "touch", "del",
                    ))
            )
            return result, is_err
        except Exception as e:
            return f"Error: {e}", True

    def _expand_env_vars(self, cmd: str) -> str:
        machine = self.current_machine
        if machine.os_type == OSType.WINDOWS and not self.ssh_connected:
            for var, val in machine.env_vars.items():
                cmd = cmd.replace(f"%{var}%", val)
        else:
            # ~ expansion: only standalone ~ or ~/... at token boundaries
            # Do NOT expand ~ after : (e.g., scp file user@host:~/dir)
            home = machine.env_vars.get("HOME", machine.to_display_path(machine.home_dir))
            cmd = re.sub(r"(?<![\\~\w:])~(?=/|\s|$)", home, cmd)
            if cmd.startswith("~/") or cmd == "~":
                cmd = home + cmd[1:]
            # $VAR and ${VAR}
            for var, val in sorted(machine.env_vars.items(), key=lambda x: -len(x[0])):
                cmd = cmd.replace(f"${{{var}}}", val)
                cmd = cmd.replace(f"${var}", val)
        return cmd

    def _resolve_alias(self, name: str) -> str:
        aliases = {
            "dir": "ls",
            "type": "cat",
            "copy": "cp",
            "move": "mv",
            "del": "rm",
            "ren": "mv",
            "rename": "mv",
        }
        return aliases.get(name, name)

    def _unknown_cmd_error(self, cmd_name: str) -> str:
        if self.current_machine.os_type == OSType.WINDOWS and not self.ssh_connected:
            return (
                f"'{cmd_name}' is not recognized as an internal or external command,\n"
                f"operable program or batch file."
            )
        return f"bash: {cmd_name}: command not found"


# ---------------------------------------------------------------------------
# Helpers for command chain parsing
# ---------------------------------------------------------------------------

def _contains_unquoted(text: str, char: str) -> bool:
    """Check if *char* appears outside of quotes."""
    in_single = False
    in_double = False
    i = 0
    while i < len(text):
        c = text[i]
        if c == "'" and not in_double:
            in_single = not in_single
        elif c == '"' and not in_single:
            in_double = not in_double
        elif c == char and not in_single and not in_double:
            # For &&, check double char
            if char == "&" and i + 1 < len(text) and text[i + 1] == "&":
                return False  # && is handled separately
            if char in (">", "<", "|"):
                return True
        i += 1
    return False


def _split_command_chain(text: str) -> List[Tuple[str, Optional[str]]]:
    """Split on ``&&`` and ``;``. Returns [(cmd, operator), ...]."""
    result = []
    current = []
    in_single = False
    in_double = False
    i = 0
    while i < len(text):
        c = text[i]
        if c == "'" and not in_double:
            in_single = not in_single
            current.append(c)
        elif c == '"' and not in_single:
            in_double = not in_double
            current.append(c)
        elif not in_single and not in_double:
            if c == ";" :
                result.append(("".join(current), ";"))
                current = []
            elif c == "&" and i + 1 < len(text) and text[i + 1] == "&":
                result.append(("".join(current), "&&"))
                current = []
                i += 1  # skip second &
            else:
                current.append(c)
        else:
            current.append(c)
        i += 1

    remaining = "".join(current).strip()
    if remaining:
        result.append((remaining, None))
    return result
