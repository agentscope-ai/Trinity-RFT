# -*- coding: utf-8 -*-
"""A Docker-free, read-only SQLite environment for multi-turn text-to-SQL (BIRD / Spider layout).

The agent may run exploratory queries and then submits one final query. The reward is BIRD / Spider
execution accuracy: 1.0 when the submitted query returns the same set of rows as the reference.

Every rollout opens its own read-only connection behind an SQLite authorizer that only allows reads,
so concurrent rollouts on the same database file cannot write to it, ATTACH other files, run
PRAGMAs or load extensions, and a runaway query is interrupted after ``timeout`` seconds.
"""

import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

_ALLOWED_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    sqlite3.SQLITE_RECURSIVE,
}
_DENIED_FUNCTIONS = {"load_extension", "readfile", "writefile", "edit", "fts3_tokenizer"}
_TAG_RE = re.compile(r"<(action|answer)>(.*?)</\1>", re.IGNORECASE | re.DOTALL)
_FENCE_RE = re.compile(r"^```[ \t]*(?:sql|sqlite)?[ \t]*\n?(.*?)```$", re.IGNORECASE | re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _authorizer(action, arg1, arg2, db_name, trigger):
    if action not in _ALLOWED_ACTIONS:
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_FUNCTION and arg2 and arg2.lower() in _DENIED_FUNCTIONS:
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def db_file(db_root: str, db_id: str) -> Path:
    """``<db_root>/<db_id>/<db_id>.sqlite``, the BIRD and Spider database layout."""
    return Path(db_root) / db_id / f"{db_id}.sqlite"


def execute_sql(
    path: Path, sql: str, timeout: float = 30.0, max_rows: int = 100_000
) -> Tuple[Optional[List[Tuple]], Optional[List[str]], Optional[str]]:
    """Run one read-only statement. Returns ``(rows, column_names, None)`` or ``(None, None, error)``."""
    sql = (sql or "").strip().rstrip(";").strip()
    if not sql:
        return None, None, "empty query"
    if not Path(path).is_file():
        return None, None, f"database not found: {path}"
    deadline = time.monotonic() + timeout
    conn = None
    try:
        conn = sqlite3.connect(
            f"{Path(path).resolve().as_uri()}?mode=ro", uri=True, check_same_thread=False
        )
        conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
        conn.set_authorizer(_authorizer)
        conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 1000)
        cursor = conn.execute(sql)
        rows = cursor.fetchmany(max_rows + 1)
        if len(rows) > max_rows:
            return None, None, f"result exceeds {max_rows} rows"
        columns = [d[0] for d in cursor.description or []]
        return [tuple(r) for r in rows], columns, None
    except sqlite3.OperationalError as e:
        return None, None, "timeout" if "interrupted" in str(e) else f"OperationalError: {e}"
    except Exception as e:  # sqlite3.Warning (several statements), DatabaseError, ...
        return None, None, f"{type(e).__name__}: {e}"
    finally:
        if conn is not None:
            conn.close()


def schema_text(path: Path) -> str:
    """The CREATE TABLE statements of every user table."""
    rows, _, error = execute_sql(path, "SELECT name, sql FROM sqlite_master WHERE type = 'table'")
    if error:
        raise ValueError(f"cannot read schema of {path}: {error}")
    return "\n\n".join(sql for name, sql in rows if sql and name != "sqlite_sequence")


def parse_response(text: str) -> Tuple[Optional[str], str]:
    """Return ``(kind, sql)`` for the last ``<action>`` or ``<answer>`` tag outside ``<think>``."""
    matches = _TAG_RE.findall(_THINK_RE.sub("", text or ""))
    if not matches:
        return None, ""
    kind, body = matches[-1]
    body = body.strip()
    fenced = _FENCE_RE.match(body)
    if fenced:
        body = fenced.group(1)
    return kind.lower(), body.strip().rstrip(";").strip()


def format_result(rows: List[Tuple], columns: List[str], max_rows: int) -> str:
    shown = rows[:max_rows]
    lines = [" | ".join(columns)] + [" | ".join(str(v) for v in row) for row in shown]
    more = f"\n... ({len(rows)} rows in total)" if len(rows) > max_rows else f"\n({len(rows)} rows)"
    return "\n".join(lines) + more


@dataclass
class StepResult:
    observation: str
    done: bool
    reward: float
    error: Optional[str] = None


class SQLiteSQLEnv:
    """One episode on one database. ``step`` takes a model response and returns a StepResult."""

    def __init__(
        self,
        db_path: Path,
        gold_sql: str,
        timeout: float = 30.0,
        max_obs_rows: int = 10,
        max_rows: int = 100_000,
    ):
        self.db_path = Path(db_path)
        self.timeout = timeout
        self.max_obs_rows = max_obs_rows
        self.max_rows = max_rows
        self.gold_rows, _, self.gold_error = execute_sql(self.db_path, gold_sql, timeout, max_rows)
        self.num_queries = 0
        self.num_errors = 0
        self.submitted = False

    @property
    def gold_empty(self) -> bool:
        return self.gold_error is None and not self.gold_rows

    def score(self, sql: str) -> Tuple[float, Optional[str]]:
        rows, _, error = execute_sql(self.db_path, sql, self.timeout, self.max_rows)
        if error or self.gold_error:
            return 0.0, error
        return float(set(rows) == set(self.gold_rows)), None

    def step(self, response_text: str) -> StepResult:
        kind, sql = parse_response(response_text)
        if kind is None:
            return StepResult(
                "Invalid format. Reply with one <action>SQL</action> to run a query, "
                "or <answer>SQL</answer> to submit your final query.",
                False,
                0.0,
                "no action",
            )
        if kind == "answer":
            self.submitted = True
            reward, error = self.score(sql)
            return StepResult("Submitted.", True, reward, error)
        self.num_queries += 1
        rows, columns, error = execute_sql(self.db_path, sql, self.timeout, self.max_rows)
        if error:
            self.num_errors += 1
            return StepResult(f"Error: {error}", False, 0.0, error)
        return StepResult(format_result(rows, columns, self.max_obs_rows), False, 0.0)
