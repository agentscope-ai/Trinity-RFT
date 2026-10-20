import sqlite3

import pytest

from trinity.common.workflows.envs.sqlite_sql.sqlite_sql_env import (
    SQLiteSQLEnv,
    db_file,
    execute_sql,
    parse_response,
    schema_text,
)

GOLD = "SELECT name FROM customers WHERE city = 'Hangzhou'"


@pytest.fixture()
def db_path(tmp_path):
    path = db_file(str(tmp_path), "shop")
    path.parent.mkdir()
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE customers(id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, city TEXT);
        INSERT INTO customers(name, city) VALUES ('Ann','Hangzhou'),('Bob','Shanghai'),('Cat','Hangzhou');
        """
    )
    conn.commit()
    conn.close()
    return path


def test_parse_response():
    assert parse_response("<think>x</think><action>SELECT 1;</action>") == ("action", "SELECT 1")
    assert parse_response("<answer>\n```sql\nSELECT 2\n```\n</answer>") == ("answer", "SELECT 2")
    assert parse_response("<action>SELECT 1</action> then <answer>SELECT 3</answer>") == (
        "answer",
        "SELECT 3",
    )
    # tags inside the thinking section are ignored
    assert parse_response("<think><answer>SELECT 9</answer></think>") == (None, "")
    assert parse_response("SELECT 1") == (None, "")


def test_schema_text_skips_internal_tables(db_path):
    schema = schema_text(db_path)
    assert "CREATE TABLE customers" in schema
    assert "sqlite_sequence" not in schema


def test_explore_then_answer(db_path):
    env = SQLiteSQLEnv(db_path, GOLD, max_obs_rows=2)

    obs = env.step("<action>SELECT name, city FROM customers ORDER BY id</action>")
    assert not obs.done and obs.error is None
    assert obs.observation.splitlines()[0] == "name | city"
    assert "Ann | Hangzhou" in obs.observation and "Cat" not in obs.observation
    assert "(3 rows in total)" in obs.observation

    result = env.step("<answer>SELECT name FROM customers WHERE city LIKE 'Hang%'</answer>")
    assert result.done and result.reward == 1.0
    assert env.submitted and env.num_queries == 1


def test_wrong_answer(db_path):
    result = SQLiteSQLEnv(db_path, GOLD).step("<answer>SELECT name FROM customers</answer>")
    assert result.done and result.reward == 0.0


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM customers",
        "UPDATE customers SET city = 'Hangzhou'",
        "DROP TABLE customers",
        "ATTACH DATABASE ':memory:' AS other",
        "PRAGMA table_info(customers)",
        "SELECT 1; DELETE FROM customers",
    ],
)
def test_writes_are_rejected_and_do_not_reach_the_database(db_path, sql):
    env = SQLiteSQLEnv(db_path, GOLD)
    obs = env.step(f"<action>{sql}</action>")
    assert obs.observation.startswith("Error:") and not obs.done
    assert env.num_errors == 1
    rows, _, _ = execute_sql(db_path, "SELECT COUNT(*) FROM customers")
    assert rows == [(3,)]


def test_runaway_query_is_interrupted(db_path):
    env = SQLiteSQLEnv(db_path, GOLD, timeout=0.5)
    obs = env.step(
        "<action>WITH RECURSIVE r(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM r) "
        "SELECT COUNT(*) FROM r</action>"
    )
    assert obs.observation == "Error: timeout"


def test_invalid_format_keeps_episode_open(db_path):
    obs = SQLiteSQLEnv(db_path, GOLD).step("I think the answer is Ann.")
    assert not obs.done and obs.error == "no action"


def test_broken_or_empty_reference(db_path):
    broken = SQLiteSQLEnv(db_path, "SELECT nope FROM customers")
    assert broken.gold_error is not None
    assert broken.step(f"<answer>{GOLD}</answer>").reward == 0.0

    empty = SQLiteSQLEnv(db_path, "SELECT name FROM customers WHERE city = 'Beijing'")
    assert empty.gold_empty
