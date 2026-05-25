import datetime
import sys
from unittest.mock import MagicMock

import pytest

# naive_dapo_score imports posix-only `resource`; mock on Windows before collection.
if sys.platform == "win32" and "resource" not in sys.modules:
    sys.modules["resource"] = MagicMock()


# Get the result of each test
@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    setattr(item, "rep_" + rep.when, rep)


# Real-time print of start and end of test
@pytest.fixture(autouse=True)
def log_test_lifecycle(request):
    node_id = request.node.nodeid
    start_time = datetime.datetime.now().strftime("%H:%M:%S")

    print(f"\n[START] {start_time} - Running: {node_id}")

    yield

    end_time = datetime.datetime.now().strftime("%H:%M:%S")
    # Get the result of each test (setup, call, teardown)
    report = getattr(request.node, "rep_call", None)

    if report:
        if report.passed:
            status = "PASSED"
        elif report.failed:
            status = "FAILED"
        else:
            status = report.outcome.upper()
    else:
        status = "UNKNOWN"

    print(f"\n[END] {end_time} - Result: {status} - {node_id}")
