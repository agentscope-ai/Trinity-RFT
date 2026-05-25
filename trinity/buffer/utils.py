import time
import traceback
from contextlib import contextmanager

from trinity.utils.log import get_logger


@contextmanager
def retry_session(session_maker, max_retry_times: int = 2, max_retry_interval: float = 1.0):
    """A Context manager for retrying session."""
    logger = get_logger(__name__)
    retries = max(1, int(max_retry_times))
    session = session_maker()

    try:
        yield session
    except StopIteration as e:
        raise e
    except Exception as e:
        # Exception raised inside with-body: rollback once and propagate.
        session.rollback()
        raise e
    else:
        last_exception = None
        for attempt in range(retries):
            try:
                session.commit()
                return
            except Exception as e:
                last_exception = e
                trace_str = traceback.format_exc()
                session.rollback()
                logger.warning(
                    f"Attempt {attempt + 1} failed, retrying in {max_retry_interval} seconds..."
                )
                logger.warning(f"trace = {trace_str}")
                if attempt < retries - 1:
                    time.sleep(max_retry_interval)
        logger.error("Max retry attempts reached, raising exception.")
        raise last_exception
    finally:
        session.close()
