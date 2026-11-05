"""Atomic campaign writes that tolerate brief Windows sync-client locks."""
import time


def replace_with_retry(source, destination):
    for attempt in range(20):
        try:
            source.replace(destination)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(min(.025 * (attempt + 1), .2))
