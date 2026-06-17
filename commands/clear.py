"""/clear — start a fresh conversation thread."""

import time


def run(**kwargs) -> str:
    new_thread_id = f"session_{int(time.time() * 1000)}"
    return f"__NEW_THREAD__{new_thread_id}"   # sentinel, parsed by main.py