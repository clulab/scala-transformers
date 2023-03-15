import time

from datetime import timedelta
from typing import Callable

class Timer:

    @classmethod
    def time(cls, func: Callable[[], None]) -> None:
        start_time = time.monotonic()
        func()
        end_time = time.monotonic()
        print(f"Elapsed training time: {timedelta(seconds=end_time - start_time)}")
