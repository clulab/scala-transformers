import time

from datetime import timedelta
from typing import Callable

__all__ = ["CluTimer"]

class CluTimer:
    @classmethod
    def time(cls, func: Callable[[], None]) -> None:
        start_time = time.monotonic()
        try:
            func()
        finally:
            end_time = time.monotonic()
            print(f"Elapsed training time: {timedelta(seconds=end_time - start_time)}")
