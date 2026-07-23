import os
import time


def ensure_dir(*paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)


def timer():
    return time.perf_counter()


def elapsed(start):
    return round(time.perf_counter() - start, 3)