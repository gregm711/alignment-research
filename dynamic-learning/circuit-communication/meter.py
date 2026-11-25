"""Measure in-process CPU energy and elapsed time around a callable."""
import ctypes
from functools import lru_cache
from pathlib import Path
import platform
import subprocess
import tempfile
import time


@lru_cache(maxsize=1)
def _counter():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return None
    source = Path(__file__).with_name("task_energy.c")
    with tempfile.TemporaryDirectory() as temp:
        binary = Path(temp) / "task_energy.dylib"
        try:
            subprocess.run(["clang", "-dynamiclib", str(source), "-o", str(binary)],
                           check=True, capture_output=True)
            read = ctypes.CDLL(str(binary)).task_energy_nj
        except (OSError, subprocess.CalledProcessError):
            return None
    read.restype = ctypes.c_int64
    return read if read() >= 0 else None


def measure(fn):
    read = _counter()
    start_nj = read() if read else -1
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    result = fn()
    cpu = time.process_time() - start_cpu
    wall = time.perf_counter() - start_wall
    end_nj = read() if read else -1
    joules = (end_nj - start_nj) / 1e9 if end_nj > start_nj >= 0 else None
    return result, dict(cpu_joules=joules, wall_seconds=wall, cpu_seconds=cpu)


if __name__ == "__main__":
    result, cost = measure(lambda: sum(range(5_000_000)))
    assert result == 4_999_999 * 5_000_000 // 2
    assert cost["wall_seconds"] > 0 and cost["cpu_seconds"] > 0
    print(cost)
