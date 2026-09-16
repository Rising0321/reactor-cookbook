import subprocess
import sys

from benchmark_guard import belongs_to, stop_owned


def test_stop_owned_does_not_stop_an_unrelated_process():
    owned = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert belongs_to(owned.pid, owned.pid)
        assert not belongs_to(unrelated.pid, owned.pid)
        stop_owned(owned)
        assert owned.poll() is not None
        assert unrelated.poll() is None
    finally:
        for process in (owned, unrelated):
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=5)
