"""Bound one benchmark to ten minutes; stop its process group on foreign GPU use."""

import argparse
import os
import signal
import subprocess
import time
from pathlib import Path


def gpu_pids():
    result = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True,
    )
    return {int(row.strip()) for row in result.splitlines() if row.strip().isdigit()}


def belongs_to(pid, root):
    while pid > 1:
        if pid == root:
            return True
        try:
            status = Path(f"/proc/{pid}/status").read_text()
        except FileNotFoundError:
            return False
        pid = int(
            next(s.split()[1] for s in status.splitlines() if s.startswith("PPid:"))
        )
    return False


def stop_owned(child):
    # torchrun workers can have their own process groups. Snapshot verified
    # descendants before stopping the launcher, and retain pidfds against PID reuse.
    handles = []
    for path in Path("/proc").iterdir():
        if path.name.isdigit() and belongs_to(int(path.name), child.pid):
            try:
                fd = os.pidfd_open(int(path.name))
                if belongs_to(int(path.name), child.pid):
                    handles.append(fd)
                else:
                    os.close(fd)
            except ProcessLookupError:
                pass
    try:
        for fd in handles:
            try:
                signal.pidfd_send_signal(fd, signal.SIGTERM)
            except ProcessLookupError:
                pass
        time.sleep(3)
        for fd in handles:
            try:
                signal.pidfd_send_signal(fd, signal.SIGKILL)
            except ProcessLookupError:
                pass
        child.wait(timeout=2)
    finally:
        for fd in handles:
            os.close(fd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if gpu_pids():
        raise SystemExit("Foreign GPU processes present; benchmark not started")
    command = args.command[1:] if args.command[0] == "--" else args.command
    with args.log.open("x") as log:
        child = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        started = time.monotonic()
        try:
            while child.poll() is None:
                foreign = {
                    pid
                    for pid in gpu_pids()
                    if Path(f"/proc/{pid}").exists() and not belongs_to(pid, child.pid)
                }
                if foreign or time.monotonic() - started > 590:
                    raise RuntimeError(
                        f"Stopping owned benchmark: foreign={foreign}, elapsed={time.monotonic() - started:.1f}s"
                    )
                time.sleep(2)
        finally:
            if child.poll() is None:
                stop_owned(child)
        raise SystemExit(child.returncode)


if __name__ == "__main__":
    main()
