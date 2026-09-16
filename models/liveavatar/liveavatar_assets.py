"""NVMe asset locations and pinned upstream setup for stage-one debugging."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path("/opt/dlami/nvme")
SOURCE_REVISION = "c3c47d031d8bf3247428333c1fe610579c71a551"
BASE_REVISION = "dab4e9c55bbe4c8c4d03db1c2c98c7f0ac9c454b"
LORA_REVISION = "92cdccd12a91e8a63767a7c821b7c75e51d5a172"
SOURCE = Path(
    os.environ.get("LIVEAVATAR_SOURCE", ROOT / "ruixing/liveavatar-upstream-20260916")
)
WORK = ROOT / ".cache_hf/reactor_registry/liveavatar-stage1"


def configure_cache_environment() -> None:
    for name, value in {
        "UV_CACHE_DIR": ROOT / ".cache_uv",
        "UV_PYTHON_INSTALL_DIR": ROOT / ".cache_uv/python",
        "HF_HOME": ROOT / ".cache_hf",
        "HF_HUB_CACHE": ROOT / ".cache_hf/hub",
        "TRANSFORMERS_CACHE": ROOT / ".cache_hf/hub",
        "XDG_CACHE_HOME": ROOT / ".cache_hf/liveavatar-cache",
        "TORCH_HOME": ROOT / ".cache_hf/torch",
        "TMPDIR": WORK / "tmp",
        "TORCHINDUCTOR_CACHE_DIR": WORK / "inductor",
        "CUTE_DSL_CACHE_DIR": WORK / "cute",
        "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR": WORK / "fa4",
    }.items():
        os.environ[name] = str(value)
        value.mkdir(parents=True, exist_ok=True)
    os.environ["ENABLE_COMPILE"] = "false"


def prepare_assets() -> tuple[Path, Path]:
    configure_cache_environment()
    if not SOURCE.exists():
        subprocess.run(
            [
                "git",
                "clone",
                "https://github.com/Alibaba-Quark/LiveAvatar.git",
                str(SOURCE),
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(SOURCE), "checkout", "--detach", SOURCE_REVISION],
            check=True,
        )
    revision = subprocess.check_output(
        ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != SOURCE_REVISION:
        raise RuntimeError(
            f"Expected upstream {SOURCE_REVISION}, found {revision}; use a separate checkout"
        )
    patch = Path(__file__).with_name("liveavatar-streaming.patch")
    reverse = subprocess.run(
        ["git", "-C", str(SOURCE), "apply", "--reverse", "--check", str(patch)],
        capture_output=True,
        check=False,
    )
    if reverse.returncode:
        subprocess.run(
            ["git", "-C", str(SOURCE), "apply", "--check", str(patch)], check=True
        )
        subprocess.run(["git", "-C", str(SOURCE), "apply", str(patch)], check=True)
    if os.environ.get("LIVEAVATAR_MODE") == "tpp":
        patch = Path(__file__).with_name("liveavatar-tpp-streaming.patch")
        reverse = subprocess.run(
            ["git", "-C", str(SOURCE), "apply", "--reverse", "--check", str(patch)],
            capture_output=True,
            check=False,
        )
        if reverse.returncode:
            subprocess.run(
                ["git", "-C", str(SOURCE), "apply", "--check", str(patch)], check=True
            )
            subprocess.run(["git", "-C", str(SOURCE), "apply", str(patch)], check=True)
    from huggingface_hub import snapshot_download

    base = Path(snapshot_download("Wan-AI/Wan2.2-S2V-14B", revision=BASE_REVISION))
    lora = Path(snapshot_download("Quark-Vision/Live-Avatar", revision=LORA_REVISION))
    return base, lora


if __name__ == "__main__":
    print(prepare_assets())
