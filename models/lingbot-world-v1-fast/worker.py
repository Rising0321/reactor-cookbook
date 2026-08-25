"""Serve upstream LingBot-World-Fast through a small JSON-line protocol."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

_RESPONSE_PREFIX = "REACTOR_LINGBOT_V1_RESPONSE "


class LingBotRuntime:
    """Own one loaded Fast model and one optional causal rollout."""

    def __init__(self, settings: dict[str, Any]) -> None:
        import torch
        import wan
        from wan.configs import WAN_CONFIGS
        from wan.interactive_fast import InteractiveFastRollout

        self._torch = torch
        self._runtime_root = Path(settings["runtime_root"]).resolve()
        self._runtime_root.mkdir(parents=True, exist_ok=True)
        self._counter = 0
        config = WAN_CONFIGS["i2v-A14B"]
        context_latents = int(settings["context_latents"])
        checkpoint_dir = Path(settings["checkpoint_dir"]).resolve()
        if "cam" not in checkpoint_dir.name:
            raise ValueError(
                "LingBot Fast camera checkpoints must use a path containing 'cam'"
            )
        torch.cuda.set_device(0)
        self._pipe = wan.WanI2VFast(
            config=config,
            checkpoint_dir=str(checkpoint_dir),
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=False,
            init_on_cpu=False,
            convert_model_dtype=False,
            pipe_dtype=torch.bfloat16,
            local_attn_size=context_latents,
            sink_size=0,
        )
        self._rollout = InteractiveFastRollout(
            self._pipe,
            max_chunks=int(settings["max_chunks"]),
            context_latents=context_latents,
            chunk_size=3,
            max_area=int(settings["max_area"]),
            shift=float(settings["shift"]),
        )
        self._active = False

    def reset(
        self, image_path: Path, intrinsics_path: Path, prompt: str, seed: int
    ) -> None:
        """Start a fresh image-conditioned causal rollout."""
        if not image_path.is_file():
            raise FileNotFoundError(
                f"LingBot anchor image does not exist: {image_path}"
            )
        if not intrinsics_path.is_file():
            raise FileNotFoundError(
                f"LingBot intrinsics do not exist: {intrinsics_path}"
            )
        with Image.open(image_path) as image:
            anchor = image.convert("RGB")
            intrinsics = np.load(intrinsics_path, allow_pickle=False)
            self._rollout.reset(prompt, anchor, intrinsics, seed)
        self._active = True

    def generate(self, relative_c2ws: np.ndarray, prompt: str) -> tuple[Path, int]:
        """Generate one chunk and persist its uint8 RGB frames for the parent."""
        if not self._active:
            raise RuntimeError("reset LingBot before generating a chunk")
        video = self._rollout.generate_chunk(relative_c2ws, prompt)
        frames = (
            video.permute(1, 2, 3, 0)
            .add(1.0)
            .mul(127.5)
            .clamp(0, 255)
            .to(self._torch.uint8)
            .cpu()
            .numpy()
        )
        frames = np.ascontiguousarray(frames)
        expected = 9 if self._rollout.chunk_index == 1 else 12
        if frames.shape[0] != expected:
            raise RuntimeError(
                f"LingBot causal VAE decoded {frames.shape[0]} frames; expected {expected}"
            )
        self._counter += 1
        output = self._runtime_root / f"chunk_{self._counter:06d}.npy"
        np.save(output, frames, allow_pickle=False)
        return output, expected

    def end_session(self) -> None:
        """Release causal caches while preserving model weights."""
        self._rollout.end()
        self._active = False

    def close(self) -> None:
        """Release active rollout state."""
        self.end_session()


def _respond(request_id: int, *, ok: bool, **payload: object) -> None:
    """Write one correlated worker response."""
    print(
        _RESPONSE_PREFIX
        + json.dumps({"id": request_id, "ok": ok, **payload}, ensure_ascii=False),
        flush=True,
    )


def main() -> int:
    """Serve requests until the parent closes stdin or asks to stop."""
    runtime: LingBotRuntime | None = None
    for line in sys.stdin:
        request = json.loads(line)
        request_id = int(request["id"])
        command = str(request["command"])
        if command == "shutdown":
            if runtime is not None:
                runtime.close()
            _respond(request_id, ok=True)
            return 0
        try:
            if command == "initialize":
                runtime = LingBotRuntime(request)
                _respond(request_id, ok=True)
            elif command == "reset":
                if runtime is None:
                    raise RuntimeError("LingBot worker is not initialized")
                runtime.reset(
                    Path(request["anchor_image"]),
                    Path(request["intrinsics"]),
                    str(request["prompt"]),
                    int(request["seed"]),
                )
                _respond(request_id, ok=True)
            elif command == "generate":
                if runtime is None:
                    raise RuntimeError("LingBot worker is not initialized")
                output, frame_count = runtime.generate(
                    np.asarray(request["relative_c2ws"], dtype=np.float32),
                    str(request["prompt"]),
                )
                _respond(
                    request_id,
                    ok=True,
                    output=str(output),
                    frame_count=frame_count,
                )
            elif command == "end_session":
                if runtime is not None:
                    runtime.end_session()
                _respond(request_id, ok=True)
            else:
                raise ValueError(f"unknown LingBot worker command: {command}")
        except Exception as error:  # noqa: BLE001 - report model failures to the parent protocol
            traceback.print_exc()
            _respond(request_id, ok=False, error=f"{type(error).__name__}: {error}")
    if runtime is not None:
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
