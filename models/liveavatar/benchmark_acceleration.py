"""Explicit-input GPU benchmark, including decoding and RGB8 delivery."""

import argparse
import json
import subprocess
import time
from pathlib import Path

import imageio.v2 as imageio
import torch

from liveavatar_assets import configure_cache_environment
from liveavatar_backend import LiveAvatarBackend


def main(args):
    configure_cache_environment()
    args.output.mkdir(parents=True, exist_ok=True)
    driving = args.output / "driving.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(args.audio),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(driving),
        ],
        check=True,
    )
    started = time.perf_counter()
    backend = LiveAvatarBackend()
    report = {
        "load_seconds": time.perf_counter() - started,
        "runs": [],
        "fa4": args.fa4,
    }
    counts = None
    if args.fa4:
        from liveavatar_acceleration import install_fa4

        counts = install_fa4()
    events = []

    def instrument(owner, name, label):
        original = getattr(owner, name)

        def call(*a, **kw):
            begin, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            begin.record()
            result = original(*a, **kw)
            end.record()
            events.append((label, begin, end))
            return result

        setattr(owner, name, call)

    instrument(backend.pipeline.noise_model, "forward", "dit")
    instrument(backend.pipeline.vae, "decode", "vae_decode")
    instrument(backend.pipeline.vae, "encode", "vae_encode")
    try:
        for run in range(args.runs):
            frames = []
            backend.start(
                image=args.image,
                audio=driving,
                pose=None,
                prompt=args.prompt,
                negative_prompt="",
                seed=420,
                max_chunks=args.chunks,
            )
            previous = time.perf_counter()
            for index in range(args.chunks):
                result = backend.next()
                if result is None:
                    break
                video, _audio = result
                torch.cuda.synchronize()
                now = time.perf_counter()
                stages = {}
                for label, begin, end in events:
                    stages[label] = (
                        stages.get(label, 0) + begin.elapsed_time(end) / 1000
                    )
                events.clear()
                row = {
                    "run": run,
                    "chunk": index,
                    "frames": len(video),
                    "wall_seconds": now - previous,
                    "realtime_x": len(video) / 25 / (now - previous),
                    "stages": stages,
                }
                report["runs"].append(row)
                print("BENCH " + json.dumps(row), flush=True)
                frames.extend(video)
                previous = time.perf_counter()
            backend.close()
            imageio.mimwrite(
                args.output / f"run-{run}.mp4", frames, fps=25, macro_block_size=1
            )
            report["attention_calls"] = counts
            (args.output / "report.json").write_text(json.dumps(report, indent=2))
    finally:
        backend.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prompt", default="A fashion blogger speaks naturally to the camera."
    )
    parser.add_argument("--chunks", type=int, default=3)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--fa4", action="store_true")
    main(parser.parse_args())
