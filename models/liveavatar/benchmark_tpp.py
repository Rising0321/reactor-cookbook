"""Benchmark the released four-stage denoising plus streaming VAE on five GPUs."""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import torch
import torch.distributed as dist
import yaml

from liveavatar_assets import SOURCE, configure_cache_environment, prepare_assets


def main(args):
    configure_cache_environment()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device(f"cuda:{rank}"))
    base, lora = prepare_assets()
    sys.path.insert(0, str(SOURCE))
    from liveavatar.models.wan.causal_s2v_pipeline_tpp import WanS2V
    from liveavatar.models.wan.wan_2_2.configs import WAN_CONFIGS

    world_size = dist.get_world_size()
    if args.shared_vae and not args.grouped:
        raise ValueError("--shared-vae requires --grouped")
    if args.grouped:
        from liveavatar_grouped_benchmark import install_grouped_generate

        print(
            "STAGE_GROUPS",
            install_grouped_generate(WanS2V, world_size, args.shared_vae),
            flush=True,
        )
    elif world_size != 5:
        raise ValueError("Use --grouped to benchmark fewer than five GPUs")
    output_rank = world_size - 1

    model = WanS2V(
        config=WAN_CONFIGS["s2v-14B"],
        checkpoint_dir=str(base),
        device_id=rank,
        rank=rank,
        single_gpu=False,
        sp_size=1,
        convert_model_dtype=True,
        init_on_cpu=False,
        offload_kv_cache=False,
    )
    settings = yaml.safe_load(
        (SOURCE / "liveavatar/configs/s2v_causal_sft.yaml").read_text()
    )
    model.noise_model = model.add_lora_to_model(
        model.noise_model,
        lora_rank=settings["lora_rank"],
        lora_alpha=settings["lora_alpha"],
        lora_target_modules=settings["lora_target_modules"],
        init_lora_weights=settings["init_lora_weights"],
        pretrained_lora_path=str(lora / "liveavatar.safetensors"),
        load_lora_weight_only=False,
    )
    if args.fa4:
        from liveavatar_acceleration import install_fa4

        counts = install_fa4()
    else:
        counts = None
    if args.vae_graph and rank == output_rank:
        from liveavatar_graph import install_vae_graph

        install_vae_graph(model.vae)
    args.output.mkdir(parents=True, exist_ok=True)
    for run in range(args.runs):
        random.seed(420)
        frames, rows = [], []
        model.vae.model.clear_cache()
        model.vae.model.first_decode = True
        model.vae.model.first_encode = True
        started = previous = time.perf_counter()

        def emit(clip, rows=rows, frames=frames, started=started):
            nonlocal previous
            video = ((clip.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
            video = video.permute(1, 2, 3, 0).contiguous().cpu().numpy()
            now = time.perf_counter()
            row = {
                "block": len(rows),
                "frames": len(video),
                "wall_seconds": now - previous,
                "elapsed_seconds": now - started,
                "realtime_x": len(video) / 25 / (now - previous),
            }
            rows.append(row)
            frames.extend(video)
            print("TPP_BENCH " + json.dumps(row), flush=True)
            previous = time.perf_counter()

        model.generate(
            input_prompt=args.prompt,
            ref_image_path=str(args.image),
            audio_path=str(args.audio),
            num_repeat=args.chunks,
            max_repeat=args.chunks,
            max_area=704 * 384,
            infer_frames=48,
            sampling_steps=4,
            sample_solver="euler",
            shift=3.0,
            guide_scale=0,
            seed=420,
            offload_model=False,
            num_gpus_dit=world_size if args.shared_vae else max(1, world_size - 1),
            enable_vae_parallel=True,
            chunk_callback=emit,
        )
        model.kv_cache1 = model.crossattn_cache = None
        if args.grouped:
            model._stage_kv = model._stage_cross = None
        if rank == output_rank:
            (args.output / f"run-{run}.json").write_text(
                json.dumps(
                    {
                        "blocks": rows,
                        "fa4_calls": counts,
                        "world_size": world_size,
                        "grouped": args.grouped,
                        "shared_vae": args.shared_vae or world_size == 1,
                        "image": str(args.image),
                        "audio": str(args.audio),
                        "prompt": args.prompt,
                        "sampling_steps": 4,
                        "fps": 25,
                    },
                    indent=2,
                )
            )
            imageio.mimwrite(
                args.output / f"run-{run}.mp4", frames, fps=25, macro_block_size=1
            )
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--prompt", default="A fashion blogger speaks naturally to the camera."
    )
    p.add_argument("--chunks", type=int, default=6)
    p.add_argument("--runs", type=int, default=2)
    p.add_argument("--fa4", action="store_true")
    p.add_argument("--vae-graph", action="store_true")
    p.add_argument("--grouped", action="store_true")
    p.add_argument("--shared-vae", action="store_true")
    main(p.parse_args())
