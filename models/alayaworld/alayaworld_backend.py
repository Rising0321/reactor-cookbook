"""Run stateful AlayaWorld v1.1 distilled chunks on the upstream model code."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np


@dataclass
class RolloutCache:
    """Keep the tensors and ViGeo state shared by consecutive chunks."""

    metadata: dict[str, Any]
    history: Any
    sink_latent: Any
    sink_indices: Any
    spatial_bank: Any
    decode_anchor: Any
    motion_nearby: Any
    context: Any
    prompt: str
    seed: int
    target_base_start: int
    round_index: int = 0


class AlayaWorldV11Backend:
    """Expose the upstream v1.1 DMD model as a stateful one-chunk backend."""

    def __init__(
        self,
        *,
        config: Any,
        modules: dict[str, Any],
        attention_function: Any | None,
        compact_cache: Any,
    ) -> None:
        self.config = config
        self.modules = modules
        self.torch = modules["torch"]
        self.compact_cache = compact_cache
        self.trainer = self._load_trainer(attention_function)
        self.mode = next(iter(self.trainer.cfg.validation.modes.values()))
        self.chunk_latents = int(self.mode.layout.output_latent_frames)
        self.history_latents = int(
            self.trainer.cfg.layout.history_latent_frames
            if self.mode.layout.history_latent_frames is None
            else self.mode.layout.history_latent_frames
        )
        self.condition_latents = int(self.mode.layout.condition_latent_frames)
        self.stride = int(self.trainer.cfg.sample.temporal_stride)
        self.fps = float(self.trainer.cfg.sample.fps)
        self.prefix_frames = int(
            self.trainer._vigeo_target_prefix_pixel_frames(
                history_latent_frames=self.history_latents
            )
        )
        self.frames_per_chunk = self.chunk_latents * self.stride
        self.cache: RolloutCache | None = None

    def _load_trainer(self, attention_function: Any | None) -> Any:
        """Load only inference components and omit DMD training replicas."""
        cfg = self.modules["load_config"](str(self.config.upstream_config))
        cfg.paths.base_transformer = str(self.config.base_model.path)
        cfg.paths.transformer = ""
        cfg.paths.continue_transformer = ""
        cfg.paths.model = ""
        cfg.paths.vae = str(self.config.base_model.path)
        cfg.paths.resume_checkpoint = str(self.config.ar_transformer.path.parent)
        cfg.paths.history_encoder = str(self.config.history_encoder.path)
        cfg.paths.dmd_resume = str(self.config.dmd_lora.path.parent)
        cfg.paths.gemma = str(self.config.gemma.path)
        cfg.spatial_memory.vigeo_repo_path = str(self.config.vigeo_source_path)
        cfg.spatial_memory.vigeo_checkpoint = str(
            self.config.vigeo_checkpoint.path.parent
        )
        cfg.dmd.enabled = False
        cfg.lora.train = False
        cfg.memory.train = False
        cfg.runtime.fsdp = False
        cfg.runtime.gradient_checkpointing = False
        cfg.runtime.vae_latent_cache_dir = None
        cfg.runtime.text_embed_cache_dir = None
        cfg.run.seed = int(self.config.seed)

        trainer = self.modules["rollout_trainer"](cfg)
        components = self.modules["build_model_components"](
            cfg,
            trainer.dist.device,
            trainer.dtype,
        )
        patchify_projection = components.transformer.patchify_proj
        history_encoder = self.modules["build_history_encoder"](
            cfg.memory,
            in_channels=patchify_projection.in_features,
            out_channels=patchify_projection.out_features,
            device=trainer.dist.device,
            dtype=trainer.dtype,
            checkpoint_path=cfg.paths.history_encoder,
        )
        if cfg.memory.use_lr_branch:
            history_encoder.setup_lr_proj_from_patchify(patchify_projection)
        self.modules["load_checkpoint_weights"](
            cfg=cfg,
            dist_state=trainer.dist,
            transformer=components.transformer,
            history_encoder=history_encoder,
            lora_manager=components.lora_manager,
        )
        components.transformer.eval()
        components.vae_decoder.eval()
        components.text_encoder.eval()
        history_encoder.eval()
        trainer.components = components
        trainer.history_encoder = history_encoder

        if attention_function is not None:
            engine = SimpleNamespace(
                transformer=components.transformer,
                text_encoder=components.text_encoder,
            )
            self.modules["set_attention_backend"](engine, attention_function)
        if self.config.compile_mode != "none":
            components.transformer = self.torch.compile(
                components.transformer,
                mode=self.config.compile_mode,
            )
        return trainer

    def reset(
        self,
        *,
        video_pixels: Any,
        metadata: dict[str, Any],
        prompt: str,
        seed: int,
    ) -> RolloutCache:
        """Initialize the v1.1 causal VAE, temporal memory, and ViGeo cache."""
        trainer = self.trainer
        metadata = dict(metadata)
        metadata["has_camera"] = True
        metadata["source"] = "custom_i2v"
        metadata.setdefault("video_id", "reactor-image")
        cameras = metadata["cam_c2w"]
        if cameras.dim() == 4:
            camera_frames = int(cameras.shape[1])
        else:
            camera_frames = int(cameras.shape[0])
        if camera_frames < self.prefix_frames:
            raise ValueError(
                f"camera template has {camera_frames} frames; v1.1 needs "
                f"at least {self.prefix_frames}"
            )
        video_pixels = video_pixels[: self.prefix_frames].contiguous()
        target_base_start = int(trainer.cfg.layout.sink_latent_frames) + int(
            self.history_latents
        )
        latent_full = trainer._build_vigeo_validation_latent_full(
            video_pixels=video_pixels,
            metadata=metadata,
            required_latents=target_base_start,
            target_base_start=target_base_start,
            history_latent_frames=self.history_latents,
            allow_short=True,
            allow_empty_target=True,
        )
        sink_count = int(trainer.cfg.layout.sink_latent_frames)
        history = (
            latent_full[:, :, sink_count : sink_count + self.history_latents]
            .clone()
            .contiguous()
        )
        sink_latent = latent_full[:, :, :sink_count].contiguous()
        motion_start = self.prefix_frames - trainer._vigeo_motion_pixel_frames()
        decode_anchor, motion_nearby = trainer._encode_vigeo_motion_window(
            trainer._slice_video_pixel_frames(
                video_pixels,
                motion_start,
                self.prefix_frames,
            )
        )
        batch, _, _, height, width = latent_full.shape
        sink_indices = trainer._indices_grid(
            batch,
            sink_count,
            height,
            width,
            t_offset=trainer._local_sink_t_offset(self.history_latents),
        )
        spatial_bank = trainer._init_validation_rollout_spatial_bank(
            video_pixels=video_pixels,
            metadata=metadata,
            target_start=target_base_start,
            history_latent_frames=self.history_latents,
        )
        if spatial_bank is None:
            raise RuntimeError("AlayaWorld v1.1 could not initialize its ViGeo cache")
        cache = RolloutCache(
            metadata=metadata,
            history=history,
            sink_latent=sink_latent,
            sink_indices=sink_indices,
            spatial_bank=spatial_bank,
            decode_anchor=decode_anchor,
            motion_nearby=motion_nearby,
            context=trainer._encode_caption(prompt, sync=False),
            prompt=prompt,
            seed=int(seed),
            target_base_start=target_base_start,
        )
        self.cache = cache
        return cache

    def generate(self, prompt: str, trajectory: np.ndarray) -> np.ndarray:
        """Generate, decode, and commit one 32-frame distilled chunk."""
        cache = self.cache
        if cache is None:
            raise RuntimeError("AlayaWorld v1.1 rollout is not initialized")
        if prompt != cache.prompt:
            cache.context = self.trainer._encode_caption(prompt, sync=False)
            cache.prompt = prompt
        self._write_camera_trajectory(cache, trajectory)
        frames = self._sample_chunk(cache)
        cache.round_index += 1
        return frames

    def _write_camera_trajectory(
        self,
        cache: RolloutCache,
        trajectory: np.ndarray,
    ) -> None:
        """Write frontend camera poses into the next ViGeo target window."""
        if trajectory.shape != (self.frames_per_chunk, 4, 4):
            raise ValueError(
                "camera trajectory must have shape "
                f"({self.frames_per_chunk}, 4, 4), got {trajectory.shape}"
            )
        start = self.prefix_frames + cache.round_index * self.frames_per_chunk
        end = start + self.frames_per_chunk
        cameras = cache.metadata["cam_c2w"]
        time_axis = 0 if cameras.dim() == 3 else 1
        current = int(cameras.shape[time_axis])
        if current < end:
            tail = cameras[-1:] if time_axis == 0 else cameras[:, -1:]
            repeats = [1] * cameras.dim()
            repeats[time_axis] = end - current
            cameras = self.torch.cat([cameras, tail.repeat(*repeats)], dim=time_axis)
        values = self.torch.from_numpy(trajectory).to(
            device=cameras.device,
            dtype=cameras.dtype,
        )
        if cameras.dim() == 3:
            cameras[start:end] = values
        else:
            cameras[:, start:end] = values.unsqueeze(0).expand(
                int(cameras.shape[0]),
                -1,
                -1,
                -1,
            )
        cache.metadata["cam_c2w"] = cameras
        cache.metadata["cam_c2w_raw"] = cameras.clone()
        cache.metadata["frame_end"] = end

    def _sample_chunk(self, cache: RolloutCache) -> np.ndarray:
        """Run the upstream v1.1 validation equations for one DMD round."""
        trainer = self.trainer
        torch = self.torch
        cfg = trainer.cfg
        components = trainer.components
        if components is None or trainer.history_encoder is None:
            raise RuntimeError("AlayaWorld v1.1 inference components are unavailable")

        batch, channels, _, height, width = cache.history.shape
        target_start = cache.target_base_start + cache.round_index * self.chunk_latents
        target_rope = trainer._local_target_t_indices(
            self.chunk_latents,
            history_latent_frames=self.history_latents,
            condition_latent_frames=self.condition_latents,
        )
        mem_tokens = None
        mem_indices = None
        if cache.round_index >= int(self.mode.memory_start_round):
            mem_tokens, mem_indices = trainer.history_encoder(cache.history)
            mem_indices = mem_indices.clone()
            mem_indices[:, 0, :, :] += trainer._local_memory_t_offset(
                self.history_latents,
                self.condition_latents,
            )
        nearby_indices = trainer._indices_grid(
            batch,
            self.condition_latents,
            height,
            width,
            t_offset=trainer._local_nearby_t_offset(
                self.history_latents,
                self.condition_latents,
            ),
        )
        spatial = trainer._build_validation_rollout_bank_spatial_context(
            bank=cache.spatial_bank,
            metadata=cache.metadata,
            target_start=target_start,
            K=self.chunk_latents,
            target_rope_t_indices=target_rope,
        )
        if spatial is None:
            raise RuntimeError(
                "AlayaWorld v1.1 could not render ViGeo spatial conditioning"
            )
        spatial_indices = trainer._indices_grid_for_t_indices(
            batch,
            spatial.get("rope_t_indices", spatial["target_indices"]),
            height,
            width,
        )
        generator = torch.Generator(device=trainer.dist.device)
        generator.manual_seed(cache.seed + cache.round_index)
        latent = torch.randn(
            batch,
            channels,
            self.chunk_latents,
            height,
            width,
            device=trainer.dist.device,
            dtype=trainer.dtype,
            generator=generator,
        )
        sigmas = trainer._validation_sigmas(latent_frames=self.chunk_latents)
        with torch.inference_mode():
            for step in range(len(sigmas) - 1):
                sigma_now = sigmas[step]
                sigma_next = sigmas[step + 1]
                velocity = components.transformer(
                    x=[latent.squeeze(0)],
                    t=(sigma_now * 1000.0)
                    .view(1)
                    .to(
                        device=trainer.dist.device,
                        dtype=latent.dtype,
                    ),
                    context=[cache.context],
                    seq_len=self.chunk_latents * height * width,
                    fps=cfg.sample.fps,
                    history_kv_tokens=mem_tokens,
                    history_indices_grid=mem_indices,
                    gen_t_indices_override=target_rope,
                    sink_latent=cache.sink_latent,
                    sink_indices_grid=cache.sink_indices,
                    spatial_latent=spatial["latent"],
                    spatial_mask_patch=spatial.get("mask_patch"),
                    spatial_indices_grid=spatial_indices,
                    nearby_latent=cache.motion_nearby,
                    nearby_indices_grid=nearby_indices,
                )
                if sigma_next.item() > 1e-5:
                    latent = (
                        latent - (sigma_now - sigma_next).to(latent.dtype) * velocity
                    )
                else:
                    latent = (latent.float() - velocity.float() * sigma_now.float()).to(
                        latent.dtype
                    )

            decoded, next_anchor, next_motion = (
                trainer._decode_and_reencode_vigeo_motion_chunk(
                    anchor_latent=cache.decode_anchor,
                    motion_latent=cache.motion_nearby,
                    target_latent=latent.detach(),
                )
            )
            trainer._record_validation_vigeo_causal_prefix(
                bank=cache.spatial_bank,
                decoded_pixels=decoded,
                target_start=target_start,
            )
            trainer._append_validation_rollout_spatial_bank_prediction(
                bank=cache.spatial_bank,
                pred_latent=latent.detach(),
                decoded_pixels=decoded,
                metadata=cache.metadata,
                target_start=target_start,
            )
        cache.history = torch.cat(
            [cache.history, latent.to(cache.history.dtype)],
            dim=2,
        )[:, :, -self.history_latents :].contiguous()
        cache.decode_anchor = next_anchor
        cache.motion_nearby = next_motion
        self.compact_cache(
            SimpleNamespace(spatial_bank=cache.spatial_bank),
            max_spatial_frames=self.config.max_spatial_frames,
            recent_spatial_frames=self.config.recent_spatial_frames,
        )
        pixels = (decoded * 0.5 + 0.5).clamp(0.0, 1.0)
        frames = (
            pixels[0]
            .permute(1, 2, 3, 0)
            .mul(255.0)
            .round()
            .to(torch.uint8)
            .cpu()
            .numpy()
        )
        if int(frames.shape[0]) != self.frames_per_chunk:
            raise RuntimeError(
                f"AlayaWorld v1.1 decoded {frames.shape[0]} frames; "
                f"expected {self.frames_per_chunk}"
            )
        return np.ascontiguousarray(frames)
