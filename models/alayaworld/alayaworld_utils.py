"""Provide media, cache, camera-tensor, and upstream import helpers."""

from __future__ import annotations

import importlib
import io
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from reactor_runtime import CommandError, UploadedFile

_UPLOAD_MAX_BYTES = 25 * 1024 * 1024
_UPLOAD_MAX_PIXELS = 100_000_000
_UPLOAD_MIME_FORMATS = {
    "image/bmp": "BMP",
    "image/jpeg": "JPEG",
    "image/png": "PNG",
    "image/webp": "WEBP",
}
_FLASH_ATTENTION_4_OP: Any | None = None


def compact_rollout_cache(
    cache: Any,
    *,
    max_spatial_frames: int,
    recent_spatial_frames: int,
) -> None:
    """Bound generated latents and spatial memory while retaining old keyframes."""
    preds = getattr(cache, "preds", None)
    if isinstance(preds, list) and len(preds) > 1:
        preds[:] = preds[-1:]

    bank = getattr(cache, "spatial_bank", None)
    if bank is None:
        return
    pixels = bank.pixels
    frame_indices = bank.frame_indices
    depths = bank.depths
    total = len(frame_indices)
    if len(pixels) != total or len(depths) != total:
        raise RuntimeError("AlayaWorld spatial bank members have different lengths")
    if total <= max_spatial_frames:
        return

    recent_count = min(recent_spatial_frames, max_spatial_frames)
    historical_count = total - recent_count
    historical_budget = max_spatial_frames - recent_count
    if historical_budget == 1:
        historical = [0]
    elif historical_budget > 1:
        historical = [
            index * (historical_count - 1) // (historical_budget - 1)
            for index in range(historical_budget)
        ]
    else:
        historical = []
    keep = historical + list(range(historical_count, total))

    bank.pixels = [pixels[index] for index in keep]
    bank.frame_indices = [frame_indices[index] for index in keep]
    bank.depths = [depths[index] for index in keep]
    world_points = getattr(bank, "world_points", None)
    if isinstance(world_points, dict):
        bank.world_points = {
            new_index: world_points[old_index]
            for new_index, old_index in enumerate(keep)
            if old_index in world_points
        }
    for attribute in (
        "vigeo_pointmaps",
        "vigeo_valid_masks",
        "vigeo_predicted_poses",
        "vigeo_intrinsics",
    ):
        values = getattr(bank, attribute, None)
        if isinstance(values, list) and len(values) == total:
            setattr(bank, attribute, [values[index] for index in keep])


def validate_uploaded_image(image: UploadedFile) -> None:
    """Reject oversized, mislabeled, or undecodable uploaded image bytes."""
    expected_format = _UPLOAD_MIME_FORMATS.get(image.mime_type.lower())
    if expected_format is None:
        raise CommandError(
            "unsupported_media",
            f"{image.name} must declare image/jpeg, image/png, image/webp, or image/bmp.",
        )
    if not image.data:
        raise CommandError("invalid_image", f"{image.name} is empty.")
    if image.size > _UPLOAD_MAX_BYTES:
        raise CommandError(
            "image_too_large",
            f"{image.name} exceeds the {_UPLOAD_MAX_BYTES // (1024 * 1024)} MiB limit.",
        )
    try:
        with Image.open(io.BytesIO(image.data)) as decoded:
            image_format = decoded.format or ""
            width, height = decoded.size
            if image_format != expected_format:
                raise CommandError(
                    "unsupported_media",
                    f"{image.name} contains {image_format or 'unknown'} data but declares "
                    f"{image.mime_type}.",
                )
            if width <= 0 or height <= 0 or width * height > _UPLOAD_MAX_PIXELS:
                raise CommandError(
                    "image_too_large",
                    f"{image.name} exceeds the {_UPLOAD_MAX_PIXELS}-pixel limit.",
                )
            decoded.verify()
    except CommandError:
        raise
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError) as error:
        raise CommandError(
            "invalid_image", f"{image.name} cannot be decoded."
        ) from error


def uploaded_image_video(
    image: UploadedFile,
    metadata: dict[str, Any],
    *,
    target_hw: tuple[int, int],
    torch_module: Any,
) -> Any:
    """Decode, center-crop, and repeat an upload over the camera template."""
    target_height, target_width = target_hw
    with Image.open(io.BytesIO(image.data)) as decoded:
        oriented = ImageOps.exif_transpose(decoded).convert("RGB")
        fitted = ImageOps.fit(
            oriented,
            (target_width, target_height),
            method=Image.Resampling.LANCZOS,
        )
        pixels = np.array(fitted, dtype=np.uint8, copy=True)
    frame = torch_module.from_numpy(pixels).permute(2, 0, 1).float() / 127.5 - 1.0
    frame_count = int(camera_frames(metadata["cam_c2w"]).shape[0])
    return frame.unsqueeze(0).expand(frame_count, -1, -1, -1).contiguous()


def load_upstream_modules(source_path: Path) -> dict[str, Any]:
    """Import the unmodified public AlayaWorld v1.1 inference surface lazily."""
    source = str(source_path)
    if source not in sys.path:
        sys.path.insert(0, source)
    rollout = importlib.import_module("alaya.inference.rollout_utils")
    return {
        "torch": importlib.import_module("torch"),
        "load_config": importlib.import_module("alaya.config.loader").load_config,
        "rollout_trainer": importlib.import_module(
            "alaya.trainer.rollout_trainer"
        ).RolloutTrainer,
        "build_model_components": importlib.import_module(
            "alaya.model.loader"
        ).build_model_components,
        "build_history_encoder": importlib.import_module(
            "alaya.memory.builder"
        ).build_history_encoder,
        "load_checkpoint_weights": importlib.import_module(
            "alaya.checkpoint"
        ).load_checkpoint_weights,
        "load_input_sample": rollout.load_input_sample,
        "check_input_resolution": rollout.check_input_resolution,
        "pytorch_attention": importlib.import_module(
            "ltx2.modules.attention"
        ).AttentionFunction.PYTORCH,
    }


def set_attention_backend(engine: Any, attention_function: Any) -> int:
    """Select an attention callable on every loaded attention module."""
    changed = 0
    for root in (engine.transformer, engine.text_encoder):
        if root is None:
            continue
        for module in root.modules():
            if hasattr(module, "attention_function"):
                module.attention_function = attention_function
                changed += 1
    if changed == 0:
        raise RuntimeError("AlayaWorld exposed no configurable attention modules")
    return changed


class FlashAttention4:
    """Route attention through FlashAttention 4, keeping masked blocks on PyTorch.

    AlayaWorld calls its attention hook both with and without an additive or
    boolean mask. FlashAttention 4 covers the unmasked calls, including the
    sliding window it accepts natively, and the masked calls fall through to the
    PyTorch implementation that builds the equivalent banded mask.
    """

    def __init__(
        self, flash_attention: Any, masked_fallback: Any, torch_module: Any
    ) -> None:
        self._flash_attention = _flash_attention_4_op(
            torch_module,
            flash_attention,
        )
        self._masked_fallback = masked_fallback
        self._torch = torch_module

    def __call__(
        self,
        q: Any,
        k: Any,
        v: Any,
        heads: int,
        mask: Any | None = None,
        window_size: tuple[int, int] | None = None,
    ) -> Any:
        """Return attention over ``[batch, tokens, heads * head_dim]`` inputs."""
        if mask is not None:
            return self._masked_fallback(q, k, v, heads, mask, window_size=window_size)
        batch, _, fused = q.shape
        head_dim = fused // heads
        query, key, value = (t.view(batch, -1, heads, head_dim) for t in (q, k, v))
        bfloat16 = self._torch.bfloat16
        window_left, window_right = window_size or (-1, -1)
        out = self._flash_attention(
            query.to(bfloat16),
            key.to(bfloat16),
            value.to(bfloat16),
            window_left,
            window_right,
        )
        return out.reshape(batch, -1, heads * head_dim).to(value.dtype)


def _flash_attention_4_op(torch_module: Any, flash_attention: Any) -> Any:
    """Expose the CuTe kernel to Dynamo as one opaque tensor operation."""
    global _FLASH_ATTENTION_4_OP
    if _FLASH_ATTENTION_4_OP is not None:
        return _FLASH_ATTENTION_4_OP

    def kernel(
        query: Any,
        key: Any,
        value: Any,
        window_left: int,
        window_right: int,
    ) -> Any:
        output = flash_attention(
            query,
            key,
            value,
            window_size=(window_left, window_right),
        )
        return output[0] if isinstance(output, tuple) else output

    operation = torch_module.library.custom_op(
        "reactor_alayaworld::flash_attention_4",
        mutates_args=(),
        schema="(Tensor query, Tensor key, Tensor value, int window_left, int window_right) -> Tensor",
    )(kernel)

    @operation.register_fake
    def fake(
        query: Any,
        _key: Any,
        _value: Any,
        _window_left: int,
        _window_right: int,
    ) -> Any:
        return torch_module.empty_like(query)

    _FLASH_ATTENTION_4_OP = operation
    return operation


def resolve_attention_backend(
    backend: str,
    *,
    pytorch_attention: Any,
    torch_module: Any,
) -> Any | None:
    """Return the attention callable for *backend*, or ``None`` to leave upstream's.

    Raises:
        RuntimeError: FlashAttention 4 was asked for but cannot be imported.
    """
    if backend == "upstream":
        return None
    if backend == "pytorch":
        return pytorch_attention
    try:
        from flash_attn.cute import flash_attn_func
    except ImportError as error:
        raise RuntimeError(
            "inference.attention_backend is flash_attention_4 but flash-attn-4 is "
            "not importable; install it or set the backend to pytorch"
        ) from error
    return FlashAttention4(flash_attn_func, pytorch_attention, torch_module)


def camera_frames(camera: Any) -> Any:
    """Return camera poses as ``[F, 4, 4]`` from batched or unbatched input."""
    if camera.dim() == 3:
        return camera
    if camera.dim() == 4:
        return camera[0]
    raise ValueError(f"cam_c2w must be [F,4,4] or [B,F,4,4], got {tuple(camera.shape)}")


def ensure_camera_capacity(camera: Any, frame_count: int, torch_module: Any) -> Any:
    """Extend a camera tensor by repeating its final pose when needed."""
    current = int(camera_frames(camera).shape[0])
    if current >= frame_count:
        return camera
    missing = frame_count - current
    time_axis = 0 if camera.dim() == 3 else 1
    tail = camera[-1:] if camera.dim() == 3 else camera[:, -1:]
    repeats = [1] * camera.dim()
    repeats[time_axis] = missing
    return torch_module.cat([camera, tail.repeat(*repeats)], dim=time_axis)


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
