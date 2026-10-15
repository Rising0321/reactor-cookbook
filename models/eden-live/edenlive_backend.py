"""Reuse the original 5B LightTAE pipeline, sampler, DINO and cache operations."""

import math
import sys
from pathlib import Path


class EdenBackend:
    def __init__(
        self, *, source: Path, weights: Path,
        attention_backend: str = "native", cuda_graph: bool = False,
        compile_model: bool = False,
    ) -> None:
        # Imports stay inside the Runner worker; source is resolved by the app.
        for directory in (
            source / "realtime/flashdreams",
            source / "realtime/integrations_v2",
        ):
            sys.path.insert(0, str(directory))
        import torch
        from eden.game5b_config import PIPELINE_GAME2V_EDEN_WORLD_V2_5B_TAEHV
        from eden.impl.game14b.dino import ProxyDinoEncoderConfig
        from flashdreams.infra.config import derive_config

        self.torch = torch
        # The released native CLI enables this before constructing the pipeline.
        torch.backends.cudnn.benchmark = True
        # The one-shot encoder stays eager; graph capture uses the author's
        # steady-cache dispatch for the DiT and causal LightTAE decoder.
        config = derive_config(
            PIPELINE_GAME2V_EDEN_WORLD_V2_5B_TAEHV,
            encoder={
                "encoder": {
                    "checkpoint_path": str(weights / "vae-5b/Wan2.2_VAE.pth"),
                    "use_compile": False,
                    "use_cuda_graph": False,
                }
            },
            decoder={
                "checkpoint_path": str(weights / "decoder-5b/lighttaew2_2.pth"),
                "use_compile": False,
                "use_cuda_graph": cuda_graph,
            },
            text_encoder={"model_id_or_local_path": str(weights / "umt5")},
            diffusion_model={
                "transformer": {
                    "dcp_checkpoint_path": str(
                        weights / "dit-5b/diffusion_pytorch_model.safetensors"
                    ),
                    "compile_network": False,
                    "use_cuda_graph": cuda_graph,
                }
            },
        )
        self.pipeline = config.setup().to("cuda").eval()
        if attention_backend == "fa4":
            from edenlive_attention import install_flash_attention4

            install_flash_attention4(self.pipeline.diffusion_model.transformer.network)
        elif attention_backend != "native":
            raise ValueError("attention_backend must be 'native' or 'fa4'")
        self.dino = ProxyDinoEncoderConfig(
            backbone_repo=str(weights / "vision-encoder/backbone"),
            projection_path=str(weights / "vision-encoder/projection.safetensors"),
            compile_encoder=False,
        ).setup()
        if compile_model:
            from flashdreams.infra.acceleration.cuda_graph_dispatch import CUDAGraphDispatch
            from flashdreams.infra.cuda_graph import CUDAGraphWrapper

            # Retain eager precision-cast boundaries and let the native cache
            # dispatch control capture after its sink/window reaches steady state.
            options = {"emulate_precision_casts": True, "triton.cudagraphs": False}
            transformer = self.pipeline.diffusion_model.transformer
            transformer.network = torch.compile(transformer.network, options=options)
            transformer._cuda_graph_dispatch = CUDAGraphDispatch(
                transformer.network,
                enabled=cuda_graph,
                capture_ar_idx=transformer._cuda_graph_capture_ar_idx,
                warmup_iters=transformer.config.cuda_graph_warmup_iters,
            )
            decoder = self.pipeline.decoder.taehv
            decoder.decoder = torch.compile(decoder.decoder, options=options)
            if cuda_graph:
                decoder._decoder_wrapper = CUDAGraphWrapper(
                    decoder.decoder, warmup_iters=decoder._warmup_iters,
                )
            self.dino._call = torch.compile(self.dino.encoder, options=options)
        self.cache = None
        self.prompt = ""

    def reset(self) -> None:
        self.cache = None
        self.prompt = ""

    def start(self, anchor) -> None:
        torch = self.torch
        with torch.no_grad():
            # Keep the native float bilinear cover resize, including rounding.
            pixels = torch.from_numpy(anchor.image).permute(2, 0, 1).float()
            scale = max(704 / pixels.shape[1], 1280 / pixels.shape[2])
            pixels = torch.nn.functional.interpolate(
                pixels[None],
                size=(
                    math.ceil(pixels.shape[1] * scale),
                    math.ceil(pixels.shape[2] * scale),
                ),
                mode="bilinear",
                align_corners=False,
            )[0]
            top, left = (pixels.shape[1] - 704) // 2, (pixels.shape[2] - 1280) // 2
            image = (
                pixels[:, top : top + 704, left : left + 1280].contiguous() / 127.5 - 1
            )
            image = image.unsqueeze(0).to(device="cuda", dtype=torch.float32)
            # A user-requested fresh world has its own native per-model RNG.
            diffusion = self.pipeline.diffusion_model
            diffusion.config.seed = anchor.seed
            diffusion._rng = None
            diffusion._initial_video_noise = None
            self.cache = self.pipeline.initialize_cache(
                text=[anchor.prompt],
                image=image,
                release_oneshot_encoders=False,
            )
            self.prompt = anchor.prompt

    def step(self, index, condition, prompt):
        torch = self.torch
        with torch.no_grad():
            images = torch.from_numpy(condition).permute(0, 3, 1, 2).contiguous()
            images = images.pin_memory().float().div_(255.0)
            control = self.dino.encode(
                images, latent_height=44, latent_width=80, dtype=torch.bfloat16
            )
            if prompt != self.prompt:
                embeddings = self.pipeline.encode_text([prompt])
                self.pipeline.replace_text_embeddings(self.cache, embeddings)
                self.prompt = prompt
            frames = self.pipeline.generate(
                autoregressive_index=index, cache=self.cache, input=control
            )
            self.pipeline.finalize(autoregressive_index=index, cache=self.cache)
            # Native TCHW [-1,1] -> RGB24 conversion, before transport encoding.
            rgb = ((frames.float().clamp(-1, 1) + 1) * 127.5).round().clamp(0, 255)
            return rgb.to(torch.uint8).permute(0, 2, 3, 1).contiguous().cpu().numpy()
