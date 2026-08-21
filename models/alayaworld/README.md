# AlayaWorld v1.1

Serve the public AlayaWorld v1.1 distilled autoregressive world model through
Reactor Runtime. Start from an uploaded or built-in image, change the prompt at
chunk boundaries, and move a six-axis camera while the generated world streams
back over `main_video`.

This recipe uses the stage3 DMD student: four denoising steps produce four
latent frames, which the LTX-2.3 VAE decodes into 32 RGB frames at 960x544. One
chunk represents about 1.33 seconds at the model's native 24 FPS. The model,
text encoder, VAE, temporal history, and ViGeo cache remain in the Runtime
process across chunks; there is no worker process or second model copy.

AlayaWorld v1.1 differs from the original recipe in two important ways. A
nine-frame causal VAE motion window replaces the static nearby condition, and
ViGeo's streaming 3D point cache replaces Depth-Anything-3. Camera AdaLN is not
used: frontend camera values become pixel-rate camera-to-world poses, and ViGeo
re-renders the spatial condition for the next chunk from those poses.

## Run

You need the `reactor` CLI, Docker with the NVIDIA Container Toolkit, and one
CUDA GPU with enough memory for the 13.1B transformer, Gemma, VAE, and ViGeo.
The checked-in resource requests one NVIDIA B200. Gemma is gated, so accept its
Hugging Face license and make a read token available on first load.

```sh
cd models/alayaworld
export HF_TOKEN=hf_your_read_token

reactor build
reactor run --gpus device=0 -e HF_TOKEN
```

The bare `-e HF_TOKEN` forwards the host value without placing the token in the
Docker command line. Select another host GPU with `--gpus device=3`. A different
port is applied to both Docker and Runtime:

```sh
reactor run --gpus device=3 -e HF_TOKEN --port 18080
curl -s localhost:18080/health
curl -s localhost:18080/schema
```

`reactor run` reuses the image from `reactor build` and builds it automatically
when the local image does not exist. Model loading can take several minutes on
the first start because it downloads assets and reads large checkpoints before
health becomes available.

## Play it in the browser

[`demo/`](./demo) is a small Next.js client for the model. With Runtime serving
on its default port, start it in another terminal:

```sh
cd models/alayaworld/demo
pnpm install
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000). For a non-default Runtime
port, copy `demo/.env.example` to `demo/.env` and set
`REACTOR_LOCAL_URL=http://localhost:18080`.

`W`/`A`/`S`/`D` move, `I`/`J`/`K`/`L` look, `Space`/`C` change height, and
`Q`/`E` roll. [`demo/README.md`](./demo/README.md) describes the complete
mapping and typed client.

## Public source and model assets

`runtime.weights_path` points at `~/.cache/reactor_registry/alayaworld-v1-1`.
The CLI bind-mounts that directory and exports it as `REACTOR_WEIGHTS_PATH`, so
image rebuilds preserve sources and checkpoints. To relocate everything,
change only that value in [`reactor.yaml`](./reactor.yaml).

The first model load prepares every missing public asset automatically:

- the pinned AlayaWorld v1.1 and ViGeo source revisions;
- the LTX-2.3 architecture, VAE, and text-connector weights from the public
  AlayaWorld bundle;
- the complete v1.1 stage2b transformer and history encoder;
- the v1.1 stage3 distilled LoRA;
- the Gemma text encoder; and
- the ViGeo 1.1 checkpoint.

Allow about 100 GB for the prepared assets and download metadata. Downloads are
resumable. Completed files and source revisions are verified on later starts;
an existing checkout at another revision fails clearly instead of being
rewritten. The two `random_image` cases ship with the pinned AlayaWorld source,
so there is no separate sample dataset.

The LTX-2.3 base filename referenced by the upstream v1.1 config is not present
in the current public LTX-2 repository snapshot. The public AlayaWorld bundle
contains the same architecture metadata, VAE, and text-connector weights. The
adapter uses those components, then loads the complete stage2b transformer with
`missing=0` and `unexpected=0` before applying the stage3 LoRA. This keeps the
recipe fully reproducible from available public artifacts.

AlayaWorld and its weights use the LTX-2 Community License for academic and
non-commercial use. Gemma and ViGeo retain their own terms. This recipe does not
redistribute their source or checkpoints.

## Controls

- `set_image` accepts uploaded JPEG, PNG, WebP, or BMP bytes and an optional
  prompt, then starts a fresh rollout.
- `random_image` selects a different built-in scene and its matching prompt.
- `set_forward`, `set_strafe`, and `set_vertical` control local Z, X, and Y
  translation. Positive values move forward, right, and up.
- `set_pitch`, `set_yaw`, and `set_roll` control local X, Y, and Z rotation.
  Positive values look up, turn right, and roll clockwise.
- `set_prompt` encodes new text for the next chunk without resetting history.
- `set_paused` stops before the next chunk and releases camera motion.
- `step` generates exactly one 32-frame chunk while paused.
- `reset` rebuilds the causal VAE, temporal history, and ViGeo state from the
  selected image. Its optional non-negative seed selects the fresh rollout.

All camera values are normalized to `[-1, 1]`. Camera and prompt commands return
typed messages with the one-based chunk expected to consume them. Successful
commands also broadcast `StateUpdate`, including queued and active prompts,
image, pause state, completed chunk count, and all six camera axes.

The frontend owns keyboard, pointer, joystick, and sensitivity mapping. The
backend integrates the normalized axes into 32 camera poses immediately before
the next ViGeo render, so a command received during an in-flight CUDA turn is
applied to the following chunk.

## Image uploads

`set_image` declares an `UploadedFile` in the model schema. A Reactor client
reserves a session upload, writes the raw bytes to its returned URL, and sends
the upload reference with the command. [`example_images/`](./example_images)
contains two licensed upstream still images ready for manual upload.

Uploads are limited to 25 MiB and 100 million decoded pixels. EXIF orientation
is applied before the image is resized and center-cropped to 960x544. The
configured `inputs.upload_template` supplies only camera calibration; its image
pixels are not used. Uploaded bytes remain session-scoped.

## Stateful chunk inference

At reset, the adapter creates the same state used by upstream v1.1 validation:
one sink latent, four temporal-history latents, a nine-pixel-frame causal motion
window, and a ViGeo stream initialized from the selected image. Each chunk:

1. samples the queued prompt and 32 frontend camera poses;
2. re-renders the spatial condition from the ViGeo point cache;
3. runs the four-step distilled transformer with temporal memory;
4. decodes exactly 32 target frames and re-encodes the next motion window; and
5. appends the generated geometry and updates the four-latent history.

Only the inference transformer is loaded. The DMD training teacher, critic,
optimizers, dataloaders, and FSDP wrappers are intentionally omitted. The
adapter calls the upstream model, checkpoint loader, history encoder, VAE, and
ViGeo helpers directly and does not modify the cloned AlayaWorld source.

The spatial bank is bounded to 320 frames, retaining 160 recent frames and 160
older keyframes. `stream.max_chunks_per_rollout` defaults to 512; reaching it
queues a reset from the selected image, active prompt, and seed without
reloading weights, so the Reactor session can continue indefinitely.

`inference.attention_backend` defaults to `flash_attention_4` for Hopper and
Blackwell GPUs. `pytorch` is useful for debugging but requires substantially
more time and memory at full resolution. `inference.compile` defaults to
`default`, and three warmup chunks run while Runtime still reports the model as
loading. They cover neutral and moving initial turns plus the temporal-history
path used from the second turn onward. The passes also initialize ViGeo before
the first viewer arrives. TorchInductor artifacts are cached under
`runtime.weights_path`; set `compile: none` and `warmup_chunks: 0` only when
debugging eager execution.

Recording captures the same interactive frames sent on `main_video` and
requires `ffmpeg` on `PATH`.
