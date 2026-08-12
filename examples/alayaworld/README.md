# Play AlayaWorld through Reactor Runtime

Serve the public AlayaWorld distilled autoregressive world model through
Reactor Runtime. The adapter calls AlayaWorld's `FlashAlayaPipeline` directly;
the weights, text encoder, spatial memory, prompt encoding, denoiser, and VAE
remain in the Runtime process and stay resident across chunks.

AlayaWorld generates four latent frames per turn. Each turn adds 32 RGB frames,
or about 1.33 seconds at 24 FPS. Prompt and six-axis camera values are sampled
at the chunk boundary. Commands received during an in-flight CUDA turn apply
to the following chunk.

A new session starts paused without choosing a scene for the user. Upload an
image with `set_image`, or invoke `random_image` to select one of the public
AlayaWorld examples. Either command initializes the autoregressive cache and
starts generation.

The adapter and upstream implementation remain separate. This directory owns
only the Reactor integration; `source.path` in `alayaworld.yaml` points to an
unmodified AlayaWorld checkout.

## Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) and Git.
- An NVIDIA GPU supported by the CUDA 12.8 PyTorch wheels.
- Access to the gated Gemma text encoder on Hugging Face.
- `ffmpeg` when recording is needed.

## Clone AlayaWorld

From the cookbook repository, enter this example directory and clone the
upstream source at the revision tested by the adapter:

```sh
cd examples/alayaworld
git clone https://github.com/AlayaLab/AlayaWorld.git
git -C AlayaWorld checkout 1abbe2e196603ef9f8eeedfddd427b4d37d57bc1
```

The default `source.path: AlayaWorld` now resolves to that checkout. To keep it
elsewhere, change only `source.path` in `alayaworld.yaml`; checkpoints, the
inference configuration, and playground inputs are all resolved from that root.

## Install

Create an isolated environment from the example directory:

```sh
uv venv --python 3.12
source .venv/bin/activate

uv pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple

uv pip install -r requirements.txt
hf auth login
```

Accept the Gemma license on Hugging Face before running `hf auth login`. The
example uses a PyTorch 2.9.1 CUDA 12.8 stack validated on NVIDIA Blackwell. The
adapter's NumPy 2 requirement follows Reactor Runtime, while AlayaWorld's
inference code remains unchanged.

The model YAML selects AlayaWorld's existing PyTorch attention callable. Stable
xFormers wheels do not provide SM100 kernels for B200, so xFormers is not part
of this environment.

On the first start, the adapter downloads the pinned merged checkpoint and
Gemma text encoder, clones the pinned Depth-Anything-3 source, and populates its
model cache. Later starts reuse those local resources. Model loading reports
download progress in the server log, and failures name the affected public
repository.

The adapter points DA3 directly at the pinned checkout's `src` directory. Its
full application package is not installed because that optional dependency set
pins NumPy below 2 and includes visualization/export tools that inference does
not import. AlayaWorld supplies local stubs for those optional export paths.

AlayaWorld and its weights use the LTX-2 Community License for academic and
non-commercial use. Gemma and Depth-Anything-3 retain their own terms. The
example does not redistribute any source or checkpoint.

## Run

Select one GPU and start Runtime from the example directory:

```sh
CUDA_VISIBLE_DEVICES=0 PYTORCH_ALLOC_CONF=expandable_segments:True \
  python -m reactor_runtime.serve
```

The backend listens on `0.0.0.0:8080`. Open the
[Reactor Sandbox](https://reactor-sandbox.vercel.app), enter
`http://localhost:8080`, and start a session. You can also check it directly:

```sh
curl http://localhost:8080/health
```

## Controls

- `set_image` accepts uploaded JPEG, PNG, WebP, or BMP bytes plus an optional
  prompt, then resets from that image.
- `random_image` selects a different built-in example when possible and applies
  its matching prompt.
- `set_forward`, `set_strafe`, and `set_vertical` control local Z, X, and Y
  translation. Positive values move forward, right, and up.
- `set_pitch`, `set_yaw`, and `set_roll` control local X, Y, and Z rotation.
  Positive values look up, turn right, and roll clockwise.
- Every camera command accepts `-1.0` to `1.0` and returns a
  `CameraMotionChanged` message containing the complete six-axis state and the
  one-based chunk expected to consume it.
- `set_prompt` changes the text condition for the next chunk and returns a
  `PromptQueued` confirmation with the expected one-based chunk number.
- `set_paused` stops before another expensive chunk starts and releases camera
  motion.
- `step` generates and plays exactly one 32-frame chunk while paused.
- `reset` rebuilds the autoregressive and spatial-memory state from the initial
  selected image. Its optional non-negative `seed` selects the next reproducible
  rollout.

The frontend owns keyboard, pointer, joystick, sensitivity, and layout mapping.
A conventional eight-key mapping uses W/S for forward, A/D for strafe, the up
and down arrows for pitch, and the left and right arrows for yaw. Key down sends
`-1` or `1`; key up sends zero. Vertical translation and roll remain available
for touch controls, a second joystick, or a six-degree-of-freedom controller.
The backend normalizes simultaneous translation and rotation axes before
integrating them into the pixel-rate camera-to-world matrices consumed by
AlayaWorld.

## Image uploads

The `set_image` command declares an `UploadedFile` parameter in Runtime's
schema. A Reactor client reserves a session upload slot, writes the raw bytes to
its returned URL, and sends the resulting upload reference with the command.
This lets a schema-driven frontend render a real file picker instead of putting
binary data inside a command message.

Ready-to-upload examples are included in [`example_images`](./example_images/).
They are copies of the two public AlayaWorld playground images, with their
upstream license and attribution kept beside them.

Uploads are limited to 25 MiB and 100 million decoded pixels. EXIF orientation
is applied before the image is resized and center-cropped to 960x544. The
configured `inputs.upload_template` supplies camera calibration and a trajectory
horizon; its pixels are not used. Uploaded bytes remain session-scoped and are
released when the session ends.

`inputs.random_images` in `alayaworld.yaml` lists the known image triplets used
by `random_image`. The current configuration exposes the two public playground
cases shipped with the pinned AlayaWorld source.

## Long-running sessions

The adapter imposes no chunk-count limit. It uses the default `torch.compile`
mode so generation keeps Inductor-compiled kernels without enabling CUDA Graph
Trees, whose thread-local execution state is incompatible with Runtime's
off-loop chunk generation.

Autoregressive history remains a 16-latent sliding window. The adapter keeps
only the latest generated latent outside that window and bounds the spatial
bank at 320 frames: 160 dense recent frames plus 160 keyframes sampled across
the older trajectory. This keeps GPU memory and retrieval work bounded while
preserving sparse long-range viewpoints. Very old revisits therefore have less
spatial detail than recent ones, and visual drift remains a model limitation
rather than a Runtime horizon.

## Streaming decode

The public VAE is non-causal. The adapter decodes each new chunk with six
latents of left context, which keeps memory bounded and avoids re-decoding the
whole history. It cannot be pixel-identical to a future-aware offline decode at
the live edge. Recording captures the same interactive frames sent over
`main_video` and requires `ffmpeg` on `PATH`.
