"""Serve one native Zing 0.5 autoregressive video block per Reactor turn."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Literal
from PIL import Image, ImageOps

import numpy as np
from reactor_runtime import (
    ClientInfo,
    CommandError,
    InputField,
    ReactorApp,
    ApplicationError,
    StepOutcome,
    get_weights_path,
    UploadedFile,
    connected,
    disconnected,
    event,
    session_ended,
    session_started,
)
from reactor_runtime.log import get_logger

from zing_assets import (
    ZingAdapterConfig,
    read_config,
)
from zing_images import validate_image
from zing_model import ZingModel, ZingInput, ZingResult
from zing_types import (
    ActionChanged,
    ChunkCompleted,
    ControlsReleased,
    ImageSelected,
    PromptQueued,
    RolloutLimitReached,
    RolloutReset,
    StateUpdate,
    ZingOutput,
    ZingState,
)

logger = get_logger(__name__)
_KEYS = ("w", "a", "s", "d", "i", "j", "k", "l")


class Zing(ReactorApp):
    """Generate a controllable Zing 0.5 world from text or one initial image."""

    state: ZingState
    buffer_size = 16

    def __init__(self) -> None:
        super().__init__()
        self._config: ZingAdapterConfig | None = None
        self.engine = ZingModel()
        self._conditioning: Literal["none", "text", "uploaded", "built_in"] = "none"
        self._image: np.ndarray | None = None
        self._image_name: str | None = None
        self._seed = 42
        self._active_prompt: str | None = None
        self._completed_chunks = 0
        self._generating = False
        self._limit_reached = False
        self._world_epoch = 0
        self._applied_world_id: int | None = None

    def load(self, config_path: Path | None) -> None:
        config = read_config(config_path, get_weights_path())
        self._config = config
        self._seed = config.seed
        self.engine.load(config_path, get_weights_path())
        logger.info(
            "Zing 0.5 ready",
            source_revision=config.source_revision,
            checkpoint_revision=config.asset_revision,
            cache_window="97/9",
            frames_per_chunk=16,
        )

    @session_started
    def on_session_started(self) -> None:
        config = self._require_config()
        self.state.prompt = ""
        self.state._pressed_keys = frozenset()
        self._applied_world_id = None
        self._conditioning = "none"
        self._image = None
        self._image_name = None
        self._seed = config.seed
        self._active_prompt = None
        self._completed_chunks = 0
        self._generating = False
        self._limit_reached = False
        self._world_epoch = 0

    @session_ended
    def on_session_ended(self) -> None:
        self._applied_world_id = None
        self.state._pressed_keys = frozenset()
        self._conditioning = "none"
        self._image = None
        self._image_name = None
        self._active_prompt = None
        self._completed_chunks = 0
        self._generating = False
        self.engine.reset()

    @connected
    async def on_connected(self, client: ClientInfo) -> None:
        """Send the complete shared world state to one joining viewer."""
        await client.send(self._state_update())

    @disconnected
    async def on_disconnected(self) -> None:
        """Release held controls when a viewer disconnects."""
        self.state._pressed_keys = frozenset()
        await self.send(self._state_update())

    @event(
        name="set_prompt",
        description=(
            "Set the text condition without restarting the current world. Before generation, the "
            "prompt starts a text-to-video world; during generation, the normalized text is "
            "sampled when the next chunk begins and preserves prior world history. Emits "
            "`prompt_queued` and broadcasts `state_update` on success, or `command_error` for "
            "empty text."
        ),
    )
    async def set_prompt(
        self,
        prompt: str = InputField(
            max_length=4096,
            moderate=True,
            description=(
                "Non-empty scene, appearance, subject, camera, and motion description, up to "
                "4096 characters. Whitespace is trimmed and the result is sampled when the next "
                "chunk starts."
            ),
        ),
    ) -> PromptQueued:
        """Queue a prompt and report the chunk expected to consume it."""
        normalized = prompt.strip()
        if self._limit_reached:
            raise CommandError(
                "rollout_limit_reached",
                "The current world is exhausted; select an image or explicitly reset "
                "to start a new world.",
            )
        if not normalized:
            raise CommandError("empty_prompt", "Zing requires a non-empty prompt.")
        initial = self._completed_chunks == 0 and self._active_prompt is None
        self.state.prompt = normalized
        starts_text_rollout = initial and self._image is None
        if starts_text_rollout:
            self._conditioning = "text"
            self._image_name = None
            self._request_reset()
        message = PromptQueued(
            prompt=normalized,
            applies_to_chunk=self._completed_chunks + 1,
            resets_rollout=starts_text_rollout,
        )
        await self.send(self._state_update())
        return message

    @event(
        name="set_image",
        description=(
            "Select an uploaded anchor image and queue a fresh world with continuous generation. "
            "Valid at any time; the image replaces prior world history before chunk one. Emits "
            "`image_selected` and broadcasts `state_update` on success, or `command_error` when "
            "the upload is empty, oversized, mislabeled, or undecodable."
        ),
    )
    async def set_image(
        self,
        image: UploadedFile = InputField(
            moderate=True,
            description=(
                "Anchor uploaded through Reactor as JPEG, PNG, WebP, or BMP, up to 25 MiB and "
                "100 million pixels. EXIF orientation is applied before resizing to `main_video`."
            ),
        ),
        prompt: str = InputField(
            default="",
            max_length=4096,
            moderate=True,
            description=(
                "Optional scene and motion description for the fresh world. An empty value uses "
                "the configured image-neutral prompt."
            ),
        ),
        seed: int = InputField(
            default=-1,
            ge=-1,
            le=2_147_483_647,
            description="Fresh-rollout seed, or -1 to retain the active seed.",
        ),
    ) -> ImageSelected:
        """Validate an uploaded image and select it for a fresh world."""
        validate_image(image)
        config = self._require_config()
        if seed >= 0:
            self._seed = seed
        self.state.prompt = prompt.strip() or config.default_prompt
        self._conditioning = "uploaded"
        with Image.open(io.BytesIO(image.data)) as decoded:
            self._image = np.asarray(ImageOps.exif_transpose(decoded).convert("RGB")).copy()
        self._image_name = image.name
        self._request_reset()
        message = ImageSelected(
            source="uploaded",
            filename=image.name,
            prompt=self.state.prompt,
            seed=self._seed,
        )
        await self.send(self._state_update())
        return message

    @event(
        name="example_image",
        description=(
            "Select Zing's public example image and matching prompt, then queue a fresh world "
            "with continuous generation. Valid whenever the configured example is available and "
            "takes no input. Emits `image_selected` and broadcasts `state_update` on success, or "
            "`command_error` when the example is unavailable."
        ),
    )
    async def example_image(self) -> ImageSelected:
        """Select the public example image for a fresh world."""
        config = self._require_config()
        image = config.source_path / "assets" / "case0.jpg"
        self.state.prompt = config.example_prompt
        self._conditioning = "built_in"
        with Image.open(image) as decoded:
            self._image = np.asarray(ImageOps.exif_transpose(decoded).convert("RGB")).copy()
        self._image_name = image.name
        self._request_reset()
        await self.send(self._state_update())
        return ImageSelected(
            source="built_in",
            filename=image.name,
            prompt=self.state.prompt,
            seed=self._seed,
        )

    @event(
        name="set_key",
        description=(
            "Press or release one held native key for forthcoming chunks. "
            "`w`: move forward; `a`: strafe left; `s`: move backward; `d`: strafe right; "
            "`i`: look up; `j`: look left; `k`: look down; `l`: look right. The complete "
            "held state is sampled when the next chunk begins and remains active until changed or "
            "released. Emits `action_changed` and broadcasts `state_update` on success."
        ),
    )
    async def set_key(
        self,
        key: Literal["w", "a", "s", "d", "i", "j", "k", "l"] = InputField(
            description=(
                "Native key to change: `w`: move forward; `a`: strafe left; "
                "`s`: move backward; `d`: strafe right; `i`: look up; "
                "`j`: look left; `k`: look down; `l`: look right. "
                "Native key names are not external action labels. "
                "Looking down requires native `k`, never native `j`."
            )
        ),
        pressed: bool = InputField(
            description="Whether to hold or release `key`; the change applies to the next chunk."
        ),
    ) -> ActionChanged:
        """Change one held control and report the complete held state."""
        if pressed and self._limit_reached:
            raise CommandError(
                "rollout_limit_reached",
                "The current world is exhausted; controls can only be released "
                "until an explicit reset.",
            )
        keys = set(self.state._pressed_keys)
        (keys.add if pressed else keys.discard)(key)
        self.state._pressed_keys = frozenset(keys)
        await self.send(self._state_update())
        return ActionChanged(
            key=key,
            pressed=pressed,
            pressed_keys=sorted(keys),
            applies_to_chunk=None if self._limit_reached else self._completed_chunks + 1,
        )

    @event(
        name="release_controls",
        description=(
            "Release every held movement and look control for forthcoming chunks. Neutral input "
            "is sampled when the next chunk begins. Emits `controls_released` and broadcasts "
            "`state_update` on success."
        ),
    )
    async def release_controls(self) -> ControlsReleased:
        """Release all held controls and report which keys changed."""
        released = sorted(self.state._pressed_keys)
        self.state._pressed_keys = frozenset()
        await self.send(self._state_update())
        return ControlsReleased(
            released_keys=released,
            applies_to_chunk=None if self._limit_reached else self._completed_chunks + 1,
        )

    @event(
        name="reset",
        description=(
            "Queue a fresh world from the selected text or image condition and current prompt. "
            "Use after selecting a prompt or image; the reset clears progress, releases held "
            "controls, and resumes continuous generation from chunk one. Emits `rollout_reset` "
            "and broadcasts `state_update` on success."
        ),
    )
    async def reset(
        self,
        seed: int = InputField(
            default=-1,
            ge=-1,
            le=2_147_483_647,
            description="Fresh-rollout seed, or -1 to retain the active seed.",
        ),
    ) -> RolloutReset:
        """Queue a fresh world and report the progress it replaces."""
        if seed >= 0:
            self._seed = seed
        replaced = self._completed_chunks
        self._request_reset()
        await self.send(self._state_update())
        return RolloutReset(seed=self._seed, replaced_chunks=replaced)

    async def process_input(self) -> ZingInput:
        if self._conditioning == "none":
            raise ApplicationError("no prompt or image selected")
        if self._limit_reached:
            raise ApplicationError("rollout limit reached")
        fresh = self._world_epoch != self._applied_world_id
        return ZingInput(
            self._world_epoch, self._image if fresh else None, self.state.prompt,
            self._seed, self.state._pressed_keys, self._conditioning in {"uploaded", "built_in"},
        )

    def generate(self, input: ZingInput) -> ZingResult:
        return self.engine.generate(input)

    async def process_output(self, outcome: StepOutcome) -> ZingOutput | None:
        if outcome.error is not None:
            # Native inference failures are fatal; a reset cannot repair them.
            raise outcome.error
        result: ZingResult = outcome.result
        self._applied_world_id = result.world_id
        self._completed_chunks = result.index
        self._active_prompt = result.prompt
        self._generating = False
        if result.complete:
            self._limit_reached = True
            self.state._pressed_keys = frozenset()
            await self.send(RolloutLimitReached(
                completed_chunks=result.index, max_chunks=self._require_config().max_chunks,
                world_epoch=result.world_id,
            ))
        await self.send(ChunkCompleted(
            chunk=result.index, video_frames=int(result.frames.shape[0]),
            generation_seconds=outcome.elapsed, prompt=result.prompt,
            action_keys=sorted(result.pressed_keys), cache_frames=result.cache_frames,
        ))
        await self.send(self._state_update())
        return ZingOutput(main_video=result.frames)

    def _request_reset(self) -> None:
        self.state._pressed_keys = frozenset()
        self._active_prompt = None
        self._completed_chunks = 0
        self._limit_reached = False
        self._world_epoch += 1
        self.output.flush()

    def _state_update(self) -> StateUpdate:
        return StateUpdate(
            prompt=self.state.prompt,
            active_prompt=self._active_prompt,
            pressed_keys=sorted(self.state._pressed_keys),
            conditioning=self._conditioning,
            image_name=self._image_name,
            seed=self._seed,
            completed_chunks=self._completed_chunks,
            reset_queued=self._world_epoch != self._applied_world_id and self._world_epoch > 0,
            generating=self._generating,
            max_chunks=self._config.max_chunks if self._config is not None else 0,
            limit_reached=self._limit_reached,
            world_epoch=self._world_epoch,
        )

    def _require_config(self) -> ZingAdapterConfig:
        if self._config is None:
            raise RuntimeError("Zing is not loaded")
        return self._config
