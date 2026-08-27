"""Define DreamX-World configuration and its public Reactor schema."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from reactor_runtime import (
    InputField,
    InputState,
    MessageField,
    ModelMessage,
    Output,
    Video,
)

ImageSource = Literal["uploaded", "built_in"]


@dataclass(frozen=True)
class RepositoryAsset:
    """Describe one public Hugging Face repository pinned to a revision."""

    path: Path
    repo_id: str
    revision: str


@dataclass(frozen=True)
class DreamXConfig:
    """Hold validated source, checkpoint, inference, and interaction settings."""

    source_path: Path
    source_url: str
    source_revision: str
    upstream_config: Path
    transformer_config: Path
    evaluation_inputs: Path
    random_images: tuple[Path, ...]
    dreamx: RepositoryAsset
    wan: RepositoryAsset
    seed: int
    motion_speed: float
    color_correction_strength: float
    max_chunks_per_rollout: int
    default_upload_prompt: str


class DreamXWorldOutput(Output):
    """Stream one generated RGB frame batch on `main_video`."""

    main_video: Video


class StateUpdate(ModelMessage):
    """Emitted when observable session state changes or a viewer connects."""

    image_source: ImageSource | None = MessageField(
        description=(
            'Source of the rollout image: "uploaded", "built_in", or null before an image '
            "selection succeeds."
        )
    )
    image_name: str | None = MessageField(
        description="Selected image filename, or null before the session has an image."
    )
    prompt: str | None = MessageField(
        description=(
            "Non-empty scene or event prompt queued for the next native chunk, or null "
            "before an image is selected."
        )
    )
    active_prompt: str | None = MessageField(
        description=(
            "Prompt used by the most recently completed native chunk, or null before the "
            "rollout generates a chunk."
        )
    )
    pressed_keys: list[str] = MessageField(
        description=(
            "Native DreamX camera keys currently held for subsequent chunks. `set_paused`, "
            "`reset`, and image-selection commands clear this list."
        )
    )
    seed: int = MessageField(
        description="Random seed used when the current or queued rollout is initialized."
    )
    paused: bool = MessageField(
        description="Whether automatic chunk generation is paused before the next chunk."
    )
    step_queued: bool = MessageField(
        description=(
            "Whether exactly one native chunk is queued while paused by `step` or image "
            "selection's automatic preview."
        )
    )
    reset_queued: bool = MessageField(
        description="Whether a fresh rollout will start before the next generated chunk."
    )
    generating: bool = MessageField(
        description="Whether a rollout reset or native chunk inference is currently running."
    )
    completed_chunks: int = MessageField(
        description="Number of native three-latent-frame chunks completed since reset."
    )
    next_chunk: int | None = MessageField(
        description=(
            "One-based chunk that will consume the queued prompt and keys, or null before an "
            "image is selected."
        )
    )
    max_chunks: int = MessageField(
        description="Chunk count that automatically starts a fresh rollout from the same image."
    )


class ImageSelected(ModelMessage):
    """Emitted when an image command queues a fresh DreamX-World rollout."""

    source: ImageSource = MessageField(
        description=(
            'Selected image source: "uploaded" for `set_image` or "built_in" for '
            "`random_image`."
        )
    )
    filename: str = MessageField(
        description="Selected filename displayed by the client for the fresh rollout."
    )
    prompt: str = MessageField(
        description="Effective non-empty prompt queued for the fresh rollout's first chunk."
    )
    applies_to_chunk: int = MessageField(
        description="One-based chunk affected by the image selection; always 1."
    )


class PromptQueued(ModelMessage):
    """Emitted when `set_prompt` queues text for the next native chunk."""

    prompt: str = MessageField(
        description="Trimmed, non-empty scene or event prompt queued by `set_prompt`."
    )
    applies_to_chunk: int = MessageField(
        description="One-based chunk that will first use the new cross-attention condition."
    )


class ActionChanged(ModelMessage):
    """Emitted after a `set_key_state` command is processed."""

    control: str = MessageField(
        description=(
            "Wire name of the command that produced this response; always `set_key_state`."
        )
    )
    paused: bool = MessageField(
        description=(
            "Pause state after the command is processed. Held keys may be changed while true "
            "and are sampled by the next `step` or resumed chunk."
        )
    )
    pressed_keys: list[str] = MessageField(
        description=(
            "Native camera keys held after the command is processed. Each key applies to "
            "subsequent chunks until `set_key_state` releases it or controls are cleared."
        )
    )


class PauseChanged(ModelMessage):
    """Emitted when `set_paused` changes generation and releases held keys."""

    paused: bool = MessageField(
        description="Whether automatic generation will stop before the next native chunk."
    )
    keys_released: bool = MessageField(
        description="Whether every held camera key was released; always true."
    )


class StepQueued(ModelMessage):
    """Emitted when `step` queues one native chunk while paused."""

    applies_to_chunk: int = MessageField(
        description="One-based chunk that the queued paused step will generate."
    )


class RolloutResetQueued(ModelMessage):
    """Emitted when a manual or automatic reset queues a fresh rollout."""

    trigger: Literal["manual", "automatic_chunk_limit"] = MessageField(
        description=(
            'Reset source: "manual" for `reset` or "automatic_chunk_limit" after '
            "`max_chunks` completes."
        )
    )
    seed: int = MessageField(
        description="Random seed that will initialize the queued fresh rollout."
    )
    completed_chunks: int = MessageField(
        description="Number of completed chunks in the rollout being replaced."
    )
    applies_to_chunk: int = MessageField(
        description="One-based chunk affected by the reset; always 1 for a fresh rollout."
    )


class ChunkGenerated(ModelMessage):
    """Emitted after one native autoregressive chunk completes successfully."""

    chunk: int = MessageField(
        description="One-based chunk completed in the active rollout."
    )
    frames: int = MessageField(
        description=(
            "RGB frames decoded from the chunk: 9 for the first chunk and 12 thereafter."
        )
    )
    prompt: str = MessageField(
        description="Scene or event prompt sampled for this completed chunk."
    )
    pressed_keys: list[str] = MessageField(
        description="Native DreamX camera keys sampled for this completed chunk."
    )
    inference_seconds: float = MessageField(
        description="Wall-clock seconds spent resetting if needed and generating this chunk."
    )


class DreamXWorldState(InputState):
    """Expose controls shared by one playable DreamX-World session."""

    prompt: str = InputField(
        default="",
        max_length=4096,
        description=(
            "Scene or event prompt queued for the next native chunk. Requires a selected "
            "image; whitespace-only values are rejected by `set_prompt`."
        ),
    )
    _paused: bool = False
    _pressed_keys: frozenset[str] = frozenset()
    _step_requested: bool = False
    _reset_requested: bool = False

    @property
    def paused(self) -> bool:
        """Return whether automatic chunk generation is paused."""
        return self._paused

    @paused.setter
    def paused(self, value: bool) -> None:
        """Set whether automatic chunk generation is paused."""
        self._paused = value
