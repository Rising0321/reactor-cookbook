"""Define the public Reactor schema for LingBot-World v1 Fast."""

from __future__ import annotations

from reactor_runtime import (
    InputField,
    InputState,
    MessageField,
    ModelMessage,
    Output,
    Video,
)


class LingBotWorldOutput(Output):
    """Carry one generated LingBot-World RGB frame."""

    main_video: Video


class StateUpdate(ModelMessage):
    """Report the complete shared world state after each accepted transition."""

    prompt: str = MessageField(
        description=(
            "Active scene prompt. A successful `set_prompt` change is encoded for the next "
            "generated `main_video` chunk while the existing visual self-KV history remains."
        )
    )
    image_source: str = MessageField(
        description=(
            "Source of the active anchor image: `built_in` after startup or `random_image`, "
            "and `upload` after `set_image`."
        )
    )
    image_name: str = MessageField(
        description="Filename of the active anchor image used by the current or queued rollout."
    )
    seed: int = MessageField(
        description=(
            "Non-negative random seed for the current or queued rollout. It changes only when "
            "`reset` receives a non-negative seed."
        )
    )
    paused: bool = MessageField(
        description=(
            "Whether continuous generation stops before the next chunk. An in-flight GPU chunk "
            "can finish before the pause takes effect."
        )
    )
    step_queued: bool = MessageField(
        description=(
            "Whether one chunk is queued while paused. It becomes false when that chunk starts "
            "or another playback command cancels it."
        )
    )
    limit_reached: bool = MessageField(
        description=(
            "Whether the rollout exhausted its safe RoPE timeline. Use `reset`, `set_image`, or "
            "`random_image` before requesting another chunk."
        )
    )
    completed_chunks: int = MessageField(
        description=(
            "Number of completed native three-latent chunks in the active world. Chunk 1 emits "
            "9 frames and each later chunk emits 12 frames."
        )
    )
    last_chunk_seconds: float | None = MessageField(
        description=(
            "Wall-clock seconds spent in model generation and causal VAE decode for the most "
            "recent completed chunk. Null before the first chunk of a fresh world."
        )
    )
    next_chunk: int | None = MessageField(
        description=(
            "One-based chunk that newly accepted camera or prompt controls will first affect. "
            "Null after the rollout limit is reached."
        )
    )
    next_chunk_frames: int | None = MessageField(
        description=(
            "Frames emitted by `next_chunk`: 9 for the first causal chunk, 12 thereafter, and "
            "null after the rollout limit is reached."
        )
    )
    max_chunks: int = MessageField(
        description="Maximum chunks available before a fresh anchor rollout is required."
    )
    forward: float = MessageField(
        description=(
            "Active backward-to-forward translation in [-1, 1], sampled at the next chunk "
            "boundary and held until changed or released."
        )
    )
    strafe: float = MessageField(
        description=(
            "Active left-to-right translation in [-1, 1], sampled at the next chunk boundary "
            "and held until changed or released."
        )
    )
    vertical: float = MessageField(
        description=(
            "Active down-to-up translation in [-1, 1], sampled at the next chunk boundary and "
            "held until changed or released."
        )
    )
    pitch: float = MessageField(
        description=(
            "Active downward-to-upward pitch in [-1, 1], sampled at the next chunk boundary and "
            "held until changed or released."
        )
    )
    yaw: float = MessageField(
        description=(
            "Active left-to-right yaw in [-1, 1], sampled at the next chunk boundary and held "
            "until changed or released."
        )
    )
    roll: float = MessageField(
        description=(
            "Active counterclockwise-to-clockwise roll in [-1, 1], sampled at the next chunk "
            "boundary and held until changed or released."
        )
    )


class RolloutLimitReached(ModelMessage):
    """Report that generation paused at the end of the safe timeline."""

    completed_chunks: int = MessageField(
        description="Number of completed chunks when the rollout reached its configured limit."
    )
    max_chunks: int = MessageField(
        description=(
            "Configured limit reached by `completed_chunks`; start a fresh anchor rollout to "
            "continue generation."
        )
    )


class LingBotWorldState(InputState):
    """Expose shared text, camera, and playback controls for one LingBot world."""

    prompt: str = InputField(
        default="",
        max_length=4096,
        description=(
            "Active non-empty scene prompt, up to 4096 characters. A change is encoded at the "
            "next generated chunk boundary without clearing visual self-KV history."
        ),
    )
    forward: float = InputField(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description=(
            "Backward (-1) to forward (1) translation sampled at chunk boundaries and held "
            "until changed or released."
        ),
    )
    strafe: float = InputField(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description=(
            "Left (-1) to right (1) translation sampled at chunk boundaries and held until "
            "changed or released."
        ),
    )
    vertical: float = InputField(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description=(
            "Down (-1) to up (1) translation sampled at chunk boundaries and held until changed "
            "or released."
        ),
    )
    pitch: float = InputField(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description=(
            "Downward (-1) to upward (1) pitch sampled at chunk boundaries and held until "
            "changed or released."
        ),
    )
    yaw: float = InputField(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description=(
            "Left (-1) to right (1) yaw sampled at chunk boundaries and held until changed or "
            "released."
        ),
    )
    roll: float = InputField(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description=(
            "Counterclockwise (-1) to clockwise (1) roll sampled at chunk boundaries and held "
            "until changed or released."
        ),
    )
    paused: bool = InputField(
        default=True,
        description=(
            "Whether generation waits before the next chunk. Startup and fresh images remain "
            "paused but automatically queue one preview chunk."
        ),
    )
    _step_requested: bool = False
    _restart_requested: bool = True
    _limit_reached: bool = False
