"""Client-facing avatar inputs, audiovisual tracks and session snapshots."""

from reactor_runtime import Audio, InputState, MessageField, ModelMessage, Output, Video


class AvatarAudio(Audio):
    sample_rate = 48000


class LiveAvatarOutput(Output):
    main_video: Video
    main_audio: AvatarAudio


class LiveAvatarState(InputState):
    _image_name: str | None = None
    _audio_name: str | None = None
    _pose_name: str | None = None
    _prompt: str = ""
    _negative_prompt: str = ""
    _seed: int = 420
    _running: bool = False
    _chunks: int = 0
    _frames: int = 0
    _max_chunks: int = 10000
    _error: str | None = None


class StateUpdate(ModelMessage):
    """Emitted on connection and after each accepted change or completed clip."""

    image_name: str | None = MessageField(
        description="Selected uploaded reference filename, or null until `set_avatar_image`."
    )
    audio_name: str | None = MessageField(
        description="Selected uploaded driving audio filename, or null until `set_audio`."
    )
    pose_name: str | None = MessageField(
        description="Optional uploaded pose video filename; null means audio-driven motion only."
    )
    prompt: str | None = MessageField(
        description="Scene description for the next take; null means no text condition."
    )
    negative_prompt: str | None = MessageField(
        description="Requested exclusions, or null to use the released model default."
    )
    seed: int = MessageField(
        description="Seed read at `start`; retained by `stop` and restored to 420 by `reset`."
    )
    ready: bool = MessageField(
        description="Whether both reference image and driving audio have been selected."
    )
    running: bool = MessageField(
        description="Whether a take is active after an accepted explicit `start`."
    )
    completed_chunks: int = MessageField(
        description="Number of clips delivered in the current or most recent take."
    )
    frames: int = MessageField(
        description="Video frames delivered this take at 25 frames per second."
    )
    max_chunks: int = MessageField(
        description="Clip limit selected by `set_generation_options` and read at `start`; audio can end the take earlier."
    )
    error: str | None = MessageField(
        description="Most recent generation failure, or null when no failure occurred."
    )

    @classmethod
    def from_state(cls, state: LiveAvatarState) -> "StateUpdate":
        return cls(
            image_name=state._image_name,
            audio_name=state._audio_name,
            pose_name=state._pose_name,
            prompt=state._prompt or None,
            negative_prompt=state._negative_prompt or None,
            seed=state._seed,
            ready=bool(state._image_name and state._audio_name),
            running=state._running,
            completed_chunks=state._chunks,
            frames=state._frames,
            max_chunks=state._max_chunks,
            error=state._error,
        )


class InputAccepted(ModelMessage):
    """Emitted as the reply when an uploaded input, prompt, or generation options are accepted."""

    field: str = MessageField(
        description="Accepted condition: avatar_image, audio, pose_video, prompt, or generation_options."
    )


class TakeChanged(ModelMessage):
    """Emitted as the reply when start, stop, or reset succeeds."""

    action: str = MessageField(
        description="Accepted command wire name; state details follow in `state_update`."
    )


class ChunkComplete(ModelMessage):
    """Emitted once for each generated clip sent on main_video and main_audio."""

    chunk: int = MessageField(
        description="One-based completed clip number, reset at `start`."
    )
    frames: int = MessageField(
        description="Frames in this clip: 45 in the first clip, then 48 at 25 FPS."
    )


class GenerationEnded(ModelMessage):
    """Emitted when a take reaches its limit or fails during generation."""

    reason: str = MessageField(
        description="complete for normal completion; otherwise the reported failure."
    )
