"""Audio-driven avatar takes; module name avoids upstream's liveavatar package."""

# ruff: noqa: B008 -- InputField defaults declare the Reactor wire schema.
from __future__ import annotations

import asyncio
import io
import subprocess
import tempfile
from collections.abc import AsyncGenerator
from pathlib import Path

from PIL import Image
from reactor_runtime import (
    ClientInfo,
    CommandError,
    InputField,
    ReactorPipeline,
    UploadedFile,
    connected,
    event,
    session_ended,
    session_started,
)

from liveavatar_assets import WORK, configure_cache_environment
from liveavatar_types import (
    ChunkComplete,
    GenerationEnded,
    InputAccepted,
    LiveAvatarOutput,
    LiveAvatarState,
    StateUpdate,
    TakeChanged,
)


class LiveAvatar(ReactorPipeline):
    state: LiveAvatarState
    fps = 25
    buffer_size = 48

    def __init__(self):
        super().__init__()
        self._backend = None
        self._directory = None
        self._image: Path | None = None
        self._audio: Path | None = None
        self._pose: Path | None = None
        self._pending = False

    def load(self, config_path: Path | None = None):
        configure_cache_environment()
        from liveavatar_backend import LiveAvatarBackend

        self._backend = LiveAvatarBackend()

    @session_started
    async def on_session_started(self):
        configure_cache_environment()
        self._directory = tempfile.TemporaryDirectory(prefix="session-", dir=WORK)
        self._image = self._audio = self._pose = None
        self._pending = False

    @session_ended
    async def on_session_ended(self):
        if self._backend is not None:
            await asyncio.to_thread(self._backend.close)
        if self._directory is not None:
            self._directory.cleanup()
            self._directory = None
        self._image = self._audio = self._pose = None
        self._pending = False

    @connected
    async def on_connected(self, client: ClientInfo):
        await client.send(StateUpdate.from_state(self.state))

    def _require_idle(self):
        if self.state._running:
            raise CommandError(
                "take_running", "Use `stop` before replacing take conditions."
            )

    def _path(self, name: str) -> Path:
        if self._directory is None:
            raise CommandError("session_required", "A session must be active.")
        return Path(self._directory.name) / name

    async def _send_state_update(self):
        await self.send(StateUpdate.from_state(self.state))

    @event(
        name="set_avatar_image",
        description="Select an uploaded identity image while idle. Replies `input_accepted` and broadcasts `state_update`; invalid images or an active take return `command_error`. Required before `start`.",
    )
    async def set_avatar_image(
        self,
        image: UploadedFile = InputField(
            moderate=True,
            description="PNG, JPEG or WebP reference, up to 25 MiB; aspect ratio is fitted by the model. Applies at the next `start`; no default image is supplied.",
        ),
    ) -> InputAccepted:
        self._require_idle()
        try:
            if not image.data or image.size > 25 * 1024**2:
                raise ValueError("Image must be nonempty and at most 25 MiB")
            with Image.open(io.BytesIO(image.data)) as decoded:
                if decoded.width * decoded.height > 40_000_000:
                    raise ValueError("Image exceeds 40 million pixels")
                decoded.convert("RGB").save(self._path("reference-new.png"))
                self._path("reference-new.png").replace(self._path("reference.png"))
        except Exception as exc:
            raise CommandError("invalid_image", str(exc)) from exc
        self._image = self._path("reference.png")
        self.state._image_name = image.name
        await self._send_state_update()
        return InputAccepted(field="avatar_image")

    @event(
        name="set_audio",
        description="Select uploaded driving audio while idle. Replies `input_accepted` and broadcasts `state_update`; unreadable audio or an active take return `command_error`. Required before `start`.",
    )
    async def set_audio(
        self,
        audio: UploadedFile = InputField(
            moderate=True,
            description="WAV, FLAC or MP3 audio up to 100 MiB, decoded to mono 16 kHz. Drives and is played with the next take; video ends at the audio or requested clip limit.",
        ),
    ) -> InputAccepted:
        self._require_idle()
        if not audio.data or audio.size > 100 * 1024**2:
            raise CommandError(
                "invalid_audio", "Audio must be nonempty and at most 100 MiB"
            )
        raw, target = self._path("audio-upload"), self._path("driving-new.wav")
        raw.write_bytes(audio.data)
        result = await asyncio.to_thread(
            subprocess.run,
            [
                "ffmpeg",
                "-v",
                "error",
                "-nostdin",
                "-y",
                "-i",
                str(raw),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                str(target),
            ],
            capture_output=True,
        )
        if result.returncode:
            raise CommandError("invalid_audio", "Cannot decode the uploaded audio")
        import soundfile as sf

        info = sf.info(target)
        if info.duration < 48 / 25:
            raise CommandError(
                "audio_too_short",
                "Supply at least 1.92 seconds of audio for one native clip",
            )
        target.replace(self._path("driving.wav"))
        self._audio = self._path("driving.wav")
        self.state._audio_name = audio.name
        await self._send_state_update()
        return InputAccepted(field="audio")

    @event(
        name="set_pose_video",
        description="Select or clear a pose-conditioning upload while idle. Replies `input_accepted` and broadcasts `state_update`; unreadable videos or an active take return `command_error`. The reference image and audio remain required.",
    )
    async def set_pose_video(
        self,
        pose_video: UploadedFile | None = InputField(
            default=None,
            moderate=True,
            description="Optional MP4 containing a prepared pose sequence, not a request to extract pose from ordinary footage. Null clears it; applies at the next `start`.",
        ),
    ) -> InputAccepted:
        self._require_idle()
        if pose_video is None:
            self._pose = None
            self.state._pose_name = None
        else:
            if not pose_video.data or pose_video.size > 100 * 1024**2:
                raise CommandError(
                    "invalid_pose", "Pose video must be nonempty and at most 100 MiB"
                )
            path = self._path("pose-new.mp4")
            path.write_bytes(pose_video.data)
            result = await asyncio.to_thread(
                subprocess.run,
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width",
                    "-of",
                    "csv=p=0",
                    str(path),
                ],
                capture_output=True,
            )
            if result.returncode or not result.stdout.strip():
                raise CommandError(
                    "invalid_pose", "Upload must contain a readable video track"
                )
            path.replace(self._path("pose.mp4"))
            self._pose = self._path("pose.mp4")
            self.state._pose_name = pose_video.name
        await self._send_state_update()
        return InputAccepted(field="pose_video")

    @event(
        name="set_prompt",
        description="Set optional scene text while idle. Replies `input_accepted` and broadcasts `state_update`; an active take returns `command_error`. Text describes appearance or motion, not speech synthesis. Negative text is accepted for compatibility but does not influence this four-step model.",
    )
    async def set_prompt(
        self,
        prompt: str = InputField(
            default="",
            max_length=4096,
            moderate=True,
            description="Optional scene and performance description applied at `start`; empty uses image and audio without additional scene text.",
        ),
        negative_prompt: str = InputField(
            default="",
            max_length=4096,
            moderate=True,
            description="Compatibility text passed at `start`; empty retains the upstream default. The released four-step model does not apply negative conditioning, so this field does not change the video.",
        ),
    ) -> InputAccepted:
        self._require_idle()
        self.state._prompt = prompt
        self.state._negative_prompt = negative_prompt
        await self._send_state_update()
        return InputAccepted(field="prompt")

    @event(
        name="set_generation_options",
        description="Set sampling seed and clip limit while idle without starting generation. Applies on the next explicit `start`. Replies `input_accepted` and broadcasts `state_update`; an active take returns `command_error`.",
    )
    async def set_generation_options(
        self,
        seed: int = InputField(
            default=420,
            ge=0,
            le=2147483647,
            description="Sampling seed for the next `start`; each successive clip uses seed plus its zero-based clip index. Does not start generation.",
        ),
        max_chunks: int = InputField(
            default=10000,
            ge=1,
            le=10000,
            description="Maximum clips for the next `start`; defaults to 10000, and audio length may end the take earlier. Does not start generation.",
        ),
    ) -> InputAccepted:
        self._require_idle()
        self.state._seed, self.state._max_chunks = seed, max_chunks
        await self._send_state_update()
        return InputAccepted(field="generation_options")

    @event(
        name="start",
        description="Begin generation explicitly after all desired inputs have been accepted. Valid only while idle with both `set_avatar_image` and `set_audio` completed; prompt and generation options are optional. Replies `take_changed` and broadcasts `state_update`; missing inputs or an active take return `command_error`. Uploading or adjusting inputs alone never starts generation.",
    )
    async def start(self) -> TakeChanged:
        self._require_idle()
        if self._image is None or self._audio is None:
            raise CommandError(
                "inputs_required", "Upload reference image and driving audio first"
            )
        self.state._running = True
        self.state._chunks = self.state._frames = 0
        self.state._error = None
        self._pending = True
        await self._send_state_update()
        return TakeChanged(action="start")

    @event(
        name="stop",
        description="End an active or idle take, retaining uploads and text for another `start`. Replies `take_changed` and broadcasts `state_update`; waits for an in-flight clip to finish safely and flushes queued output.",
    )
    async def stop_take(self) -> TakeChanged:
        if self._backend is not None:
            await asyncio.to_thread(self._backend.close)
        self.state._running = self._pending = False
        self.output.flush()
        await self._send_state_update()
        return TakeChanged(action="stop")

    @event(
        name="reset",
        description="End the take and clear uploaded inputs, prompts and progress in any session state. Replies `take_changed` and broadcasts `state_update`; upload a new image and audio before `start`.",
    )
    async def reset(self) -> TakeChanged:
        await self.stop_take()
        self._image = self._audio = self._pose = None
        self.state = LiveAvatarState()
        await self._send_state_update()
        return TakeChanged(action="reset")

    async def inference(self) -> AsyncGenerator[LiveAvatarOutput | None, None]:
        while True:
            if not self.state._running:
                yield None
                continue
            try:
                if self._pending:
                    self._pending = False
                    await asyncio.to_thread(
                        self._backend.start,
                        image=self._image,
                        audio=self._audio,
                        pose=self._pose,
                        prompt=self.state._prompt,
                        negative_prompt=self.state._negative_prompt,
                        seed=self.state._seed,
                        max_chunks=self.state._max_chunks,
                    )
                # Runtime calls handlers between inference turns, not within GPU work.
                result = await asyncio.to_thread(self._backend.next)
                if result is None:
                    await asyncio.to_thread(self._backend.close)
                    self.state._running = False
                    await self.send(GenerationEnded(reason="complete"))
                    await self._send_state_update()
                    yield None
                    continue
                video, audio = result
                self.state._chunks += 1
                self.state._frames += len(video)
                await self.send(
                    ChunkComplete(chunk=self.state._chunks, frames=len(video))
                )
                await self._send_state_update()
                yield LiveAvatarOutput(main_video=video, main_audio=audio)
            except Exception as exc:  # noqa: BLE001 - report upstream failure and release take state
                await asyncio.to_thread(self._backend.close)
                self.state._running = False
                self.state._error = str(exc) or type(exc).__name__
                await self.send(GenerationEnded(reason=self.state._error))
                await self._send_state_update()
                yield None
