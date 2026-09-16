# Audio fix — 2026-09-16, v0.2.0

## Cause and correction

Runtime 3.2.5's outbound WebRTC path sends PCM as 48 kHz but does not resample
the previous 16 kHz waveform. This caused approximately 3x-speed playback,
pitch changes and underrun silence. It was not an encoded bitrate mismatch.

`liveavatar_audio.playback_audio` now polyphase-resamples the full take from
16 to 48 kHz before clipping into output chunks. The track declares 48 kHz;
45 frames have 86400 samples and 48 frames have 92160 samples at native 25 FPS.
Resampling once avoids independent per-chunk filter boundary artifacts.
The uploaded conditioning WAV remains mono 16 kHz and the upstream inference,
motion history, cache sizes and cache updates are unchanged.

## Verification

- 16 CPU tests passed, Ruff passed, schema rendered with 48000 Hz output.
- Actual Runtime 3.2.5 → SDK loopback: synthetic CPU video and explicit uploaded
  speech; received 141 video frames and mono 48000 Hz PCM.
- Three speech templates (0.75, 2.25, 4.0 seconds) matched at correlations
  0.6732, 0.9589, 0.9956. The old wrong-rate interpretation matched only
  0.2108, 0.3399, 0.3262. Opus is lossy, so bit-identical PCM is not expected.
- Separate uploaded 1000 Hz tone remained 1000 Hz in received SDK PCM,
  rather than the old erroneous 3000 Hz.
- Neither verification loaded model weights or used a GPU. The CPU service was
  stopped afterward. This validates the audio path, not a new GPU inference run
  or a subjective listening assessment.

Artifacts and repeatable speech analysis:
`/opt/dlami/nvme/ruixing/liveavatar-test-kit-20260916/verification/`
and `../verify_received_audio.py` relative to that directory.
Historical GPU recordings remain unchanged as regression evidence.

## Remaining limitation

The unaccelerated model can still cause real-time gaps while generating the next
clip. Correcting sample rate does not make inference real-time. Always inspect
received `audio.wav`; `take.mp4` uses original uploaded audio for its offline
model-time preview and does not verify live audio transport.
