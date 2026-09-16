# Browser uploads and explicit start

Use the native Runtime endpoint with https://reactor-sandbox.vercel.app/.
The top-bar Start creates a session; the model's Start in Controls begins generation.

## Upload audio

The Sandbox deployment inspected on 2026-09-16 hardcodes `accept="image/*"`
for file parameters. Expand Set Audio, then run in the browser developer console:

```javascript
document.querySelectorAll('input[type="file"]').forEach(el => el.removeAttribute('accept'));
```

This only relaxes the currently rendered file pickers, not backend validation or
authentication. Refreshing or reopening a form may require repeating it.

## Order

1. Connect and wait for Controls.
2. Set Avatar Image: choose a reference, wait for upload, then click Execute.
3. Set Audio: choose WAV/MP3/FLAC, wait for upload, then click Execute.
4. Optionally submit Set Prompt or Set Pose Video with Execute.
5. Optionally adjust Set Generation Options (seed / max_chunks); this never starts generation.
6. Confirm `state_update` contains the selected image/audio names, `ready=true`, and `running=false`.
7. Click the standalone Start in Controls.

Image and audio can be submitted in either order. Selecting a file without
Execute does not select it for the model. Missing image or audio makes Start
return `inputs_required`; no default image is used.

Stop retains inputs for another take; Reset clears them. A running take rejects
input changes. Stop safely waits for the in-flight clip, not instant GPU cancellation.
Track playback pause buttons belong to the SDK/frontend and do not stop inference.

## v0.3.0 SDK migration

Use `set_generation_options` with `{seed, max_chunks}`, then `start` with `{}`.
The former is optional (defaults: seed 420, clip limit 10000; audio may finish earlier).
The previous parameterized start and model pause/resume commands are removed.
The bundled `check_model.py` implements the updated sequence.
