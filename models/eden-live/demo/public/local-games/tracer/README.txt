Blink through the fight — Tracer-inspired playable proxy

Server: npm start
Game: http://127.0.0.1:4174/game.html?game=tracer
Local play: http://127.0.0.1:4174/game.html?game=tracer&play=proxy
Gallery: http://127.0.0.1:4174/gallery.html

WASD moves, arrows look, F fires alternating pulse pistols, Space jumps.
Shift blinks in the movement direction, or forward while standing still.
E recalls, R reloads. No mouse aiming or firing. Extra ability key presses
are queued until the next simulation step; holding Shift/E does not retrigger.
Editing prompt fields, leaving the page, and pausing clear gameplay input.

Rules:
- Six robots in three successive pairs. Clear all six to win; zero health loses.
- Player cyan projectiles and enemy orange projectiles have finite travel time.
  Segment collision tests prevent fast bullets from tunneling through targets
  or cover. Robots take eight hits; enemy bullets deal 12 health damage.
- 24 shared rounds. Reload takes 1.15 simulation seconds. Holding fire with an
  empty magazine starts a reload. Shots alternate left/right at eight per second.
- Blink moves five world units over 0.25 seconds. Three charges, one replenished
  every three seconds. Movement uses small collision steps and cannot pass through
  crates or walls. Blink ignores incoming player damage during the movement.
- Recall retraces the previous three seconds of player movement in 0.65 seconds,
  then restores the saved health. Cooldown is eight seconds. Enemy deaths, ammo,
  charges and the world clock do not rewind. Incoming damage is ignored while
  recalling. The bounded history and recall path are included in snapshots.
- Demo uses the same inputs and rules. It uses blink, recall and reload and can
  clear the three waves. Pure simulation and snapshot replay are deterministic
  in one runtime; cross-runtime floating point checks allow 1e-8 error.

Files:
game.js — state, input, projectile collision, abilities, demo, derived prompts.
scene.js — adapted original street, paired guns and robots; deterministic effects.
adapter.js — shared app configuration, extra action keys, copy and cover storage.
The original offline tracer.js, timeline and video remain separate.

The shared app supports Game View, Compare View, fullscreen and independent
Game UI/Action Overlay switches. Proxy HUD shows health, ammo, wave, blink
charges and recall cooldown. Generated video shows the static objective and
crosshair, plus a health bar and numeric health that follow the current game
state (including damage and recall), labeled LIVE · AHEAD OF VIDEO.
The health overlay follows Game UI in both views and fullscreen. Its live
action keys are explicitly ahead of video, not synchronized
to delayed output. Completed-session UI shows kills and remaining health.
Model condition PNGs and recorded WebM contain no UI.

First-frame, global and dynamic prompts are English. Dynamic text comes from
the same state group as the three uploaded images: shot, hit, kill, blink
direction, recall, reload and wave transitions. A group sends the full prompt
only with the third image. Records use gameId: tracer; transport is unchanged.

Cover manifest: data/games/tracer/latest.json
Browser cover key: tracer-cover-v1 (separate from freeze and spiders).
Shared image/session records: data/records/cover-* and stream-*.
The manifest links the saved example recording and input-state JSONL.

Checks:
npm run verify:interactive-tracer       pure mechanics and replay
npm run check:interactive-tracer        browser controls, Gallery and mocked AI
npm run smoke:interactive-tracer        real AI session; requires idle services
For cover only: npm run smoke:interactive-tracer -- --cover-only
To reroll: add --generate-cover --seed=<integer>.
To regenerate with current source prompts: add --default-prompts.

See ../../../README.md for streaming v2, recording details,
delivery metrics and the current lack of a video-to-game-frame timing contract.

Real service validation (2026-10-10): full six-robot victory, two blinks, one
recall and two reloads; state/action replay and third-frame prompts passed.
An isolated run received 573 of 573 generated frames but decoded 539.
Browser-side video drops remain; the second run did not isolate the cause
to concurrent browser tests. This is not a zero-drop delivery result. See
data/games/tracer/qa/verification.json and stream-result.json.
