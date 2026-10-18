Burn the skyline — Godzilla-inspired playable city destruction

Server: npm start
Game: http://127.0.0.1:4174/game.html?game=godzilla
Local: http://127.0.0.1:4174/game.html?game=godzilla&play=proxy
Gallery: http://127.0.0.1:4174/gallery.html

WASD moves relative to the monster's heading. Left/right arrows turn the
monster and following camera; up/down arrows raise/lower the breath aim.
Hold F to breathe fire. Tap Space to stomp (this game does not jump).
No mouse aiming. Prompt editing, pause and loss of focus clear inputs.

Destroy the six bronze defense towers, recognizable by rooftop cannons and
amber beacons. Other grey buildings are destructible but do not count toward
the objective. Health reaching zero loses; victory and defeat support replay.
- Breath: 36-unit ray from the mouth area, 60 damage per simulation second.
  The nearest intact building blocks the jet. Aiming away or beyond range
  misses. Flame presentation is separate from the damage simulation.
- Heat: increases by 26/second while firing and cools by 22/second otherwise.
  At 100, firing locks until heat falls to 30. Holding F cannot bypass it.
- Stomp: 65 damage to nearby structures within 7.5 units of their footprint,
  with a 3.5-second cooldown. Holding Space does not repeat the stomp.
- Ramming: moving against a building deals 45 damage/second while its solid
  footprint prevents walking through. Collapsed rubble can be traversed.
- Towers telegraph shots with red beacons for 0.75 seconds, locking a point
  before firing. Slow orange shells deal 8 health damage. Keep moving to
  dodge. Intact buildings block shells; destroying a tower cancels its charge.
- All buildings retain destruction state. Chunk and dust animation follows
  actual collapse times, not the original offline sample's timeline.

The pure 24 Hz simulation stores projectiles, heat, cooldowns, tower AI,
building health and collapse times in snapshots. Demo pilot uses the same
inputs. Action/state replay is deterministic in a runtime; cross-runtime
floating point validation allows 1e-8 error.

The camera follows above and behind the monster and shortens its offset if
an intact building blocks the view. The third-person view intentionally has
no first-person crosshair. Health, heat, tower progress and stomp/cooling
status follow Game UI; Action Overlay is independent. Generated-view values
show live simulation state, ahead of delayed video. Model PNGs and received
video recordings contain no HUD; exact video-to-state timing is unavailable.

Files:
game.js — combat, movement, destruction, projectiles, demo and dynamic prompts.
scene.js — adapted original kaiju, city geometry, flame, rubble and camera.
adapter.js — shared application controls, status UI, cover storage and endings.
The original godzilla.js, godzilla-timeline.js and offline video are retained.

Cover key: godzilla-cover-v1. Manifest: data/games/godzilla/latest.json.
Cover images and streaming records use data/records/cover-* and
stream-*. States and telemetry identify gameId: godzilla. Full prompts describe
the same simulation group as the condition images and are sent on image three.

Checks:
npm run verify:interactive-godzilla   mechanics, collisions, heat and replay
npm run check:interactive-godzilla    Chrome controls/UI with mocked services
npm run smoke:interactive-godzilla    real services, only if idle
Append -- --cover-only to generate/load only a matching cover. To replace a
cover, append -- --generate-cover --default-prompts --seed=<integer>.
The mock UI check uses a proxy PNG as its fake cover, not model output.

Real service validation (2026-10-11): six-tower victory, 68 remaining health,
two stomps, complete state/action replay within 1e-8 across Chrome and Node,
and third-condition prompt checks passed. Received and decoded 405/405 frames,
zero reported frames dropped, RTP packets lost and NACK feedback messages.
The browser-reencoded WebM fully decodes. This is one short measured session,
not a guarantee of sustained network delivery or exact original-frame recording.
See qa/stream-verification.json and the recording linked in latest.json.

Visual review: fire and building collapse follow gameplay. AI-generated rubble
still occasionally resembles monster fragments despite single-monster prompt
constraints, and generated geometry may differ from the authoritative proxy.
