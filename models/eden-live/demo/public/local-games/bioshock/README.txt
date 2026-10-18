Shock the depths — Bioshock-inspired playable proxy

Server: npm start
Game: http://127.0.0.1:4174/game.html?game=bioshock
Local play: http://127.0.0.1:4174/game.html?game=bioshock&play=proxy
Gallery: http://127.0.0.1:4174/gallery.html

Keyboard: WASD moves, arrows look, hold F to swing the wrench, tap E for an
electro bolt, Space jumps. No mouse aiming or attacks. Holding E casts once;
release and press again for another cast. Prompt editing, pause and loss of
focus clear pending inputs. Demo pilot uses the same input and combat rules.

Clear four masked enemies in the underwater machine room. Zero health loses.
The first two approach across dry tiles. Two more guard the shallow rear pool.
- Wrench: 2.8-unit reach, view-direction cone, impact after 0.29 seconds,
  0.85-second repeat interval. Each swing hits once. Turning during the
  windup changes the target. Normal hits deal 35 damage; stunned hits deal 60.
- Electro bolt: 20 EVE, 0.9-second cooldown, 10-unit range. Direct hits deal
  10 damage and stun for 2.2 seconds, interrupting pending enemy strikes.
  Aim at the water surface to electrify it, rather than directly at a guard.
- Water conducts for 1.6 seconds. Every 0.5 seconds it deals 20 damage to
  each enemy in the pool, and 12 to a player standing in it. Jumping above
  the surface avoids a damage tick. Stay on dry tiles when casting.
- Enemies telegraph a close strike for 0.65 seconds, then deal 16 damage
  if still in reach. Back away, jump or stun them to avoid the hit.
- EVE slowly recovers at 6 per simulation second after 3 seconds without
  casting. Empty-energy casts spend nothing. Health does not regenerate.
- Generator, boilers, walls, columns and living enemy bodies are solid.
  Obstacles block electric bolts and melee line of sight. Water guards stay
  in the pool. Corpses remain where they fell and do not block movement.

The 24 Hz simulation is independent of the renderer. Full snapshots include
enemy AI, attack/cast timing and water damage cadence for deterministic replay.
This is a small original combat prototype inspired by the existing offline
sample, not a recreation of the complete commercial game.

Game View and Compare View show health and EVE with Game UI enabled. Generated
view counters show LIVE state, ahead of delayed AI video. Action Overlay is
independent. Model PNG inputs and received-track WebM recordings contain no HUD.
Precise video-to-state synchronization is not part of the upstream contract.

game.js: simulation, geometry collision, demo inputs and state-derived prompts.
scene.js: adapted offline machine room, moving enemies, wrench and electricity.
adapter.js: shared application configuration, E action, health/EVE and end cards.
Original offline source, timeline, overlays and videos remain in the separate proxy-video-generator repository.

Cover storage: bioshock-cover-v1; manifest: data/games/bioshock/latest.json.
Images and streaming records use the shared data/records directory.
New states and telemetry identify gameId: bioshock. Full prompts are submitted
only with the third condition image and describe the actual state group.

Checks:
npm run verify:interactive-bioshock   mechanics, collision, demo and replay
npm run check:interactive-bioshock    real Chrome, mocked image/video services
npm run smoke:interactive-bioshock    real first-frame/streaming services if idle
Append -- --cover-only to generate/load only a matching first frame. Append
-- --generate-cover --default-prompts --seed=<integer> to replace the cover.
The mock check deliberately uses a proxy PNG as its fake cover; QA files with
mock in the name do not demonstrate model output. Real service results are
saved separately as qa/stream-result.json and linked by latest.json.

Real service validation (2026-10-11): four-enemy victory, four combo hits,
two water casts, exact state/action replay within 1e-8 across Chrome and Node,
and third-frame prompt checks passed. Received 273 generated frames, decoded
263, dropped 10 in the browser; zero reported RTP loss or NACKs. The received
WebM decodes, with a duplicate output timestamp warning from FFmpeg. This is
a browser re-encoding, not a frame-exact archive or a zero-drop stream result.
See qa/stream-verification.json and the recording linked in latest.json.
