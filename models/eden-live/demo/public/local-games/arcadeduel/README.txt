Break the guard — interactive arcade duel

Original reference in proxy-video-generator: src/arcadeduel.js and arcadeduel-timeline.js, the intense
blue/orange combo fighting proxy. Its original offline animation is preserved.
The interactive scene reuses its fighter geometry, IK helpers and arena.

Start: npm start
Play: /game.html?game=arcadeduel&play=proxy
AI: /game.html?game=arcadeduel

Single player controls the blue fighter (initially left), against orange CPU.
A/D move; hold F for jab/cross/spinning-kick chains; hold G to guard;
Space jumps; E uppercut costs 15 energy; R wave costs 30; Shift dash costs 10.
Energy recovers at 5/second and successful hits restore 9. Guard blocks HP
damage but consumes guard strength. Breaking guard causes a brief stun.
Attacks check facing, range and height; waves travel and are consumed on hits.
Dash grants brief invulnerability. Grounded fighters cannot pass through one
another. Jumping and launches allow aerial attacks; the arena has solid bounds.
Both fighters start at 100 HP. Win by KO or higher HP when 60 seconds expires;
a tie is a round without a win. KO holds the scene for one second before results.

CPU is deliberately relaxed for approximately one second of video latency:
- Ground speed 1.35 versus the player's 2.6 units/second.
- Samples player position when making a decision, usually every 30 frames;
  attack decisions wait 60 frames. Never reads the player's attack buttons.
- Adds 30 frames to every attack startup: 1.375 seconds before a jab can hit,
  1.458 before an uppercut, and 1.667 before releasing a wave.
- An orange floor ring and raised/retracted hands telegraph attacks; the live
  overlay also flags CPU windup/attack. Real input-to-video latency is variable.
- Deals ceil(65% of the player's base damage). No immediate reactive counters.

Game View shows both HP bars, energy, guard and timer. These are authoritative
live state, ahead of AI video. Proxy capture excludes HUD and input overlays.
Generated character appearances can differ from the proxy team colors.
Demo pilot uses the same inputs and damage rules as human play.

game.js: deterministic 24 Hz combat, AI, snapshots and state-derived prompts.
scene.js: fighter poses, projectiles, telegraph, impact sparks, dash ghosts, HUD.
adapter.js: shared app integration, held guard, side-view controls and endings.
Cover manifest: data/games/arcadeduel/latest.json.
Cover/snapshot and stream recordings use data/records/cover-* and
stream-*; recording rows carry gameId: arcadeduel. Prompts are sent only on the
third condition frame and describe that same group of simulation states.

Checks:
npm run verify:interactive-arcadeduel  combat, delayed CPU, reaction guard, replay
npm run check:interactive-arcadeduel   browser/UI and full stream with mocks
npm run smoke:interactive-arcadeduel   real cover and complete AI match, if idle
Smoke options: --cover-only, --generate-cover, --default-prompts, --seed=<uint>.
Mock first frames use proxy PNGs; only smoke output is actual generated video.

Validation (2026-10-13): mechanics and browser checks passed, including holding
guard, a guard input after one second of CPU windup, clean condition captures,
mobile/fullscreen controls, mock victory/replay and switching back to Bioshock.
Shared display-mode regression passed. Real AI demo completed an 11-hit combo
and KO: 213/213 received and decoded, zero dropped frames, RTP loss and NACKs.
State/action replay and third-frame prompts matched; the WebM fully decodes.
Visual inspection confirmed impacts and the final KO fall. The generated cover
uses grey/red clothing rather than the proxy blue/orange; team labels refer to
game state. This is a short measured session, not a long-run network guarantee.
See qa/stream-verification.json and the video linked in latest.json.
