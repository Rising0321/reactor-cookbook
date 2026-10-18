Clear the cave — playable proxy + AI stream

Start the existing server: npm start
Game: http://127.0.0.1:4174/game.html?game=spiders
Local play: http://127.0.0.1:4174/game.html?game=spiders&play=proxy
Gallery: http://127.0.0.1:4174/gallery.html

WASD moves, arrow keys rotate the view, hold F to swing the axe, Space jumps.
No mouse aiming, clicking attacks, or pointer lock. Click Play proxy for local
gameplay, or Start AI session to use the saved cave first frame. Demo pilot
uses the same input and combat rules. Defeat all six spiders to win; health
reaching zero ends the run. Local and AI sessions support replay.

Combat runs at 24 fixed steps/second. Each spider takes two hits. One swing
has one impact after 0.29 seconds, a 0.8 second repeat cooldown, 3.3 unit reach,
and a view-direction cone. Impact tests the current aim rather than a scripted
target. Spiders approach, telegraph a bite, and damage the player if still
within reach when the bite lands. Backpedaling or a well-timed jump avoids it.
Living spiders block movement and keep their centers spaced. Fallen spiders
remain on the ground. The winding tunnel limits movement; no spiderwebs.

game.js is pure simulation, snapshot restore, demo controls and state-derived
video prompts. scene.js adapts the original cave geometry without changing
the offline caveaxe scene/video. adapter.js supplies the shared app's game
identity, first-frame key, prompts, copy and HUD statistics.

First-frame caption describes visible axe, spiders, cave rocks and torches.
Global prompt holds the setting and identity stable. Dynamic prompt describes
the actual same input group: turning, movement, swing, hit/miss, kill, bite,
dodge and end state. All UI and prompt text is English.

The saved cave first frame and exact snapshot are selected via
data/games/spiders/latest.json; browser cover storage uses
spiders-cover-v1, separate from the freeze scene. New browser sessions load
the saved cover automatically. The Gallery's AI button does not consume a
generation session until Start AI session is clicked.

Transport and recordings are shared in src/runtime/ and server/; see the root README.
New states.jsonl groups and telemetry summaries include gameId: spiders.
Game View / Compare View, independent UI/actions and fullscreen still work.
AI live UI has a static objective/crosshair plus a health bar and numeric
health, labeled LIVE · AHEAD OF VIDEO. Health follows the current game state,
including spider bites, death and reset; it is not synchronized to delayed
video. The health overlay follows Game UI in both views and fullscreen.
Health and kill counts also appear on the proxy HUD and completed-session
card. Action Overlay is explicitly LIVE INPUT, ahead of video. Model input
images and recordings contain no HUD.

Checks:
npm run verify:interactive-spiders       combat, dodge, collision and replay
npm run check:interactive-spiders        gallery, keyboard, local replay,
                                        cover isolation and mocked streaming
npm run smoke:interactive-spiders        real services; only while idle
Add -- --generate-cover to the smoke command to replace the saved cover,
or -- --cover-only to only prepare one. Real smoke uses up to 40 input groups.

2026-10-10 real service check: six spiders defeated, 381/381 frames received
and decoded, zero RTP loss and NACKs. State/action replay matches within
1e-8 across Chrome and Node (their trigonometric roundoff can differ).
See data/games/spiders/qa/stream-verification.json and latest.json
for the saved WebM and JSONL state/action records. Received WebM is a browser
re-encoding of the video track, not an exact archive of the original H.264
frames. This short-run result is not a sustained-network guarantee.
