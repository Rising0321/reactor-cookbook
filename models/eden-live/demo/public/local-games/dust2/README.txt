Clear the passage — Dust2-style interactive desert FPS

Reference in the original proxy-video-generator repository: src/dustfps.js and dustfps-timeline.js. Reuses the original rifle,
hands, soldiers, sandy alley, arch, crates and courtyards. Original offline
animation is retained. This is the existing small Dust2-inspired proxy layout.

Start: npm start
Play: /game.html?game=dust2&play=proxy
AI: /game.html?game=dust2

WASD move; arrows aim; hold F to fire; R reload; E throw flashbang; Space jump.
30-round magazine, 90 reserve rounds, 1.8-second reload. Empty magazines auto
reload when reserve remains. Rifle fires six rounds/second, three hits per foe.
A 4.9-degree assist cone only selects visible targets; walls/crates block shots.
Space jumps for dodging; solid obstacles cannot be climbed by jumping.

Progression:
1. Clear two soldiers in the alley and move through the arch (z < -5.8).
2. Clear two courtyard soldiers, then take the right passage (x >= 12).
3. Clear two stronghold soldiers, followed by two reinforcements after 1.2s.
4. With all eight defeated, reach the green marker at x=22, z=-24.2. Remain
   within 1.5 units for one second to extract. Merely clearing enemies is not a win.

Enemy behavior accommodates approximately one second of video latency:
- At most two soldiers prepare/fire at a time.
- Two seconds of visible weapon-raising before any shot. HUD warns ENEMY AIMING.
- Three-shot bursts aim at the position captured at windup start, not the latest
  player position. Slow visible projectiles use swept collision and deal 8 damage.
- 3.2-second recovery after bursts. Losing current sight cancels pending fire.
- No shots or aim assistance through opaque cover. Movement respects walls,
  crates, barrels and living soldiers. The arch opening is traversable; narrow
  slabs approximate the curved solid upper arch for projectile occlusion.

Two flashbangs per run. A thrown grenade bursts after 0.65s; enemies within ten
units with unobstructed sight are blinded for four seconds. Pending bursts are
cancelled and a fresh two-second windup is required after recovery. There is no
self-blind mechanic. An occluded flash cannot disable enemies through walls.

HUD: player health, magazine/reserve, remaining flashes, kills and current area.
Live HUD/input is ahead of generated video. Proxy captures exclude all HUD and
keyboard overlays. AI scenery can add visual details (for example stairs) that
are not part of the authoritative collision map; gameplay uses proxy geometry.

game.js — deterministic 24 Hz movement/combat/AI, snapshots and dynamic prompts.
scene.js — adapted original scene, gunfire, reload, flash and extraction marker.
adapter.js — shared app, health/ammo, controls, objectives and results.
Demo pilot uses normal inputs and rules; snapshots reproduce every action.
Manifest: data/games/dust2/latest.json. Cover and streaming data use
data/records/cover-* and stream-*; state rows identify gameId: dust2.
Dynamic prompts describe the same condition-image group and are sent on image 3.

Validation:
npm run verify:interactive-dust2  collision, delayed AI, flash, ammo and replay
npm run check:interactive-dust2   Chrome controls/UI and full mocked stream
npm run smoke:interactive-dust2   real first image and complete extraction, if idle
Smoke options: --cover-only, --generate-cover, --default-prompts, --seed=<uint>.
The mocked first-image service returns the proxy PNG, not a generated image.

Real validation (2026-10-13): eight kills, successful extraction, one flashbang,
100 remaining HP. Received/decoded 489/489, zero frame drops, RTP loss and NACKs.
Every recorded state replays from inputs; third-frame prompt checks passed.
The browser-reencoded WebM fully decodes. Visual inspection confirms traversal,
gunfire and defeated soldiers. These short-run measurements do not guarantee
long-run network delivery or exact correspondence of AI detail to game geometry.
See qa/stream-verification.json and the recording in latest.json.
