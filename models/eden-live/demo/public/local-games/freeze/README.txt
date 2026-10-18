Freeze the crossing

Start: npm start
Open: http://127.0.0.1:4174/game.html?game=freeze
Local: add &play=proxy

WASD moves; arrows aim; hold F to freeze stepping stones; Space jumps.
Reach the far platform before temporary ice melts. Demo pilot follows the same physics.

game.js: deterministic simulation, snapshots, prompts and demo input.
scene.js / lava-world.js: Three.js world and separate proxy HUD.
adapter.js: runtime integration, matching cover validation and ending statistics.
Shared transport, recording, views and instructions: ../../../README.md.

Tests: npm run verify:interactive-freeze; npm run check:interactive-freeze-ui; npm run check:stream.
