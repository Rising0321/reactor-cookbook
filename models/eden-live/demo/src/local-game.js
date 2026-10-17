// One official game run inside this page, for the "Game in your browser" source.
//
// The game's files are served from /local-games/ (the model bundle's games/ directory:
// host/, runtime/, one directory per game, and three's build). They are not part of this
// repo; without them mounting fails with a message saying so. The game runs in a
// same-origin iframe because its modules import "three" and "/host/..." by bare or root
// path, which only an import map in the game's own document can resolve. The page reads
// the game's canvas (rendered with preserveDrawingBuffer, so it can be copied at any time)
// and forwards the player's keys into the game's own input object.

// Resolved against the page, so the build works from any static host or sub-path; absolute,
// because the game's own document is an about:srcdoc iframe.
export const LOCAL_GAMES = new URL(`${import.meta.env.BASE_URL}local-games`, document.baseURI).href;

const gamePage = (slug) => `<!doctype html>
<html><head><meta charset="utf-8">
<script type="importmap">${JSON.stringify({
  imports: {
    three: `${LOCAL_GAMES}/node_modules/three-demo/build/three.module.js`,
    "/host/": `${LOCAL_GAMES}/host/`,
  },
})}</script>
<style>html,body,#app{margin:0;width:100%;height:100%;overflow:hidden;background:#11120f}
canvas{display:block;width:100%;height:100%;object-fit:contain}</style>
</head><body><div id="app"><canvas id="film" width="1280" height="704"></canvas></div>
<script type="module">
window.ready = (async () => {
  const {createInput} = await import("${LOCAL_GAMES}/host/input.js");
  const input = createInput();
  input.attach();
  addEventListener("blur", () => input.clear());
  const {createRuntime} = await import("${LOCAL_GAMES}/${slug}/main.js");
  const runtime = await createRuntime({input, mode: "play", restart: () => runtime.reset()});
  runtime.reset();
  // The model's own frame size, whatever the size of the pane showing it.
  runtime.layout(1280, 704);
  let last = performance.now();
  const frame = (now) => {
    const dt = Math.min(0.05, Math.max(0, (now - last) / 1000));
    last = now;
    runtime.update(dt, input);
    runtime.render(dt);
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
  return {runtime, input};
})();
</script></body></html>`;

// The demo's key names -> KeyboardEvent.code, which is what the games bind.
const CODES = {
  " ": "Space",
  shift: "ShiftLeft",
  arrowup: "ArrowUp",
  arrowdown: "ArrowDown",
  arrowleft: "ArrowLeft",
  arrowright: "ArrowRight",
};
const codeOf = (key) => CODES[key] || `Key${key.toUpperCase()}`;

/** Load game `slug` into `iframe`; resolves once it renders. */
export function mountLocalGame(iframe, slug) {
  return new Promise((resolve, reject) => {
    iframe.onload = async () => {
      const win = iframe.contentWindow;
      // A load of the previous, emptied document; ours sets `ready` before its load event.
      if (!win.ready) return;
      iframe.onload = null;
      try {
        const { runtime, input } = await win.ready;
        const held = new Set();
        const key = (name, down) => {
          if (held.has(name) === down) return;
          down ? held.add(name) : held.delete(name);
          win.dispatchEvent(
            new win.KeyboardEvent(down ? "keydown" : "keyup", { code: codeOf(name), key: name }),
          );
        };
        resolve({
          canvas: runtime.canvas,
          /** Press or release one of the demo's keys (w, a, " ", shift, arrowup, ...). */
          key,
          release() {
            for (const name of [...held]) key(name, false);
            input.clear();
          },
          /** Game state as the game's HUD sees it: state, health, outcome, paused. */
          read: () => runtime.read(),
          setPaused(paused) {
            if (runtime.read().paused !== paused) runtime.handleUiAction("pause");
          },
          reset() {
            runtime.reset();
            held.clear();
          },
          destroy() {
            runtime.destroy?.();
            iframe.srcdoc = "";
          },
        });
      } catch (e) {
        reject(
          Error(
            `The ${slug} game did not load from ${LOCAL_GAMES}/ (copy the bundle's games there): ${e.message || e}`,
          ),
        );
      }
    };
    setTimeout(() => reject(Error(`The ${slug} game did not load within 20 s.`)), 20000);
    iframe.srcdoc = gamePage(slug);
  });
}
