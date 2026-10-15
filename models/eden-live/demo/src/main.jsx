import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  ReactorProvider,
  ReactorView,
  useReactor,
  useReactorMessage,
} from "@reactor-team/js-sdk";
import { Logo } from "@reactor-team/ui";
import { createControls } from "./controls.js";
import "./style.css";

const games = [
  [
    "freeze",
    "Cryo Crossing",
    "Freeze a path across a volcanic cavern.",
    "F · Freeze / Space · Jump",
  ],
  [
    "spiders",
    "Spider Hollow",
    "Explore a torch-lit cave with an axe.",
    "F · Swing / Space · Jump",
  ],
  [
    "tracer",
    "Pulse Skirmish",
    "Dual-pistol combat through a sunlit city.",
    "F · Fire / Shift · Blink / E · Recall / R · Reload",
  ],
  [
    "bioshock",
    "Deepwater Shock",
    "Electricity and a wrench, deep underwater.",
    "F · Wrench / E · Electro bolt",
  ],
  [
    "godzilla",
    "Kaiju Rampage",
    "A giant monster in a miniature city.",
    "F · Breath / Space · Stomp",
  ],
  [
    "arcadeduel",
    "Arcade Duel",
    "Two fighters, one arena.",
    "A/D · Move / F · Combo / G · Guard / E · Uppercut / R · Wave / Shift · Dash",
  ],
  [
    "dust2",
    "Desert Crossfire",
    "Rifle combat through a desert passage.",
    "F · Fire / E · Flashbang / R · Reload",
  ],
];
const keys = [
  "w",
  "a",
  "s",
  "d",
  "arrowup",
  "arrowdown",
  "arrowleft",
  "arrowright",
  "f",
  " ",
  "shift",
  "e",
  "r",
  "g",
];
const labels = {
  arrowup: "↑",
  arrowdown: "↓",
  arrowleft: "←",
  arrowright: "→",
  " ": "Space",
};
const gameKeys = {
  freeze: keys.slice(0, 10),
  spiders: keys.slice(0, 10),
  godzilla: keys.slice(0, 10),
  tracer: keys.slice(0, 13),
  bioshock: [...keys.slice(0, 10), "e"],
  dust2: [...keys.slice(0, 10), "e", "r"],
  arcadeduel: ["a", "d", "f", " ", "g", "e", "r", "shift"],
};

function Workspace() {
  const status = useReactor((s) => s.status),
    connect = useReactor((s) => s.connect),
    disconnect = useReactor((s) => s.disconnect);
  const send = useReactor((s) => s.sendCommand),
    upload = useReactor((s) => s.uploadFile),
    lastError = useReactor((s) => s.lastError);
  const [world, setWorld] = useState(null),
    [prompt, setPrompt] = useState(""),
    [seed, setSeed] = useState(2026092901);
  const [busy, setBusy] = useState(""),
    [error, setError] = useState(""),
    [pressed, setPressed] = useState([]);
  const held = useRef(new Set()),
    commandLock = useRef(false);
  const controlSend = useRef(null);
  controlSend.current = (keys) => request("set_keys", { keys });
  const controller = useRef(null);
  if (!controller.current) {
    controller.current = createControls(
      (keys) => controlSend.current(keys),
      (e) => setError(String(e)),
    );
  }
  const active = status === "ready" && world?.started && !world.paused;
  const selected = games.find((g) => g[0] === world?.game);
  const availableKeys = gameKeys[world?.game] || [];
  useReactorMessage((message) => {
    if (message.type === "state_update") setWorld(message.data);
  });
  useEffect(() => {
    if (status !== "ready") {
      setWorld(null);
      held.current.clear();
      setPressed([]);
    }
  }, [status]);
  async function request(name, args = {}) {
    const answer = await send(name, args);
    if (!answer) throw Error("Command refused. See the server's reason below.");
    setWorld(answer.data);
    return answer.data;
  }
  async function perform(label, fn) {
    if (commandLock.current) return;
    commandLock.current = true;
    setBusy(label);
    setError("");
    try {
      await fn();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy("");
      commandLock.current = false;
    }
  }
  function controls(next) {
    held.current = new Set(next);
    setPressed([...held.current]);
    controller.current.set(next);
  }
  function edge(key, down) {
    if (!active || held.current.has(key) === down) return;
    const next = new Set(held.current);
    down ? next.add(key) : next.delete(key);
    controls(next);
  }
  useEffect(() => {
    if (!active) {
      held.current.clear();
      setPressed([]);
      void controller.current.stop();
      return;
    }
    controller.current.start();
    const onDown = (e) => {
      if (
        /INPUT|TEXTAREA|SELECT/.test(e.target.tagName) ||
        e.ctrlKey ||
        e.metaKey ||
        e.altKey
      )
        return;
      const key = e.key.toLowerCase();
      if (availableKeys.includes(key)) {
        e.preventDefault();
        if (!e.repeat) edge(key, true);
      }
    };
    const onUp = (e) => {
      const key = e.key.toLowerCase();
      if (keys.includes(key)) {
        e.preventDefault();
        edge(key, false);
      }
    };
    const release = () => {
      if (held.current.size) controls([]);
    };
    const hidden = () => {
      if (document.hidden) release();
    };
    window.addEventListener("keydown", onDown);
    window.addEventListener("keyup", onUp);
    window.addEventListener("blur", release);
    document.addEventListener("visibilitychange", hidden);
    return () => {
      window.removeEventListener("keydown", onDown);
      window.removeEventListener("keyup", onUp);
      window.removeEventListener("blur", release);
      document.removeEventListener("visibilitychange", hidden);
      void controller.current.stop();
    };
  }, [active, world?.game]);
  return (
    <div className="app">
      <header>
        <div className="brand">
          <Logo variant="symbol" width={24} height={24} />
          <strong>EdenLive</strong>
          <span>on Reactor</span>
        </div>
        <div className="connection">
          <span className={status === "ready" ? "status live" : "status"}>
            {status}
          </span>
          <button
            disabled={!!busy || !["ready", "disconnected"].includes(status)}
            onClick={() =>
              perform(
                status === "ready" ? "Disconnecting…" : "Connecting…",
                async () => {
                  if (status === "ready") {
                    held.current.clear();
                    setPressed([]);
                    await controller.current.stop();
                    // The server's disconnected hook releases keys. Sending another
                    // command here can accidentally reopen an already-closing session.
                    await disconnect();
                    setWorld(null);
                  } else await connect();
                },
              )
            }
          >
            {status === "ready" ? "Disconnect" : "Connect"}
          </button>
        </div>
      </header>
      <main>
        <aside>
          <span className="eyebrow">01 / CHOOSE YOUR WORLD</span>
          <h1>From code to life.</h1>
          <p className="intro">
            One simulation. Two views.
            <br />
            Steer the draft and watch Eden render it.
          </p>
          <label>
            Official game
            <select
              aria-label="Official game"
              value={world?.game || ""}
              disabled={status !== "ready" || active || !!busy}
              onChange={(e) =>
                perform("Loading game…", async () => {
                  const data = await request("select_game", {
                    game: e.target.value,
                  });
                  setPrompt(data.prompt);
                })
              }
            >
              <option value="" disabled>
                Choose a game…
              </option>
              {games.map((g) => (
                <option key={g[0]} value={g[0]}>
                  {g[1]}
                </option>
              ))}
            </select>
          </label>
          {selected && (
            <p className="hint">
              {selected[2]} Its original cover and prompt are selected together.
            </p>
          )}
          <label className="upload">
            Appearance reference
            <span>{world?.image_name || "Choose a game first"}</span>
            <input
              aria-label="Appearance reference"
              type="file"
              accept="image/png,image/jpeg,image/webp"
              disabled={!selected || active || !!busy}
              onChange={(e) => {
                const file = e.target.files[0];
                if (file)
                  perform("Uploading reference…", async () => {
                    const handle = await upload(file);
                    await request("set_image", { image: handle });
                  });
                e.target.value = "";
              }}
            />
          </label>
          <p className="hint">
            An upload only stages the reference. Match the game's opening
            composition for best results.
          </p>
          <label>
            Scene prompt
            <textarea
              aria-label="Scene prompt"
              rows={6}
              value={prompt}
              maxLength={4096}
              onChange={(e) => setPrompt(e.target.value)}
            />
          </label>
          <button
            className="secondary"
            disabled={status !== "ready" || !prompt.trim() || !!busy}
            onClick={() =>
              perform("Updating prompt…", () =>
                request("set_prompt", { prompt }),
              )
            }
          >
            Update prompt
          </button>
          <p className="hint">
            Applies to the next generated chunk, preserving visual history.
            Game-event descriptions are added automatically.
          </p>
          <label>
            Seed
            <input
              aria-label="Seed"
              type="number"
              min={0}
              max={2147483647}
              value={seed}
              onChange={(e) => setSeed(Number(e.target.value))}
            />
          </label>
          <button
            className="primary"
            disabled={!selected || !prompt.trim() || !!busy || active}
            onClick={() =>
              perform("Starting…", async () => {
                await request("set_prompt", { prompt });
                await request("start", { seed });
              })
            }
          >
            {world?.started ? "Start new world" : "Start"}
          </button>
          {world?.started && !world.ended && (
            <button
              className="secondary"
              disabled={!!busy}
              onClick={() =>
                perform("Updating…", async () => {
                  controls([]);
                  await controller.current.stop();
                  await request("set_paused", { paused: !world.paused });
                })
              }
            >
              {world.paused ? "Resume" : "Pause"}
            </button>
          )}
          <p className="progress" role="status">
            {busy ||
              (world?.ended
                ? "Game ended. Press Start new world to play again."
                : active
                  ? "Generating paired chunks…"
                  : world?.paused && world?.started
                    ? "Paused. Resume keeps the same world."
                    : "Connect, choose a game, then Start.")}
          </p>
          {(error || lastError) && (
            <div className="error" role="alert">
              {error || String(lastError)}
              <button onClick={() => setError("")}>Dismiss</button>
            </div>
          )}
        </aside>
        <section className="workspace">
          <div className="heading">
            <div>
              <span className="eyebrow">02 / ONE TIMELINE</span>
              <h2>Play the same moment.</h2>
            </div>
            <span className="tag">24 FPS · synchronized</span>
          </div>
          <div className="view-labels">
            <span>Native draft</span>
            <span>EdenLive render</span>
          </div>
          <div className="stage">
            <div style={{ display: "flex", position: "absolute", inset: 0 }}>
              <div style={{ flex: 1, minWidth: 0, position: "relative" }}>
                <ReactorView
                  track="draft_video"
                  muted
                  videoObjectFit="contain"
                  className="video"
                />
              </div>
              <div style={{ flex: 1, minWidth: 0, position: "relative" }}>
                <ReactorView
                  track="main_video"
                  muted
                  videoObjectFit="contain"
                  className="video"
                />
              </div>
            </div>
            {!world?.chunks && (
              <div className="placeholder">
                <span className="eyebrow">
                  {active ? "PREPARING FIRST FRAMES" : "READY WHEN YOU ARE"}
                </span>
                <h2>
                  {active
                    ? "Rendering your world…"
                    : "Choose a world. Press Start."}
                </h2>
                <p>Each finished chunk reveals both views together.</p>
              </div>
            )}
          </div>
          <div className="metrics">
            <span>
              World {world?.world_id || "—"} · {world?.chunks || 0} chunks
            </span>
            <span>
              {world?.seconds == null
                ? "Waiting for first chunk"
                : `${world.seconds.toFixed(2)} s / model chunk`}
            </span>
          </div>
          <div className="control-heading">
            <span className="eyebrow">03 / TAKE CONTROL</span>
            <span>
              {pressed.length
                ? pressed.map((k) => labels[k] || k.toUpperCase()).join(" + ")
                : "Controls released"}
            </span>
          </div>
          <div className="controls">
            {availableKeys.map((key) => (
              <button
                key={key}
                disabled={!active}
                className={pressed.includes(key) ? "pressed" : ""}
                onPointerDown={(e) => {
                  e.preventDefault();
                  e.currentTarget.setPointerCapture(e.pointerId);
                  edge(key, true);
                }}
                onPointerUp={() => edge(key, false)}
                onPointerCancel={() => edge(key, false)}
                onLostPointerCapture={() => edge(key, false)}
              >
                {labels[key] || key.toUpperCase()}
              </button>
            ))}
          </div>
          <p className="hint">
            {selected?.[3] || "Choose a game to see its actions."}{" "}
            {world?.game !== "arcadeduel" && "WASD · Move / Arrows · Aim"}
          </p>
          <p className="hint">
            Held keys are polled every 33 ms; only the latest state is retained.
            Hold an action until it is sampled; brief taps are not replayed.
            One draft chunk overlaps inference. Both views stay aligned;
            already captured and buffered frames still
            play. Pause to change the game or image.
          </p>
          <footer>
            <span>Original code games + EdenLive 5B</span>
            <span>Local research preview</span>
          </footer>
        </section>
      </main>
    </div>
  );
}

createRoot(document.querySelector("#root")).render(
  <ReactorProvider
    modelName="eden-live"
    local
    apiUrl={import.meta.env.VITE_REACTOR_URL || "http://localhost:8080"}
    modelTracks={[
      { name: "source_video", kind: "video", direction: "sendonly" },
      { name: "main_video", kind: "video", direction: "recvonly" },
      { name: "draft_video", kind: "video", direction: "recvonly" },
    ]}
    connectOptions={{ autoConnect: false }}
  >
    <Workspace />
  </ReactorProvider>,
);
