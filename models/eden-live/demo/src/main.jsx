import React, { useCallback, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  ReactorProvider,
  ReactorView,
  useReactor,
  useReactorMessage,
} from "@reactor-team/js-sdk";
import { Logo } from "@reactor-team/ui";
import { createControls } from "./controls.js";
import { MODEL_H, MODEL_W, checkFile, prepareReference } from "./reference.js";
import { SEND_RATES, createLocalSource } from "./local-source.js";
import { mountLocalGame } from "./local-game.js";
import officialPrompts from "./official-prompts.json";
import {
  ConnectionError,
  DEFAULT_MODEL,
  TARGETS,
  checkLocalUrl,
  describeError,
  loadKey,
  loadSettings,
  mintToken,
  saveKey,
  saveSettings,
  tokenResolver,
} from "./connection.js";
import "./style.css";

// Relative to the page, so the build works from any static host or sub-path.
const coverUrl = (game) => `${import.meta.env.BASE_URL}covers/${game}.png`;

// Keep a handle on the peer connections the SDK opens, so the page can read the
// browser's own receive/decode/drop counters for the video tracks.
const peers = [];
if (window.RTCPeerConnection) {
  const Native = window.RTCPeerConnection;
  window.RTCPeerConnection = class extends Native {
    constructor(...args) {
      super(...args);
      peers.push(this);
    }
  };
}

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
  shift: "Shift",
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
// Mouse look: pointer speed (px/s) that maps to the fullest turn rate, how
// fast the sent rate follows the pointer, and how often it is sent (30 Hz).
const LOOK_FULL_SCALE = 1920;
const LOOK_SMOOTHING = 0.5;
const LOOK_PERIOD_MS = 33;
// Playing with no new chunk for this long shows the "no new video" warning.
const STALL_AFTER_MS = 2500;
const clampUnit = (v) => Math.max(-1, Math.min(1, v));
const sentences = (text) =>
  text
    .split(/(?<=[.!?])\s+/)
    .map((s) => s.trim())
    .filter(Boolean);
// Steady-state sentences the games repeat every idle chunk; they would bury the events.
const IDLE = /\b(remains|stays) in place\b|\bholds position\b/i;

/** The game's own state from state_update: status badge, health, stalled video. */
function GameStatus({ world, active, stalledFor }) {
  if (!world?.started) return null;
  const state = world.game_state || "";
  const reason = world.outcome?.[1] || world.outcome?.[0] || "";
  let badge = ["playing", "Playing"];
  if (state === "WON") badge = ["won", "Won"];
  else if (state === "LOST") badge = ["lost", "Lost"];
  else if (world.ended)
    badge = ["lost", state ? state[0] + state.slice(1).toLowerCase() : "Ended"];
  else if (world.paused) badge = ["paused", "Paused"];
  const hasHealth = world.health_max != null && world.health != null;
  const share = hasHealth
    ? Math.max(0, Math.min(1, world.health / world.health_max))
    : 0;
  return (
    <div className="game-status">
      <span className={`badge ${badge[0]}`} aria-live="polite" aria-label="Game status">
        {badge[1]}
        {(badge[0] === "lost" || badge[0] === "won") && reason ? ` · ${reason}` : ""}
      </span>
      {hasHealth ? (
        <div className="health" aria-label="Health">
          <div
            className="health-bar"
            role="meter"
            aria-valuemin={0}
            aria-valuemax={world.health_max}
            aria-valuenow={world.health}
          >
            <div
              className="health-fill"
              style={{
                width: `${share * 100}%`,
                background: share > 0.5 ? "#8fbf6a" : share > 0.25 ? "#d6a64a" : "#d0574b",
              }}
            />
          </div>
          <span>
            Health {Math.round(world.health)} / {Math.round(world.health_max)}
          </span>
        </div>
      ) : (
        <span className="health-none">No health in this game</span>
      )}
      {active && stalledFor != null && (
        <span className="stalled" role="alert">
          No new video for {stalledFor} s
        </span>
      )}
    </div>
  );
}

// Per-game action names, read from the game's hint string ("F · Swing / Space · Jump").
const keyNames = { space: " ", shift: "shift" };
function actionsFor(game) {
  const entry = games.find((g) => g[0] === game);
  if (!entry) return {};
  const actions = {};
  if (game !== "arcadeduel") {
    for (const k of ["w", "a", "s", "d"]) actions[k] = "Move";
    for (const k of ["arrowup", "arrowdown", "arrowleft", "arrowright"]) actions[k] = "Aim";
  }
  for (const part of entry[3].split(" / ")) {
    const [names, action] = part.split(" · ");
    for (const name of names.split("/")) {
      const key = keyNames[name.trim().toLowerCase()] || name.trim().toLowerCase();
      actions[key] = action;
    }
  }
  return actions;
}
// Key caps in keyboard positions: [key, grid column start, column span], per row.
const LEFT_ROWS = [
  [["w", 3, 2], ["e", 5, 2], ["r", 7, 2]],
  [["a", 2, 2], ["s", 4, 2], ["d", 6, 2], ["f", 8, 2], ["g", 10, 2]],
  [["shift", 1, 3], [" ", 4, 8]],
];
const ARROW_ROWS = [
  [["arrowup", 3, 2]],
  [["arrowleft", 1, 2], ["arrowdown", 3, 2], ["arrowright", 5, 2]],
];

/** Keyboard and mouse, laid out as on a desk; held inputs light up. */
// noMouse: the game in this page reads keys only.
function ControlMap({ game, active, pressed, mouseHeld, lookOn, onEdge, noMouse }) {
  const used = new Set(gameKeys[game] || []);
  const actions = actionsFor(game);
  const cap = ([key, col, span]) => {
    const isUsed = used.has(key);
    const classes = ["cap"];
    if (!isUsed) classes.push("unused");
    if (pressed.includes(key)) classes.push("pressed");
    return (
      <button
        key={key}
        className={classes.join(" ")}
        style={{ gridColumn: `${col} / span ${span}` }}
        disabled={!active || !isUsed}
        aria-label={`${labels[key] || key.toUpperCase()}${isUsed ? ` · ${actions[key] || ""}` : ""}`}
        onPointerDown={(e) => {
          e.preventDefault();
          e.currentTarget.setPointerCapture(e.pointerId);
          onEdge(key, true);
        }}
        onPointerUp={() => onEdge(key, false)}
        onPointerCancel={() => onEdge(key, false)}
        onLostPointerCapture={() => onEdge(key, false)}
      >
        <b>{labels[key] || key.toUpperCase()}</b>
        <small>{isUsed ? actions[key] || "" : ""}</small>
      </button>
    );
  };
  const cluster = (rows, cols, name) => (
    <div className={`keys ${name}`} style={{ gridTemplateColumns: `repeat(${cols}, 1fr)` }}>
      {rows.map((row, i) => (
        <React.Fragment key={i}>{row.map(cap)}</React.Fragment>
      ))}
    </div>
  );
  const hasMouse = !noMouse && game && game !== "arcadeduel";
  const button = (key, side) => {
    const isUsed = used.has(key);
    const lit = mouseHeld.includes(key);
    return (
      <div className={`mouse-btn ${side}${isUsed ? "" : " unused"}${lit ? " pressed" : ""}`}>
        <b>{side === "left" ? "Left" : "Right"}</b>
        <small>{isUsed ? `${key.toUpperCase()} · ${actions[key] || ""}` : ""}</small>
      </div>
    );
  };
  return (
    <div className="control-map">
      {cluster(LEFT_ROWS, 12, "left-hand")}
      {cluster(ARROW_ROWS, 6, "arrows")}
      <div className={hasMouse ? (lookOn ? "mouse look" : "mouse") : "mouse unused"}>
        <div className="mouse-body">
          {button("f", "left")}
          {button("e", "right")}
          <div className="mouse-palm">{hasMouse ? (lookOn ? "move = look" : "click video") : ""}</div>
        </div>
      </div>
    </div>
  );
}

/** The reference exactly as the model will use it: cover-resized and centre-cropped. */
function ReferencePreview({ source }) {
  const [info, setInfo] = useState(null),
    [failure, setFailure] = useState("");
  const input = useRef(null);
  useEffect(() => {
    let live = true;
    setInfo(null);
    setFailure("");
    prepareReference(source).then(
      (result) => live && setInfo(result),
      (e) => live && setFailure(String(e.message || e)),
    );
    return () => {
      live = false;
    };
  }, [source]);
  useEffect(() => {
    if (!info) return;
    const out = input.current;
    out.width = MODEL_W;
    out.height = MODEL_H;
    out.getContext("2d").putImageData(info.input, 0, 0);
  }, [info]);
  if (failure) return <p className="ref-failure">{failure}</p>;
  return <canvas ref={input} className="ref-preview" aria-label="Reference image as used" />;
}

function Workspace({ conn }) {
  const status = useReactor((s) => s.status),
    connect = useReactor((s) => s.connect),
    disconnect = useReactor((s) => s.disconnect);
  const send = useReactor((s) => s.sendCommand),
    upload = useReactor((s) => s.uploadFile),
    lastError = useReactor((s) => s.lastError);
  const publish = useReactor((s) => s.publish),
    unpublish = useReactor((s) => s.unpublish),
    reactor = useReactor((s) => s.internal.reactor);
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
  // The model and release the platform actually served, read once the session is ready.
  const served =
    status === "ready"
      ? (() => {
          const m = reactor?.getSessionInfo?.()?.model;
          return m?.name ? `${m.name}${m.version ? ` ${m.version}` : ""}` : "";
        })()
      : "";
  const active = status === "ready" && world?.started && !world.paused;
  // Where the draft comes from:
  //   "server"  the server runs the game (source `game`), steered with set_keys;
  //   "browser" the same game runs in this page, and its canvas is sent on source_video;
  //   "video"   a video file or the webcam is sent on source_video.
  // Both sending modes are source `client` on the server, which then plays its output
  // at the rate the frames arrive.
  const [mode, setMode] = useState("server"),
    [sendFps, setSendFps] = useState(SEND_RATES[0]),
    [clipName, setClipName] = useState(""),
    [localGame, setLocalGame] = useState(null),
    [localHud, setLocalHud] = useState(null),
    [published, setPublished] = useState(false),
    [sentFps, setSentFps] = useState(null);
  const local = useRef(null),
    gameFrame = useRef(null),
    sourceSynced = useRef(false),
    statusNow = useRef(status);
  statusNow.current = status;
  if (!local.current) local.current = createLocalSource();
  const sends = mode !== "server";
  const wantSource = sends ? "client" : "game";
  const sending = mode === "browser" ? !!localGame : mode === "video" && !!clipName;
  // Keys steer the server's game, or the game running in this page.
  const playing = active && mode === "server";
  const localPlaying = active && mode === "browser" && !!localGame;
  // The game reached an outcome: the server stops generating, so the last frame
  // stays on screen. An end screen says so until a new world starts.
  const over = !!(world?.started && world.ended);
  const won = world?.game_state === "WON";
  // Start with source client needs a reference (a card's cover or an upload) and frames.
  const canStart =
    !!prompt.trim() &&
    !busy &&
    !active &&
    (sends
      ? world?.source === "client" && !!world?.image_name && published
      : world?.source !== "client" && !!world?.game);
  // A game picked before connecting is held here and loaded once the session is ready.
  const [pendingGame, setPendingGame] = useState("");
  // The user's own reference image. Uploading needs a session, so one chosen
  // before connecting (or kept over a reconnect) is queued and sent after the
  // pending game loads.
  const [localRef, setLocalRef] = useState(null);
  const refQueued = useRef(false);
  const [open, setOpen] = useState(true);
  const shownGame = world?.game || pendingGame;
  const selected = games.find((g) => g[0] === shownGame);
  const availableKeys = gameKeys[world?.game] || [];
  const [feed, setFeed] = useState([]),
    [lastVideoAt, setLastVideoAt] = useState(null),
    [now, setNow] = useState(Date.now()),
    [lookOn, setLookOn] = useState(false),
    [mouseHeld, setMouseHeld] = useState([]);
  const lastChunk = useRef(""),
    feedEnd = useRef(null),
    stageRef = useRef(null),
    look = useRef({ dx: 0, dy: 0, yaw: 0, pitch: 0, sent: "0,0", pending: false });
  useReactorMessage((message) => {
    if (message.type === "state_update") setWorld(message.data);
  });
  // A new world starts an empty event feed and the "no new video" clock.
  useEffect(() => {
    setFeed([]);
    lastChunk.current = "";
    setLastVideoAt(world?.started ? Date.now() : null);
  }, [world?.world_id, world?.started]);
  // Every state_update repeats the latest chunk's events; take them once per chunk.
  useEffect(() => {
    if (!world?.started || !world.chunks) return;
    const key = `${world.world_id}:${world.chunks}`;
    if (key === lastChunk.current) return;
    lastChunk.current = key;
    setLastVideoAt(Date.now());
    const lines = (world.events || []).flatMap(sentences).filter((s) => !IDLE.test(s));
    if (lines.length)
      setFeed((f) =>
        [
          ...f,
          ...lines.map((text, i) => ({ id: `${key}:${i}`, chunk: world.chunks, text })),
        ].slice(-80),
      );
  }, [world?.world_id, world?.chunks]);
  useEffect(() => {
    const list = feedEnd.current;
    if (list) list.scrollTop = list.scrollHeight;
  }, [feed]);
  // Frames the generated video actually presents per second, counted over the last 2 s,
  // so the tag shows the server's playback rate rather than an assumed one.
  const [shownFps, setShownFps] = useState(null),
    [rates, setRates] = useState(null);
  useEffect(() => {
    if (status !== "ready") {
      setShownFps(null);
      setRates(null);
      setSentFps(null);
      return;
    }
    let stopped = false,
      handle = null,
      video = null,
      stamps = [];
    const ticks = [];
    const count = (now, meta) => {
      stamps.push(meta.mediaTime);
      if (!stopped) handle = video.requestVideoFrameCallback(count);
    };
    const attach = () => {
      const videos = stageRef.current?.querySelectorAll("video") || [];
      const next = videos[videos.length - 1];
      if (next && next !== video && next.requestVideoFrameCallback) {
        if (video && handle != null) video.cancelVideoFrameCallback(handle);
        video = next;
        stamps = [];
        handle = video.requestVideoFrameCallback(count);
      }
    };
    const timer = setInterval(() => {
      attach();
      // Distinct media timestamps presented in each 0.5 s tick, over the last 4 ticks.
      ticks.push(new Set(stamps.splice(0)).size);
      if (ticks.length > 4) ticks.shift();
      const frames = ticks.reduce((s, n) => s + n, 0);
      setShownFps(frames ? Math.round(frames / (ticks.length * 0.5)) : null);
      readStats();
    }, 500);
    // The browser's counters for the incoming video, per second over the last 2 s:
    // what arrived, what was decoded, and what the jitter buffer dropped. The SDK
    // hands the <video> a different track id than the receiver's, so every inbound
    // video stream is read and the slowest one shown (both tracks carry the same frames).
    const history = new Map();
    const readStats = async () => {
      const pc = [...peers].reverse().find((p) => p.connectionState !== "closed");
      if (!pc) return;
      const report = await pc.getStats();
      const t = performance.now();
      let worst = null,
        sent = null;
      report.forEach((s) => {
        // What the page sends on source_video: frames encoded and sent per second.
        if (s.type === "outbound-rtp" && s.kind === "video") {
          const h = history.get(s.id) || [];
          h.push({ t, sent: s.framesSent ?? 0 });
          while (h.length > 5) h.shift();
          history.set(s.id, h);
          if (h.length > 1)
            sent = Math.round((h[h.length - 1].sent - h[0].sent) / ((t - h[0].t) / 1000));
          return;
        }
        if (s.type !== "inbound-rtp" || s.kind !== "video") return;
        const h = history.get(s.id) || [];
        h.push({
          t,
          received: s.framesReceived ?? 0,
          decoded: s.framesDecoded ?? 0,
          dropped: s.framesDropped ?? 0,
          freezes: s.freezeCount ?? 0,
        });
        while (h.length > 5) h.shift();
        history.set(s.id, h);
        if (h.length < 2) return;
        const a = h[0],
          b = h[h.length - 1],
          dt = (b.t - a.t) / 1000;
        const per = (k) => Math.round((b[k] - a[k]) / dt);
        const rates = {
          received: per("received"),
          decoded: per("decoded"),
          dropped: per("dropped"),
          freezes: b.freezes,
          decoder: s.decoderImplementation || "",
        };
        if (!worst || rates.decoded < worst.decoded) worst = rates;
      });
      if (worst) setRates(worst);
      setSentFps(sent);
    };
    return () => {
      stopped = true;
      clearInterval(timer);
      if (video && handle != null) video.cancelVideoFrameCallback(handle);
    };
  }, [status]);
  useEffect(() => {
    if (!active) return;
    const timer = setInterval(() => setNow(Date.now()), 500);
    return () => clearInterval(timer);
  }, [active]);
  // Mouse look: click the video to take pointer lock; Esc releases it.
  useEffect(() => {
    const onChange = () =>
      setLookOn(document.pointerLockElement === stageRef.current);
    document.addEventListener("pointerlockchange", onChange);
    return () => document.removeEventListener("pointerlockchange", onChange);
  }, []);
  useEffect(() => {
    if (!playing && document.pointerLockElement === stageRef.current)
      document.exitPointerLock();
  }, [playing]);
  useEffect(() => {
    if (!lookOn) return;
    const state = look.current;
    state.dx = state.dy = state.yaw = state.pitch = 0;
    const sendLook = (yaw, pitch) => {
      const signature = `${yaw},${pitch}`;
      if (signature === state.sent || state.pending) return;
      state.sent = signature;
      state.pending = true;
      request("set_look", { yaw, pitch })
        .catch((e) => setError(String(e)))
        .finally(() => {
          state.pending = false;
        });
    };
    const onMove = (e) => {
      state.dx += e.movementX;
      state.dy += e.movementY;
    };
    let last = performance.now();
    const timer = setInterval(() => {
      const t = performance.now(),
        seconds = Math.max(1, t - last) / 1000;
      last = t;
      // Pointer speed -> target rate; mouse up looks up. Smoothed, and snapped
      // to zero once the pointer has stopped.
      const yawTarget = clampUnit(state.dx / seconds / LOOK_FULL_SCALE),
        pitchTarget = clampUnit(-state.dy / seconds / LOOK_FULL_SCALE);
      state.dx = state.dy = 0;
      state.yaw += (yawTarget - state.yaw) * LOOK_SMOOTHING;
      state.pitch += (pitchTarget - state.pitch) * LOOK_SMOOTHING;
      const round = (v) => (Math.abs(v) < 0.02 ? 0 : Math.round(v * 100) / 100);
      sendLook(round(state.yaw), round(state.pitch));
    }, LOOK_PERIOD_MS);
    // While the pointer is locked: left button holds F (attack), right button holds E.
    const BUTTON_KEYS = { 0: "f", 2: "e" };
    const onButton = (down) => (e) => {
      const key = BUTTON_KEYS[e.button];
      if (!key) return;
      e.preventDefault();
      edge(key, down);
      setMouseHeld((m) => (down ? [...new Set([...m, key])] : m.filter((k) => k !== key)));
    };
    const onMouseDown = onButton(true),
      onMouseUp = onButton(false),
      noMenu = (e) => e.preventDefault();
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mousedown", onMouseDown);
    document.addEventListener("mouseup", onMouseUp);
    document.addEventListener("contextmenu", noMenu);
    return () => {
      clearInterval(timer);
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mousedown", onMouseDown);
      document.removeEventListener("mouseup", onMouseUp);
      document.removeEventListener("contextmenu", noMenu);
      Object.values(BUTTON_KEYS).forEach((key) => edge(key, false));
      setMouseHeld([]);
      state.pending = false;
      sendLook(0, 0);
    };
  }, [lookOn]);
  useEffect(() => {
    if (status !== "ready") sourceSynced.current = false;
    if (status !== "ready" || busy) return;
    // On connect, the server's source follows the page's, before any game is loaded
    // (with source game, select_game would launch a browser the client source never uses).
    if (sends && !world) return;
    if (world && !sourceSynced.current) {
      sourceSynced.current = true;
      if (world.source !== wantSource) {
        perform("Switching source…", () => request("set_source", { source: wantSource }));
        return;
      }
    }
    // Waits for the Connect command to release the lock, or loadGame would be dropped.
    if (pendingGame && !world?.game)
      perform("Loading game…", async () => {
        const data = await request("select_game", { game: pendingGame });
        // Keep a prompt edited before connecting; Start sends it.
        setPrompt((p) =>
          p.trim() && p !== officialPrompts[pendingGame]?.prompt ? p : data.prompt,
        );
        const queued = refQueued.current && localRef;
        refQueued.current = false;
        if (queued) {
          setBusy("Uploading reference…");
          try {
            await sendReference(localRef);
          } catch (e) {
            setLocalRef(null);
            throw e;
          }
        }
      });
  }, [status, busy, pendingGame, world?.game, !!world, wantSource]);
  // The page's frames go out on source_video while connected in a sending mode.
  useEffect(() => {
    if (status !== "ready" || !sending) return;
    let live = true;
    publish("source_video", local.current.track)
      .then(() => {
        if (!live) return;
        setPublished(true);
        // Without a ceiling, Chromium caps a sender above 960x540 at 2.5 Mbps.
        return reactor.setTrackBitrate("source_video", { maxBps: 8_000_000 });
      })
      .catch((e) => live && setError(`Sending your video failed: ${e.message || e}`));
    return () => {
      live = false;
      setPublished(false);
      if (statusNow.current === "ready") void unpublish("source_video");
    };
  }, [status, sending]);
  useEffect(() => local.current.setRate(sendFps), [sendFps]);
  // "browser" mode: the selected game runs in the draft pane, and its canvas is sent.
  useEffect(() => {
    if (mode !== "browser" || !shownGame) return;
    let live = true;
    mountLocalGame(gameFrame.current, shownGame)
      .then((game) => {
        if (!live) return;
        setLocalGame(game);
        return local.current.use({ kind: "canvas", canvas: game.canvas });
      })
      .catch((e) => live && setError(String(e.message || e)));
    return () => {
      live = false;
      local.current.stop();
      setLocalGame(null);
      setLocalHud(null);
    };
  }, [mode, shownGame]);
  // The game in this page runs only while a world plays, and its own state stands in for
  // the server's game status.
  useEffect(() => {
    if (!localGame) return;
    localGame.setPaused(!active);
    if (!active) localGame.release();
    const read = () => setLocalHud(localGame.read());
    read();
    const timer = setInterval(read, 500);
    return () => clearInterval(timer);
  }, [localGame, active]);
  // Each new world starts the game in this page from its opening, as the server's would.
  useEffect(() => {
    if (localGame && world?.started) localGame.reset();
  }, [localGame, world?.world_id]);
  // The keys drive the game in this page; nothing goes to the server.
  useEffect(() => {
    if (!localPlaying) return;
    const usable = gameKeys[shownGame] || [];
    const onKey = (down) => (e) => {
      if (/INPUT|TEXTAREA|SELECT/.test(e.target.tagName) || e.ctrlKey || e.metaKey || e.altKey)
        return;
      const key = e.key.toLowerCase();
      if (!usable.includes(key)) return;
      e.preventDefault();
      if (!e.repeat) localEdge(key, down);
    };
    const onDown = onKey(true),
      onUp = onKey(false);
    const release = () => {
      localGame.release();
      held.current.clear();
      setPressed([]);
    };
    window.addEventListener("keydown", onDown);
    window.addEventListener("keyup", onUp);
    window.addEventListener("blur", release);
    return () => {
      window.removeEventListener("keydown", onDown);
      window.removeEventListener("keyup", onUp);
      window.removeEventListener("blur", release);
      release();
    };
  }, [localPlaying, shownGame]);
  useEffect(() => {
    if (status !== "ready") {
      // Keep the last game picked, so reconnecting loads it again.
      if (world?.game) setPendingGame(world.game);
      if (localRef) refQueued.current = true;
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
  // Switching source is valid at any time, in or out of a world; a world keeps its history.
  function chooseMode(next) {
    if (next === mode || busy) return;
    setMode(next);
    local.current.stop();
    setClipName("");
    if (status === "ready") {
      const source = next === "server" ? "game" : "client";
      if (world?.source !== source)
        perform("Switching source…", () => request("set_source", { source }));
    }
  }
  function chooseClip(input, name) {
    local.current.use(input).then(
      () => setClipName(name),
      (e) => setError(String(e.message || e)),
    );
  }
  function localEdge(key, down) {
    if (!localPlaying || held.current.has(key) === down) return;
    down ? held.current.add(key) : held.current.delete(key);
    setPressed([...held.current]);
    localGame.key(key, down);
  }
  async function sendReference(file) {
    const handle = await upload(file);
    await request("set_image", { image: handle });
  }
  // Picking a game stages its original cover and prompt, replacing any upload.
  function chooseGame(game) {
    setOpen(game !== shownGame || !open);
    if (game === shownGame) return;
    setPendingGame(game);
    setLocalRef(null);
    refQueued.current = false;
    if (status === "ready")
      perform("Loading game…", async () => {
        const data = await request("select_game", { game });
        setPrompt(data.prompt);
      });
    else setPrompt(officialPrompts[game]?.prompt || "");
  }
  function chooseReference(file) {
    const problem = checkFile(file);
    if (problem) return setError(problem);
    // Decode it here too: a file the browser cannot read is refused before upload.
    prepareReference(file).then(
      () => {
        if (status !== "ready") {
          setLocalRef(file);
          refQueued.current = true;
          return;
        }
        perform("Uploading reference…", async () => {
          await sendReference(file);
          setLocalRef(file);
        });
      },
      (e) => setError(String(e.message || e)),
    );
  }
  // Back to the game's cover, keeping the prompt the user has written.
  function restoreCover() {
    if (status !== "ready") {
      setLocalRef(null);
      refQueued.current = false;
      return;
    }
    perform("Restoring cover…", async () => {
      const data = await request("select_game", { game: world.game });
      if (prompt.trim() && prompt !== data.prompt) await request("set_prompt", { prompt });
      setLocalRef(null);
    });
  }
  function startWorld() {
    perform("Starting…", async () => {
      await request("set_prompt", { prompt });
      await request("start", { seed });
    });
  }
  // The end screen: pointer lock off, Enter starts a new world.
  const startRef = useRef(startWorld);
  startRef.current = canStart ? startWorld : null;
  useEffect(() => {
    if (!over) return;
    if (document.pointerLockElement) document.exitPointerLock();
    const onKey = (e) => {
      if (e.key !== "Enter" || e.repeat || /INPUT|TEXTAREA|SELECT|BUTTON/.test(e.target.tagName))
        return;
      e.preventDefault();
      startRef.current?.();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [over]);
  // Reset: the selected game's original cover and prompt, and no world.
  function reset() {
    const game = shownGame;
    if (status !== "ready") {
      setLocalRef(null);
      refQueued.current = false;
      setPrompt(officialPrompts[game]?.prompt || "");
      return;
    }
    perform("Resetting…", async () => {
      if (world?.started && !world.paused) {
        controls([]);
        await controller.current.stop();
        await request("set_paused", { paused: true });
      }
      const data = await request("select_game", { game });
      setPrompt(data.prompt);
      setLocalRef(null);
      setFeed([]);
    });
  }
  async function perform(label, fn) {
    if (commandLock.current) return;
    commandLock.current = true;
    setBusy(label);
    setError("");
    try {
      await fn();
    } catch (e) {
      setError(e instanceof ConnectionError ? e.message : String(e));
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
    if (!playing || held.current.has(key) === down) return;
    const next = new Set(held.current);
    down ? next.add(key) : next.delete(key);
    controls(next);
  }
  useEffect(() => {
    if (!playing) {
      if (mode === "server") {
        held.current.clear();
        setPressed([]);
      }
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
  }, [playing, world?.game]);
  // The selected game's setup: its reference as the model sees it, prompt and seed.
  const uploadLocked = active || !!busy;
  const fromCover = !localRef;
  function setup() {
    return (
      <div className="setup" key="setup" role="region" aria-label={`${selected[1]} setup`}>
        {mode !== "video" && <p className="setup-blurb">{selected[2]}</p>}
        <div className="field-head">
          <span>Reference</span>
          <span className="ref-source">
            {fromCover ? "Official cover" : "Your upload"}
          </span>
        </div>
        <ReferencePreview source={localRef || coverUrl(selected[0])} />
        <div className="ref-actions">
          <label className={uploadLocked ? "file-button disabled" : "file-button"}>
            Replace image…
            <input
              aria-label="Appearance reference"
              type="file"
              accept="image/png,image/jpeg,image/webp"
              disabled={uploadLocked}
              onChange={(e) => {
                const file = e.target.files[0];
                if (file) chooseReference(file);
                e.target.value = "";
              }}
            />
          </label>
          {!fromCover && (
            <button className="link" disabled={uploadLocked} onClick={restoreCover}>
              Use original cover
            </button>
          )}
        </div>
        <div className="field-head">
          <label htmlFor="scene-prompt">Scene prompt</label>
          <button
            className="small"
            disabled={status !== "ready" || !prompt.trim() || !!busy}
            title="Applies to the next generated chunk, preserving visual history"
            onClick={() =>
              perform("Updating prompt…", () => request("set_prompt", { prompt }))
            }
          >
            Update prompt
          </button>
        </div>
        <textarea
          id="scene-prompt"
          className="compact"
          aria-label="Scene prompt"
          rows={3}
          value={prompt}
          maxLength={4096}
          onChange={(e) => setPrompt(e.target.value)}
        />
        <label className="seed">
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
      </div>
    );
  }
  return (
    <div className="app">
      <header>
        <div className="brand">
          <Logo variant="symbol" width={24} height={24} />
          <strong>EdenLive</strong>
          <span>on Reactor</span>
        </div>
        <div className="connection">
          <ConnectionPanel conn={conn} locked={status !== "disconnected" || !!busy} />
          <span className={status === "ready" ? "status live" : "status"} aria-live="polite">
            {status}
            {status === "ready" && served ? ` · ${served}` : ""}
          </span>
          <button
            className="connect"
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
                    conn.onDisconnected();
                  } else if (await conn.onConnect()) await connect();
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
          <div className="source-switch" role="radiogroup" aria-label="Draft source">
            {[
              ["server", "Game on server"],
              ["browser", "Game in your browser"],
              ["video", "Your video"],
            ].map(([id, label]) => (
              <button
                key={id}
                role="radio"
                aria-checked={mode === id}
                className={mode === id ? "on" : ""}
                disabled={!!busy}
                onClick={() => chooseMode(id)}
              >
                {label}
              </button>
            ))}
          </div>
          {sends && (
            <div className="send-panel">
              {mode === "video" && (
                <>
                  <label className={busy ? "file-button disabled" : "file-button"}>
                    Video file…
                    <input
                      aria-label="Video file"
                      type="file"
                      accept="video/*"
                      disabled={!!busy}
                      onChange={(e) => {
                        const file = e.target.files[0];
                        if (file) chooseClip({ kind: "file", file }, file.name);
                        e.target.value = "";
                      }}
                    />
                  </label>
                  <button
                    className="small"
                    disabled={!!busy}
                    onClick={() => chooseClip({ kind: "webcam" }, "Webcam")}
                  >
                    Webcam
                  </button>
                </>
              )}
              <label className="send-rate">
                Send
                <select
                  aria-label="Send rate"
                  value={sendFps}
                  onChange={(e) => setSendFps(Number(e.target.value))}
                >
                  {SEND_RATES.map((r) => (
                    <option key={r} value={r}>
                      {r} fps
                    </option>
                  ))}
                </select>
              </label>
              <span className="send-what">
                {mode === "video"
                  ? clipName || "No video chosen"
                  : localGame
                    ? `${selected?.[1] || ""} in this page`
                    : shownGame
                      ? "Loading the game…"
                      : "Choose a game"}
              </span>
            </div>
          )}
          <div className="actions">
            <button
              className="primary"
              disabled={!canStart}
              onClick={startWorld}
            >
              {world?.started ? "Start new world" : "Start world"}
            </button>
            {world?.started && !world.ended && (
              <button
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
            <button
              disabled={!shownGame || !!busy}
              title="Back to the game's original cover and prompt, with no world running"
              onClick={reset}
            >
              Reset
            </button>
          </div>
          <p className="progress" role="status">
            {busy ||
              (sends && !active && (!shownGame || !sending)
                ? mode === "video"
                  ? `Choose ${shownGame ? "" : "a look below and "}a video file or the webcam, connect, then Start world.`
                  : "Choose a game below; it runs in this page and is sent as the draft."
                : sends && status === "ready" && !active && !published
                  ? "Publishing your frames…"
                  : world?.ended
                ? "Game ended. Start a new world or Reset."
                : active
                  ? "Generating paired chunks…"
                  : world?.paused && world?.started
                    ? "Paused. Resume keeps the same world."
                    : status !== "ready"
                      ? shownGame
                        ? "Connect to load this game, then Start world."
                        : "Choose a game, connect, then Start world."
                      : world?.game
                        ? "Ready. Start world uses the reference and prompt below."
                        : "Choose a game below.")}
          </p>
          {(error || lastError) && (
            <div className="error" role="alert">
              {error || describeError(lastError, conn.live || conn.settings)}
              <button onClick={() => setError("")}>Dismiss</button>
            </div>
          )}
          <span className="eyebrow">
            {mode === "video" ? "01 / CHOOSE A LOOK" : "01 / CHOOSE YOUR WORLD"}
          </span>
          <div className="games" role="list" aria-label="Official games">
            {games.flatMap((g, i) => {
              const isSelected = g[0] === shownGame;
              const card = (
                <button
                  key={g[0]}
                  role="listitem"
                  className={isSelected ? "game-card selected" : "game-card"}
                  aria-pressed={isSelected}
                  aria-expanded={isSelected && open}
                  disabled={active || !!busy}
                  title={g[2]}
                  onClick={() => chooseGame(g[0])}
                >
                  <img src={coverUrl(g[0])} alt="" loading="lazy" />
                  <span>{g[1]}</span>
                </button>
              );
              // The setup opens under the row that holds the selected card.
              const selectedIndex = games.findIndex((x) => x[0] === shownGame);
              const rowEnd = Math.min(games.length - 1, selectedIndex | 1);
              if (!open || selectedIndex < 0 || i !== rowEnd) return [card];
              return [card, setup()];
            })}
          </div>
        </aside>
        <section className="workspace">
          <div className="heading">
            <div>
              <span className="eyebrow">02 / ONE TIMELINE</span>
              <h2>Play the same moment.</h2>
            </div>
            <span className="tag" title="Generated track, frames per second over the last 2 s">
              {sends ? `sending ${sentFps ?? "—"} fps (set ${sendFps}) · ` : ""}
              {rates
                ? `received ${rates.received} · decoded ${rates.decoded} · dropped ${rates.dropped} · `
                : ""}
              {shownFps ? `${shownFps} FPS on screen` : "— FPS"}
              {rates ? ` · freezes ${rates.freezes}${rates.decoder ? ` · ${rates.decoder}` : ""}` : ""}
            </span>
          </div>
          <div className="view-labels">
            <span>
              {mode === "video" ? "Your video" : mode === "browser" ? "Game in your browser" : "Native draft"}
            </span>
            <span>EdenLive render</span>
          </div>
          <div
            className={["stage", lookOn && "look-on", over && (won ? "over won" : "over lost")]
              .filter(Boolean)
              .join(" ")}
            ref={stageRef}
            title={playing ? "Click for mouse look · Esc releases" : undefined}
            onClick={() => {
              if (playing && document.pointerLockElement !== stageRef.current)
                Promise.resolve(stageRef.current.requestPointerLock()).catch(
                  (e) => setError(`Mouse look unavailable: ${e}`),
                );
            }}
            // Right button is E in game; over the video it never opens the browser menu.
            onContextMenu={(e) => e.preventDefault()}
          >
            <div style={{ display: "flex", position: "absolute", inset: 0 }}>
              <div style={{ flex: 1, minWidth: 0, position: "relative" }}>
                <div className="pane" hidden={sends}>
                  <ReactorView
                    track="draft_video"
                    muted
                    videoObjectFit="contain"
                    className="video"
                  />
                </div>
                {/* What this page sends: the game itself, or the frames as sent. */}
                {mode === "browser" && (
                  <iframe ref={gameFrame} className="pane local-game" title="Game in your browser" />
                )}
                {mode === "video" && (
                  <div
                    className="pane local-preview"
                    ref={(el) => el && el.firstChild !== local.current.canvas && el.append(local.current.canvas)}
                  />
                )}
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
              <div className={sends ? "placeholder half" : "placeholder"}>
                <span className="eyebrow">
                  {active ? "PREPARING FIRST FRAMES" : "READY WHEN YOU ARE"}
                </span>
                <h2>
                  {active
                    ? "Rendering your world…"
                    : mode === "video"
                      ? "Choose a look and a video. Press Start."
                      : "Choose a world. Press Start."}
                </h2>
                <p>Each finished chunk reveals both views together.</p>
              </div>
            )}
            {over && (
              <div className="end-screen" key={world.world_id} role="alertdialog" aria-label="Game over">
                <div className="end-card">
                  <span className="eyebrow">{won ? "YOU WON" : "GAME OVER"}</span>
                  <h2>{world.outcome?.[0] || (won ? "You won" : "Game over")}</h2>
                  {(world.outcome || []).slice(1).map((line, i) => (
                    <p key={i} className={i === 0 ? "reason" : "advice"}>
                      {line}
                    </p>
                  ))}
                  <div className="end-actions">
                    <button
                      className="primary"
                      disabled={!canStart}
                      onClick={(e) => {
                        e.stopPropagation();
                        startWorld();
                      }}
                    >
                      Start new world
                    </button>
                    <button
                      disabled={!!busy}
                      onClick={(e) => {
                        e.stopPropagation();
                        reset();
                      }}
                    >
                      Reset
                    </button>
                  </div>
                  <small>Enter · Start new world (seed {seed})</small>
                </div>
              </div>
            )}
            {lookOn && (
              <span className="look-badge">Mouse look on · Esc releases</span>
            )}
          </div>
          {mode !== "video" && (
          <GameStatus
            world={
              mode === "browser"
                ? localHud && {
                    started: true,
                    game_state: localHud.state,
                    outcome: localHud.outcome,
                    health: localHud.health,
                    health_max: localHud.healthMax,
                    paused: !active,
                    ended: localHud.state !== "RUNNING",
                  }
                : world
            }
            active={active}
            stalledFor={
            active && lastVideoAt != null && now - lastVideoAt > STALL_AFTER_MS
              ? Math.floor((now - lastVideoAt) / 1000)
              : null
          } />
          )}
          {mode === "browser" && localHud && localHud.state !== "RUNNING" && (
            <button className="small" onClick={() => localGame?.reset()}>
              Restart the game in your browser
            </button>
          )}
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
          {mode !== "video" && (
          <>
          <div className="control-heading">
            <span className="eyebrow">03 / TAKE CONTROL</span>
            <span>
              {pressed.length
                ? pressed.map((k) => labels[k] || k.toUpperCase()).join(" + ")
                : "Controls released"}
            </span>
          </div>
          <ControlMap
            game={shownGame}
            active={mode === "browser" ? localPlaying : playing}
            pressed={pressed}
            mouseHeld={mouseHeld}
            lookOn={lookOn}
            onEdge={mode === "browser" ? localEdge : edge}
            noMouse={mode === "browser"}
          />
          <p className="hint controls-hint">
            {mode === "browser"
              ? "Keys play the game in this page (no mouse); the server never sees them. "
              : "Hold an action until it is sampled (every 33 ms); taps are not replayed. "}
            {mode === "server" && shownGame && shownGame !== "arcadeduel"
              ? lookOn
                ? "Mouse look on · Esc releases. "
                : "Click the video for mouse look. "
              : ""}
            Pause to change the game or image.
          </p>
          </>
          )}
          {mode === "video" && (
            <p className="hint controls-hint">
              Your video is sent at the chosen rate, and the generated video plays at that rate.
              Keep this tab visible: browsers slow timers in background tabs.
            </p>
          )}
          {mode === "server" && (
          <div className="event-feed" aria-label="Game events">
            <span className="eyebrow">GAME EVENTS</span>
            <ol ref={feedEnd}>
              {feed.map((item) => (
                <li key={item.id}>
                  <span className="chunk">#{item.chunk}</span> {item.text}
                </li>
              ))}
              {!feed.length && (
                <li className="empty">Events from the game appear here as chunks arrive.</li>
              )}
            </ol>
          </div>
          )}
          <footer>
            <span>Original code games + EdenLive 5B</span>
            <span>Local research preview</span>
          </footer>
        </section>
      </main>
    </div>
  );
}

/** Where to connect: DEV / PROD with the visitor's API key, or a model server URL. */
function ConnectionPanel({ conn, locked }) {
  const { settings, update, apiKey, setApiKey, remember, setRemember } = conn;
  const platform = settings.target !== "local";
  return (
    <fieldset className="endpoint" disabled={locked} aria-label="Connection">
      <div className="target-switch" role="radiogroup" aria-label="Platform">
        {Object.entries(TARGETS).map(([id, t]) => (
          <button
            key={id}
            role="radio"
            aria-checked={settings.target === id}
            className={settings.target === id ? "on" : ""}
            onClick={() => update({ target: id })}
            title={t.url ? `Reactor ${t.label} (${t.url.replace("https://", "")})` : "A model server you run (direct, no API key)"}
          >
            {t.label}
          </button>
        ))}
      </div>
      {platform ? (
        <>
          <input
            className="api-key"
            type="password"
            aria-label={`${TARGETS[settings.target].label} API key`}
            placeholder={`${TARGETS[settings.target].label} API key (rk_…)`}
            autoComplete="off"
            spellCheck={false}
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
          />
          <label className="remember" title="Off: the key is forgotten when this tab closes.">
            <input type="checkbox" checked={remember} onChange={(e) => setRemember(e.target.checked)} />
            Remember on this device
          </label>
        </>
      ) : (
        <input
          className="server-url"
          type="url"
          aria-label="Model server URL"
          placeholder="http://localhost:8080"
          spellCheck={false}
          value={settings.localUrl}
          onChange={(e) => update({ localUrl: e.target.value })}
        />
      )}
      <input
        className="model-name"
        aria-label="Model"
        title="Model name. The platform serves its deployed release of it."
        list="model-names"
        spellCheck={false}
        value={settings.model}
        onChange={(e) => update({ model: e.target.value })}
      />
      <datalist id="model-names">
        <option value={DEFAULT_MODEL} />
      </datalist>
    </fieldset>
  );
}

const MODEL_TRACKS = [
  { name: "source_video", kind: "video", direction: "sendonly" },
  { name: "main_video", kind: "video", direction: "recvonly" },
  { name: "draft_video", kind: "video", direction: "recvonly" },
];
const LOCAL_DEFAULT = import.meta.env.VITE_REACTOR_URL || "http://localhost:8080";
const IDLE_CONNECT = { autoConnect: false },
  AUTO_CONNECT = { autoConnect: true };

function App() {
  const [settings, setSettings] = useState(() =>
    loadSettings({ target: import.meta.env.VITE_REACTOR_URL ? "local" : "dev", localUrl: LOCAL_DEFAULT }),
  );
  const [{ key: apiKey, remember }, setStored] = useState(() => loadKey(settings.target));
  // The connection in use: set on Connect, cleared on Disconnect. The provider is rebuilt from
  // it and connects on its own, so endpoint, auth and model always change together.
  const [live, setLive] = useState(null);
  const sessionId = useRef(() => undefined);

  const update = useCallback((patch) => {
    if (patch.target) setStored(loadKey(patch.target));
    setSettings((prev) => {
      const next = { ...prev, ...patch };
      saveSettings(next);
      return next;
    });
  }, []);
  const conn = {
    settings,
    live,
    update,
    apiKey,
    remember,
    setApiKey: (key) => {
      saveKey(settings.target, key, remember);
      setStored({ key, remember });
    },
    setRemember: (on) => {
      saveKey(settings.target, apiKey, on);
      setStored({ key: apiKey, remember: on });
    },
    onDisconnected: () => setLive(null),
    /** Resolves true when the provider already holds this connection and must be told to connect. */
    async onConnect() {
      const model = settings.model.trim() || DEFAULT_MODEL;
      const target = settings.target;
      if (target === "local") {
        const apiUrl = checkLocalUrl(settings.localUrl);
        if (live?.local && live.apiUrl === apiUrl && live.model === model) return true;
        setLive({ target, model, apiUrl, local: true, jwt: undefined });
        return false;
      }
      const key = apiKey.trim();
      const first = await mintToken({ target, apiKey: key, model });
      const jwt = tokenResolver(first, {
        target,
        apiKey: key,
        model,
        sessionId: () => sessionId.current(),
      });
      setLive({ target, model, apiUrl: TARGETS[target].url, local: false, jwt });
      return false;
    },
  };
  return (
    <ReactorProvider
      modelName={live?.model || DEFAULT_MODEL}
      local={live ? live.local : true}
      apiUrl={live ? live.apiUrl : LOCAL_DEFAULT}
      jwtToken={live?.jwt}
      modelTracks={MODEL_TRACKS}
      connectOptions={live ? AUTO_CONNECT : IDLE_CONNECT}
    >
      <SessionIdBridge target={sessionId} />
      <Workspace conn={conn} />
    </ReactorProvider>
  );
}

/** Lets the token refresher read the running session's id without re-rendering App. */
function SessionIdBridge({ target }) {
  const id = useReactor((s) => s.sessionId);
  const now = useRef(id);
  now.current = id;
  target.current = () => now.current;
  return null;
}

createRoot(document.querySelector("#root")).render(<App />);
