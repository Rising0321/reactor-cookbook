/**
 * host/clock.js — fixed-timestep clock, repo-owned replacement for the
 * per-game TimeUtils.js the adapted games used to vendor.
 *
 * Determinism is the point: the game simulation advances in exact 1/60 s
 * slices no matter how uneven the renderer's frame intervals are. The
 * accumulated wall time (backlog) is capped and the step count per frame
 * is bounded, so a hiccup produces skipped simulation rather than a death
 * spiral of catch-up steps — and a tab hidden by the browser produces no
 * steps at all (the caller passes `active: false`, we drop the backlog).
 *
 * This file is part of the game contract surface (docs/game-contract.md):
 * games receive `dt` strictly from this clock and must not run their own
 * timers for simulation.
 */

export const FIXED_DT = 1 / 60;         // seconds of simulation per step
export const MAX_STEPS = 24;            // most steps consumed in one frame (~0.4s)
export const MAX_BACKLOG_SECONDS = 0.4; // accumulated-but-unstepped time cap

export function createClock() {
  let frameIndex = 0;   // monotonic step counter; total simulation time = frameIndex * FIXED_DT
  let backlog = 0;      // un-stepped wall time, clamped to MAX_BACKLOG_SECONDS
  let steps = 0;        // steps consumed by the most recent advance()

  /**
   * Consume `rawDt` seconds of wall time as fixed steps.
   *
   * stepFn(dt) runs once per simulation step. When `active` is false
   * (hidden tab, paused game) the backlog is discarded instead of stepped,
   * so the sim never fast-forwards on return.
   */
  function advance(rawDt, stepFn, { active = true } = {}) {
    steps = 0;
    if (!active) { backlog = 0; return 0; }
    backlog += Math.min(rawDt, 1);              // ignore absurd gaps (debugger pauses)
    if (backlog > MAX_BACKLOG_SECONDS) {
      backlog = MAX_BACKLOG_SECONDS;           // drop the excess, keep the cap
    }
    while (backlog >= FIXED_DT && steps < MAX_STEPS) {
      backlog -= FIXED_DT;
      frameIndex += 1;
      steps += 1;
      stepFn(FIXED_DT);
    }
    if (steps === MAX_STEPS) backlog = 0;      // still behind: forfeit the remainder
    return steps;
  }

  /** Snapshot for debugging and the __gameTest automation surface. */
  function read() {
    return {
      frameIndex,
      steps,
      backlog,
      simDt: FIXED_DT,
      simTime: frameIndex * FIXED_DT,
    };
  }

  return Object.freeze({ advance, read });
}