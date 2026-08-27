"use client";

import { useState } from "react";

import type { Model, WorldState } from "@/lib/model";

import { Button, Dot, Panel, StatRow } from "./ui";

/**
 * Playback and the life of the current world.
 *
 * Each turn generates one second of synchronized audio-video. The panel exposes
 * rollout progress and deterministic restart controls while playback continues.
 */
export function RolloutPanel({
  model,
  world,
  enabled,
}: {
  model: Model;
  world: WorldState | null;
  enabled: boolean;
}) {
  const [seed, setSeed] = useState("");

  const completed = world?.completed_chunks ?? 0;
  const max = world?.max_chunks ?? 0;
  const progress = max > 0 ? Math.min(100, (completed / max) * 100) : 0;

  return (
    <Panel
      title="Playback"
      hint={
        enabled
          ? "Echo-WM continuously generates native audio-video chunks from the selected scene."
          : "Available once the world has a starting image."
      }
      disabled={!enabled}
    >
      <div className="flex gap-2">
        <input
          value={seed}
          onChange={(event) =>
            setSeed(event.target.value.replace(/[^0-9]/g, ""))
          }
          inputMode="numeric"
          placeholder="seed (optional)"
          className="h-8 min-w-0 flex-1 rounded-md border border-edge bg-panel-raised px-2.5 text-xs tabular-nums text-ink placeholder:text-ink-faint focus:border-ink-faint focus:outline-none"
        />
        <Button
          tone="danger"
          title="Rebuild the world from the same image and prompt"
          onClick={() => {
            void model.reset(seed ? { seed: Number(seed) } : { seed: -1 });
            setSeed("");
          }}
        >
          Restart world
        </Button>
      </div>

      <div className="mt-3 border-t border-edge pt-2">
        <StatRow
          label="Chunks"
          value={max > 0 ? `${completed} / ${max}` : String(completed)}
          title="The world restarts by itself once it reaches the chunk limit."
        />
        <div className="h-1 overflow-hidden rounded-full bg-panel-raised">
          <div
            className="h-full rounded-full bg-ink-faint transition-[width] duration-300"
            style={{ width: `${progress}%` }}
          />
        </div>
        <StatRow label="Next chunk" value={world?.next_chunk ?? "—"} />
        <StatRow label="Seed" value={world?.seed ?? "—"} />
        <StatRow
          label="State"
          value={
            <span className="inline-flex items-center gap-1.5">
              <Dot
                tone={world?.generating ? "live" : "pending"}
                pulse={world?.generating}
              />
              {world?.generating
                ? "generating"
                : world?.reset_queued
                  ? "restarting"
                  : "waiting"}
            </span>
          }
        />
      </div>
    </Panel>
  );
}
