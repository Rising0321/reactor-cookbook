"use client";

import { useCallback, useEffect, useState } from "react";

import {
  type Axis,
  BINDING_BY_CODE,
  type Motion,
  PAD_ROWS,
} from "@/lib/controls";
import type { Model, WorldState } from "@/lib/model";

import { useCameraKeys } from "./useCameraKeys";
import { AxisMeter, Button, cx } from "./ui";

const METERS: {
  axis: Axis;
  label: string;
  negative: string;
  positive: string;
}[] = [
  { axis: "forward", label: "forward", negative: "back", positive: "fwd" },
  { axis: "strafe", label: "strafe", negative: "left", positive: "right" },
  { axis: "pitch", label: "pitch", negative: "down", positive: "up" },
  { axis: "yaw", label: "yaw", negative: "left", positive: "right" },
];

function FovControl({ model, value }: { model: Model; value: number }) {
  const [draft, setDraft] = useState(value);
  useEffect(() => setDraft(value), [value]);
  const commit = () => void model.setFov({ fov_degrees: draft });

  return (
    <label className="block min-w-48 text-[10px] text-ink-faint">
      <span className="flex justify-between">
        <span>field of view</span>
        <span>{draft.toFixed(0)}°</span>
      </span>
      <input
        type="range"
        min={30}
        max={120}
        step={1}
        value={draft}
        onChange={(event) => setDraft(Number(event.target.value))}
        onPointerUp={commit}
        onKeyUp={commit}
        onBlur={commit}
        className="mt-1 w-full accent-[var(--color-accent)]"
      />
    </label>
  );
}

/**
 * Echo-WM's four camera axes, kept under the video where their effect is visible.
 *
 * The meters read from the model's own snapshot rather than from local key
 * state, so they show the velocity the next chunk will actually use — including
 * the zeroes the model applies on its own when playback pauses.
 */
export function CameraBar({
  model,
  world,
  enabled,
}: {
  model: Model;
  world: WorldState | null;
  enabled: boolean;
}) {
  const onMotion = useCallback(
    (motion: Motion) => {
      void model.setCameraMotion(motion);
    },
    [model],
  );

  const { pressed, press, release, releaseAll } = useCameraKeys({
    enabled,
    onMotion,
  });

  return (
    <div
      className={cx(
        "shrink-0 border-t border-edge bg-panel px-4 py-3 transition-opacity",
        !enabled && "pointer-events-none opacity-40",
      )}
    >
      <div className="flex flex-wrap items-start gap-x-8 gap-y-4">
        <div>
          <p className="mb-2 text-[11px] font-semibold uppercase tracking-[0.14em] text-ink-dim">
            Camera
          </p>
          <div className="space-y-1.5">
            {PAD_ROWS.map((row, index) => (
              <div key={index} className="flex gap-1.5">
                {row.map((code) => {
                  const binding = BINDING_BY_CODE[code];
                  const down = pressed.has(code);
                  return (
                    <button
                      key={code}
                      type="button"
                      title={binding.action}
                      onPointerDown={(event) => {
                        event.preventDefault();
                        press(code);
                      }}
                      onPointerUp={() => release(code)}
                      onPointerLeave={() => release(code)}
                      onPointerCancel={() => release(code)}
                      className={cx(
                        "flex w-[4.5rem] flex-col items-center gap-0.5 rounded-md border px-1 py-1 text-[10px] transition-colors select-none",
                        down
                          ? "border-accent bg-accent/15 text-accent"
                          : "border-edge bg-panel-raised text-ink-dim hover:border-ink-faint",
                      )}
                    >
                      <span className="font-semibold">{binding.key}</span>
                      <span className="text-[9px] text-ink-faint">
                        {binding.action}
                      </span>
                    </button>
                  );
                })}
              </div>
            ))}
          </div>
        </div>

        <div className="grid min-w-[18rem] flex-1 grid-cols-1 gap-x-6 gap-y-2 sm:grid-cols-2 xl:grid-cols-4">
          {METERS.map((meter) => (
            <AxisMeter
              key={meter.axis}
              label={meter.label}
              negative={meter.negative}
              positive={meter.positive}
              value={world?.[meter.axis] ?? 0}
            />
          ))}
        </div>

        <div className="flex min-w-48 flex-col gap-2">
          <FovControl model={model} value={world?.fov_degrees ?? 70} />
          <Button
            onClick={() => {
              releaseAll();
              void model.releaseCamera();
            }}
          >
            Neutral camera
          </Button>
        </div>

        <p className="max-w-xs text-[11px] leading-relaxed text-ink-faint">
          {enabled
            ? "Held keys are sent as one four-axis command. Echo-WM samples it when the next 24-frame chunk starts."
            : "Choose a starting image to unlock the controls."}
        </p>
      </div>
    </div>
  );
}
