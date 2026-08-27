import type { EchoWmFlashStateUpdateMessage } from "./echo_wm/client";
import type { useEchoWmFlash } from "./echo_wm/client.react";

/** The typed command surface the generated hook returns. */
export type Model = ReturnType<typeof useEchoWmFlash>;

/** The model's own snapshot of everything a client can observe. */
export type WorldState = EchoWmFlashStateUpdateMessage;
