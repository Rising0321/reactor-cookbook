// Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.

// Generated from the Echo-WM Flash Reactor schema.
// Model: echo-wm-flash v0.0.0

"use client";

import {
  ReactorProvider,
  ReactorView,
  type ReactorViewProps,
  useReactor,
  useReactorMessage,
} from "@reactor-team/js-sdk";
import type { ReactElement } from "react";

import {
  EchoWmFlashTracks,
  MODEL_NAME,
  type EchoWmFlashAutomaticResetQueuedMessage,
  type EchoWmFlashCameraMotionChangedMessage,
  type EchoWmFlashChunkCompletedMessage,
  type EchoWmFlashImageSelectedMessage,
  type EchoWmFlashMessage,
  type EchoWmFlashOptions,
  type EchoWmFlashPauseChangedMessage,
  type EchoWmFlashPromptQueuedMessage,
  type EchoWmFlashRecvTrackName,
  type EchoWmFlashResetParams,
  type EchoWmFlashRolloutResetQueuedMessage,
  type EchoWmFlashSetCameraMotionParams,
  type EchoWmFlashSetFovParams,
  type EchoWmFlashSetImageParams,
  type EchoWmFlashSetPausedParams,
  type EchoWmFlashSetPromptParams,
  type EchoWmFlashStateUpdateMessage,
  type EchoWmFlashStepQueuedMessage,
} from "./client";

function unwrapMessage<T>(raw: unknown): T {
  const envelope = raw as { type?: string; data?: Record<string, unknown> };
  if (
    envelope &&
    typeof envelope === "object" &&
    envelope.data &&
    typeof envelope.data === "object"
  ) {
    return { ...envelope.data, type: envelope.type } as T;
  }
  return raw as T;
}

export type EchoWmFlashProviderProps = Omit<
  Parameters<typeof ReactorProvider>[0],
  "modelName" | "modelTracks"
>;

export function EchoWmFlashProvider({
  children,
  ...rest
}: EchoWmFlashProviderProps): ReactElement {
  return (
    <ReactorProvider
      {...rest}
      modelName={MODEL_NAME}
      modelTracks={[...EchoWmFlashTracks]}
    >
      {children}
    </ReactorProvider>
  );
}

/** Return the SDK store and Echo-WM's schema-derived commands. */
export function useEchoWmFlash() {
  const connect = useReactor((state) => state.connect);
  const connectOptions = useReactor((state) => state.connectOptions);
  const disconnect = useReactor((state) => state.disconnect);
  const downloadClipAsFile = useReactor((state) => state.downloadClipAsFile);
  const jwtToken = useReactor((state) => state.jwtToken);
  const lastError = useReactor((state) => state.lastError);
  const publish = useReactor((state) => state.publish);
  const reconnect = useReactor((state) => state.reconnect);
  const requestClip = useReactor((state) => state.requestClip);
  const requestRecording = useReactor((state) => state.requestRecording);
  const sendCommand = useReactor((state) => state.sendCommand);
  const sessionExpiration = useReactor((state) => state.sessionExpiration);
  const sessionId = useReactor((state) => state.sessionId);
  const status = useReactor((state) => state.status);
  const tracks = useReactor((state) => state.tracks);
  const unpublish = useReactor((state) => state.unpublish);
  const uploadFile = useReactor((state) => state.uploadFile);

  return {
    connect,
    connectOptions,
    disconnect,
    downloadClipAsFile,
    jwtToken,
    lastError,
    publish,
    reconnect,
    requestClip,
    requestRecording,
    sendCommand,
    sessionExpiration,
    sessionId,
    status,
    tracks,
    unpublish,
    uploadFile,
    setImage: (params: EchoWmFlashSetImageParams): Promise<void> =>
      sendCommand("set_image", params),
    randomImage: (): Promise<void> => sendCommand("random_image", {}),
    setPrompt: (params: EchoWmFlashSetPromptParams): Promise<void> =>
      sendCommand("set_prompt", params),
    setCameraMotion: (
      params: EchoWmFlashSetCameraMotionParams,
    ): Promise<void> => sendCommand("set_camera_motion", params),
    releaseCamera: (): Promise<void> => sendCommand("release_camera", {}),
    setFov: (params: EchoWmFlashSetFovParams): Promise<void> =>
      sendCommand("set_fov", params),
    setPaused: (params: EchoWmFlashSetPausedParams): Promise<void> =>
      sendCommand("set_paused", params),
    step: (): Promise<void> => sendCommand("step", {}),
    reset: (params: EchoWmFlashResetParams): Promise<void> =>
      sendCommand("reset", params),
  };
}

export function useEchoWmFlashMessage(
  handler: (message: EchoWmFlashMessage) => void,
): void {
  useReactorMessage((raw: unknown) => handler(unwrapMessage(raw)));
}

function useMessageType<T extends EchoWmFlashMessage>(
  type: T["type"],
  handler: (message: T) => void,
): void {
  useReactorMessage((raw: unknown) => {
    const message = unwrapMessage<EchoWmFlashMessage>(raw);
    if (message.type === type) handler(message as T);
  });
}

export function useEchoWmFlashStateUpdate(
  handler: (message: EchoWmFlashStateUpdateMessage) => void,
): void {
  useMessageType("state_update", handler);
}

export function useEchoWmFlashImageSelected(
  handler: (message: EchoWmFlashImageSelectedMessage) => void,
): void {
  useMessageType("image_selected", handler);
}

export function useEchoWmFlashPromptQueued(
  handler: (message: EchoWmFlashPromptQueuedMessage) => void,
): void {
  useMessageType("prompt_queued", handler);
}

export function useEchoWmFlashCameraMotionChanged(
  handler: (message: EchoWmFlashCameraMotionChangedMessage) => void,
): void {
  useMessageType("camera_motion_changed", handler);
}

export function useEchoWmFlashPauseChanged(
  handler: (message: EchoWmFlashPauseChangedMessage) => void,
): void {
  useMessageType("pause_changed", handler);
}

export function useEchoWmFlashStepQueued(
  handler: (message: EchoWmFlashStepQueuedMessage) => void,
): void {
  useMessageType("step_queued", handler);
}

export function useEchoWmFlashRolloutResetQueued(
  handler: (message: EchoWmFlashRolloutResetQueuedMessage) => void,
): void {
  useMessageType("rollout_reset_queued", handler);
}

export function useEchoWmFlashChunkCompleted(
  handler: (message: EchoWmFlashChunkCompletedMessage) => void,
): void {
  useMessageType("chunk_completed", handler);
}

export function useEchoWmFlashAutomaticResetQueued(
  handler: (message: EchoWmFlashAutomaticResetQueuedMessage) => void,
): void {
  useMessageType("automatic_reset_queued", handler);
}

export function useEchoWmFlashTrack(name: EchoWmFlashRecvTrackName) {
  return useReactor((state) => state.tracks[name]);
}

export type EchoWmFlashMainVideoViewProps = Omit<ReactorViewProps, "track">;

/** Render Echo-WM video and optionally attach its generated audio track. */
export function EchoWmFlashMainVideoView(
  props: EchoWmFlashMainVideoViewProps,
): ReactElement {
  return <ReactorView {...props} track="main_video" />;
}

export type { EchoWmFlashOptions };
