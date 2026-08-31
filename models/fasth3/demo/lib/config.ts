export type Mode = "local" | "hosted";

export interface DemoConfig {
  mode: Mode;
  modelName: string;
  apiUrl?: string;
}

const DEFAULT_MODEL_NAME = "fasth3";
const DEFAULT_LOCAL_URL = "http://localhost:8080";

/** Resolve model connection settings without exposing an API key to the browser. */
export function readConfig(): DemoConfig {
  const modelName =
    process.env.REACTOR_MODEL_NAME?.trim() || DEFAULT_MODEL_NAME;

  if (process.env.REACTOR_API_KEY?.trim()) {
    return {
      mode: "hosted",
      modelName,
      apiUrl: process.env.REACTOR_API_URL?.trim() || undefined,
    };
  }

  return {
    mode: "local",
    modelName,
    apiUrl: process.env.REACTOR_LOCAL_URL?.trim() || DEFAULT_LOCAL_URL,
  };
}
