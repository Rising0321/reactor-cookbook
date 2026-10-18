import * as adapter from './adapter.js';
import {createDemoRuntime} from '../host/demo-runtime.js';

export async function createRuntime(host) {
  const response = await fetch(new URL('./initial-state.json', import.meta.url));
  if (!response.ok) throw Error('Cannot load the game initial state');
  return createDemoRuntime(host, adapter, await response.json());
}
