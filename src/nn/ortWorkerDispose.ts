import { env } from './ortRuntime.ts';

/** Only for terminal teardown of a dedicated ORT worker, never a shared page runtime.
 * Capture the device before release (ORT may drop env.device during release).
 * Release sessions/bindings while the device is valid, then destroy/drop it even on failure.
 */
export async function releaseOrtWorkerResources(release: () => Promise<void>): Promise<void> {
  const webgpu = env.webgpu as unknown as { device?: { destroy?: () => void } };
  const device = await webgpu.device;
  try {
    await release();
  } finally {
    try {
      device?.destroy?.();
    } finally {
      if (webgpu.device === device) delete webgpu.device;
    }
  }
}
