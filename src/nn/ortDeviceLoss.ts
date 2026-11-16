import { env } from './ortRuntime.ts';

export interface OrtDeviceLoss {
  reason: string;
  message: string;
}
/** Observe the actual ORT device, not a separate diagnostic device. No extra allocation. */
export function watchOrtDeviceLoss(onLost: (loss: OrtDeviceLoss) => void): () => void {
  let active = true;
  const device = env.webgpu.device as unknown as { lost?: Promise<{ reason?: string; message?: string }> } | undefined;
  if (device?.lost)
    void device.lost
      .then((info) => {
        if (active) onLost({ reason: info.reason ?? 'unknown', message: info.message ?? '' });
      })
      .catch(() => undefined);
  return () => {
    active = false;
  };
}
