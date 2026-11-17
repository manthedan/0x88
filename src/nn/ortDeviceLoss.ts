import { env } from './ortRuntime.ts';

export interface OrtDeviceLoss {
  reason: string;
  message: string;
}
type OrtDevice = { lost?: Promise<{ reason?: string; message?: string }> };
type DeviceWatch = { listeners: Set<(loss: OrtDeviceLoss) => void>; loss: OrtDeviceLoss | null };
const watches = new WeakMap<OrtDevice, DeviceWatch>();
const currentDevice = () => env.webgpu.device as unknown as OrtDevice | undefined;

/** One promise listener per actual ORT device, with removable owner subscriptions. */
export function watchOrtDeviceLoss(onLost: (loss: OrtDeviceLoss) => void): () => void {
  const device = currentDevice();
  if (!device?.lost) return () => {};
  let watch = watches.get(device);
  if (!watch) {
    watch = { listeners: new Set(), loss: null };
    watches.set(device, watch);
    const entry = watch;
    void device.lost
      .then((info) => {
        entry.loss = { reason: info.reason ?? 'unknown', message: info.message ?? '' };
        const listeners = [...entry.listeners];
        entry.listeners.clear();
        for (const listener of listeners) listener(entry.loss);
      })
      .catch(() => undefined);
  }
  let active = true;
  const notify = (loss: OrtDeviceLoss) => {
    if (!active) return;
    try {
      onLost(loss);
    } catch {
      /* Diagnostics must not break other owners, including late subscribers. */
    }
  };
  if (watch.loss) {
    const loss = watch.loss;
    queueMicrotask(() => notify(loss));
  } else watch.listeners.add(notify);
  return () => {
    active = false;
    watch.listeners.delete(notify);
  };
}

/** A worker/session owner refreshes this after init; same-device re-init does not resubscribe. */
export function createOrtDeviceLossMonitor(onLost: (loss: OrtDeviceLoss) => void): { refresh(): void; stop(): void } {
  let device: OrtDevice | undefined;
  let unsubscribe = () => {};
  const stop = () => {
    unsubscribe();
    device = undefined;
    unsubscribe = () => {};
  };
  return {
    refresh() {
      const next = currentDevice();
      if (next === device) return;
      stop();
      device = next;
      unsubscribe = watchOrtDeviceLoss(onLost);
    },
    stop,
  };
}
