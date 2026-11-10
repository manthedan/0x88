import { version } from '$app/environment';

/**
 * The worker's script URL carries the build version. A different script URL
 * makes the browser install a new worker on every deploy even though
 * public/sw.js itself rarely changes, and the worker names its cache after the
 * version so that install/activate can drop the previous build's files.
 */
function serviceWorkerUrl(): string {
  return `/sw.js?v=${encodeURIComponent(version)}`;
}

export function registerServiceWorker(): void {
  if (typeof window === 'undefined') return;
  if (!('serviceWorker' in navigator)) return;
  if (!import.meta.env.PROD) {
    navigator.serviceWorker
      .getRegistrations()
      .then((registrations) => Promise.all(registrations.map((registration) => registration.unregister())))
      .catch(() => undefined);
    return;
  }
  if (location.protocol !== 'https:' && location.hostname !== 'localhost' && location.hostname !== '127.0.0.1') return;

  const register = () => {
    navigator.serviceWorker.register(serviceWorkerUrl()).catch(() => undefined);
  };
  // Hydrated onMount callbacks can run after the load event on fast/static
  // pages. Register immediately in that case instead of installing a listener
  // that can never fire.
  if (document.readyState === 'complete') register();
  else window.addEventListener('load', register, { once: true });
}
