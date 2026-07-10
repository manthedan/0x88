export function registerServiceWorker(): void {
  if (typeof window === 'undefined') return;
  if (!('serviceWorker' in navigator)) return;
  if (!import.meta.env.PROD) {
    navigator.serviceWorker.getRegistrations()
      .then((registrations) => Promise.all(registrations.map((registration) => registration.unregister())))
      .catch(() => undefined);
    return;
  }
  if (location.protocol !== 'https:' && location.hostname !== 'localhost' && location.hostname !== '127.0.0.1') return;

  const register = () => {
    navigator.serviceWorker.register('/sw.js').catch(() => undefined);
  };
  // Hydrated onMount callbacks can run after the load event on fast/static
  // pages. Register immediately in that case instead of installing a listener
  // that can never fire.
  if (document.readyState === 'complete') register();
  else window.addEventListener('load', register, { once: true });
}
