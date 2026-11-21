export function classifyChunkLoadError(error: unknown): { kind: 'offline' | 'chunk'; message: string } | undefined {
  const detail = error instanceof Error ? error.message : String(error);
  if (!/failed to fetch dynamically imported module|error loading dynamically imported module|importing a module script failed|unable to preload css for|loading chunk \S+ failed/i.test(detail)) {
    return undefined;
  }
  return globalThis.navigator?.onLine === false
    ? { kind: 'offline', message: 'You appear to be offline. Reconnect and reload this page to try again.' }
    : { kind: 'chunk', message: 'Part of the app could not load. Reload this page to try again.' };
}

export function showRouteControllerFailure(node: Element | null, error: unknown): void {
  if (!node) return;
  const chunk = classifyChunkLoadError(error);
  node.textContent = chunk?.message ?? `Page failed to initialize: ${error instanceof Error ? error.message : String(error)}`;
  if (!chunk) return;
  const reload = node.ownerDocument.createElement('button');
  reload.type = 'button';
  reload.textContent = 'Reload page';
  reload.onclick = () => location.reload();
  node.append(' ', reload);
}
