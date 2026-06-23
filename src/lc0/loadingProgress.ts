export interface LoadingProgressItem {
  id?: string;
  label: string;
  phase?: string;
  loadedBytes?: number;
  totalBytes?: number;
}

function escapeHtml(value: string): string {
  return value.replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]!));
}

export function formatBytes(bytes: number): string {
  if (bytes >= 1_000_000_000) return `${(bytes / 1_000_000_000).toFixed(1)} GB`;
  if (bytes >= 1_000_000) return `${(bytes / 1_000_000).toFixed(0)} MB`;
  if (bytes >= 1_000) return `${(bytes / 1_000).toFixed(0)} KB`;
  return `${Math.max(0, Math.floor(bytes))} B`;
}

export function loadingProgressText(item: LoadingProgressItem): string {
  const prefix = item.phase ? `${item.phase}: ` : '';
  if (item.loadedBytes === undefined) return `${prefix}${item.label}`;
  const loaded = formatBytes(item.loadedBytes);
  if (item.totalBytes && item.totalBytes > 0) return `${prefix}${item.label}: ${loaded} / ${formatBytes(item.totalBytes)}`;
  return `${prefix}${item.label}: ${loaded}`;
}

export function renderLoadingProgress(container: HTMLElement, item: LoadingProgressItem | LoadingProgressItem[]): void {
  const items = Array.isArray(item) ? item : [item];
  container.hidden = items.length === 0;
  container.innerHTML = items.map((entry) => {
    const value = entry.loadedBytes !== undefined ? ` value="${Math.max(0, Math.floor(entry.loadedBytes))}"` : '';
    const max = entry.totalBytes !== undefined && entry.totalBytes > 0 ? ` max="${Math.max(1, Math.floor(entry.totalBytes))}"` : '';
    return `<div class="loading-progress-row" data-progress-id="${escapeHtml(entry.id ?? entry.label)}"><progress${value}${max}></progress><div class="dl-label small">${escapeHtml(loadingProgressText(entry))}</div></div>`;
  }).join('');
}

export function hideLoadingProgress(container: HTMLElement): void {
  container.hidden = true;
  container.innerHTML = '';
}
