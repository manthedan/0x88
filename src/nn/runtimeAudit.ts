export type BrowserRuntimeAuditDetail = {
  source: string;
  surface?: string;
  family?: string;
  engineLabel?: string;
  modelId?: string;
  modelUrl?: string;
  metaUrl?: string;
  requestedRuntime?: string;
  resolvedRuntime?: string;
  runtimeConfigId?: string;
  manifestUrl?: string;
  fallbackReason?: string;
  searchBudget?: string;
  notes?: string[];
};

export const BROWSER_RUNTIME_AUDIT_EVENT = 'lc0-browser-runtime-audit';

export function sanitizeBrowserRuntimeAudit(detail: BrowserRuntimeAuditDetail): BrowserRuntimeAuditDetail {
  return Object.fromEntries(
    Object.entries(detail).filter(([, value]) => value !== undefined && value !== ''),
  ) as BrowserRuntimeAuditDetail;
}

export function formatBrowserRuntimeAudit(detail: BrowserRuntimeAuditDetail): string {
  const clean = sanitizeBrowserRuntimeAudit(detail);
  const parts = [
    clean.surface,
    clean.engineLabel ?? clean.family,
    clean.modelId ? `model ${clean.modelId}` : clean.modelUrl,
    clean.requestedRuntime ? `requested ${clean.requestedRuntime}` : undefined,
    clean.resolvedRuntime ? `resolved ${clean.resolvedRuntime}` : undefined,
    clean.runtimeConfigId ? `config ${clean.runtimeConfigId}` : undefined,
    clean.searchBudget,
    clean.manifestUrl ? `manifest ${clean.manifestUrl}` : undefined,
    clean.fallbackReason ? `fallback ${clean.fallbackReason}` : undefined,
  ];
  return parts.filter((part): part is string => !!part).join(' · ') || 'runtime audit unavailable';
}

export function publishBrowserRuntimeAudit(detail: BrowserRuntimeAuditDetail): void {
  const sanitized = sanitizeBrowserRuntimeAudit(detail);
  console.info('[lc0-browser-runtime-audit]', sanitized);
  if (typeof window !== 'undefined') {
    window.dispatchEvent(new CustomEvent(BROWSER_RUNTIME_AUDIT_EVENT, { detail: sanitized }));
  }
}
