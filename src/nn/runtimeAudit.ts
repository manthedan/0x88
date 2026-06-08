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

export function publishBrowserRuntimeAudit(detail: BrowserRuntimeAuditDetail): void {
  const sanitized = Object.fromEntries(
    Object.entries(detail).filter(([, value]) => value !== undefined && value !== ''),
  ) as BrowserRuntimeAuditDetail;
  console.info('[lc0-browser-runtime-audit]', sanitized);
  if (typeof window !== 'undefined') {
    window.dispatchEvent(new CustomEvent(BROWSER_RUNTIME_AUDIT_EVENT, { detail: sanitized }));
  }
}
