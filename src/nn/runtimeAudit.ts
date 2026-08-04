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
  return Object.fromEntries(Object.entries(detail).filter(([, value]) => value !== undefined && value !== '')) as BrowserRuntimeAuditDetail;
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
export function formatRuntimeFallbackWarning(detail: BrowserRuntimeAuditDetail): string | null {
  if (!detail.fallbackReason) return null;
  const label = detail.engineLabel ?? detail.family ?? 'Engine';
  const from = detail.requestedRuntime ? ` from ${detail.requestedRuntime}` : '';
  const to = detail.resolvedRuntime ? ` to ${detail.resolvedRuntime}` : ' to a fallback runtime';
  return `${label} switched${from}${to}. Performance may be reduced. Reason: ${detail.fallbackReason}`;
}

export function browserRuntimeAuditIdentity(detail: BrowserRuntimeAuditDetail): string {
  return [detail.surface ?? '', detail.family ?? '', detail.engineLabel ?? '', detail.modelId ?? detail.modelUrl ?? '', detail.requestedRuntime ?? ''].join(
    '\u0000',
  );
}

type RuntimeFallbackWarningTarget = Pick<HTMLElement, 'dataset' | 'hidden' | 'textContent'>;

type RuntimeFallbackWarningState = {
  family: string;
  surface: string;
  text: string;
};

const runtimeFallbackWarnings = new WeakMap<RuntimeFallbackWarningTarget, Map<string, RuntimeFallbackWarningState>>();

function hideRuntimeFallbackWarning(target: RuntimeFallbackWarningTarget): void {
  delete target.dataset.runtimeFallbackFamily;
  delete target.dataset.runtimeFallbackIdentity;
  delete target.dataset.runtimeFallbackSurface;
  target.textContent = '';
  target.hidden = true;
}

function renderRuntimeFallbackWarnings(target: RuntimeFallbackWarningTarget): void {
  const warnings = runtimeFallbackWarnings.get(target);
  if (!warnings?.size) {
    runtimeFallbackWarnings.delete(target);
    hideRuntimeFallbackWarning(target);
    return;
  }
  let latestIdentity = '';
  let latest: RuntimeFallbackWarningState | undefined;
  const texts: string[] = [];
  for (const [identity, warning] of warnings) {
    latestIdentity = identity;
    latest = warning;
    texts.push(warning.text);
  }
  target.dataset.runtimeFallbackFamily = latest?.family ?? '';
  target.dataset.runtimeFallbackIdentity = latestIdentity;
  target.dataset.runtimeFallbackSurface = latest?.surface ?? '';
  target.textContent = texts.join(' ');
  target.hidden = false;
}

export function clearRuntimeFallbackWarning(target: RuntimeFallbackWarningTarget): void {
  runtimeFallbackWarnings.delete(target);
  hideRuntimeFallbackWarning(target);
}

export function reconcileRuntimeFallbackWarning(target: RuntimeFallbackWarningTarget, activeIdentities: ReadonlySet<string>): void {
  const warnings = runtimeFallbackWarnings.get(target);
  if (!warnings) {
    const identity = target.dataset.runtimeFallbackIdentity;
    if (identity && !activeIdentities.has(identity)) hideRuntimeFallbackWarning(target);
    return;
  }
  for (const identity of warnings.keys()) {
    if (!activeIdentities.has(identity)) warnings.delete(identity);
  }
  renderRuntimeFallbackWarnings(target);
}

export function updateRuntimeFallbackWarning(target: RuntimeFallbackWarningTarget, detail: BrowserRuntimeAuditDetail): void {
  const identity = browserRuntimeAuditIdentity(detail);
  const warning = formatRuntimeFallbackWarning(detail);
  if (warning) {
    let warnings = runtimeFallbackWarnings.get(target);
    if (!warnings) {
      warnings = new Map();
      runtimeFallbackWarnings.set(target, warnings);
    }
    warnings.delete(identity);
    warnings.set(identity, { family: detail.family ?? '', surface: detail.surface ?? '', text: warning });
    renderRuntimeFallbackWarnings(target);
    return;
  }
  const warnings = runtimeFallbackWarnings.get(target);
  if (!warnings?.delete(identity)) return;
  renderRuntimeFallbackWarnings(target);
}

export function publishBrowserRuntimeAudit(detail: BrowserRuntimeAuditDetail): void {
  const sanitized = sanitizeBrowserRuntimeAudit(detail);
  console.info('[lc0-browser-runtime-audit]', sanitized);
  if (typeof window !== 'undefined') {
    window.dispatchEvent(new CustomEvent(BROWSER_RUNTIME_AUDIT_EVENT, { detail: sanitized }));
  }
}
