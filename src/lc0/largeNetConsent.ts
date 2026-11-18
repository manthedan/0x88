import { type BigNetConfig, bigNetLoadWarning } from './bt4Engine.ts';
import { isMemoryConstrainedDevice } from './deviceMemoryClass.ts';
import { inspectLargeNetDownload, type LargeNetDownloadState } from './largeNetDownload.ts';
import { PHONE_LARGE_NET_REASON, phoneBigNetBlocked, phoneLargeNetPreference } from './phoneLargeNetPolicy.ts';

export function showPhoneLargeNetFallback(modelOverride = false): void {
  const notice = document.getElementById('phoneLargeNetFallback');
  if (notice) {
    notice.textContent = modelOverride
      ? 'On phones, choose large nets from the engine picker, not a model URL override. Using the default small engine instead.'
      : PHONE_LARGE_NET_REASON;
    notice.hidden = false;
  }
}

let activeConsent: AbortController | undefined;

/**
 * 'cached' approves no download: the model was cached when checked, and storage
 * can evict it before the load. 'download' covers a download.
 */
export type LargeNetConsent = false | 'cached' | 'download';

/** A newer picker choice supersedes an earlier metadata probe or open dialog. */
export async function requestLargeNetConsent(...args: Parameters<typeof requestLargeNetDownloadConsent>): Promise<boolean> {
  return (await requestLargeNetDownloadConsent(...args)) !== false;
}

export async function requestLargeNetDownloadConsent(
  config: BigNetConfig,
  signal?: AbortSignal,
  desktopWarning = true,
  downloadNotice?: string,
  inspect = inspectLargeNetDownload,
): Promise<LargeNetConsent> {
  if (signal?.aborted) return false;
  const request = new AbortController();
  activeConsent?.abort();
  activeConsent = request;
  try {
    return await showLargeNetConsent(config, signal ? AbortSignal.any([signal, request.signal]) : request.signal, desktopWarning, downloadNotice, inspect);
  } finally {
    if (activeConsent === request) activeConsent = undefined;
  }
}

/** Native modal, like Analysis's engine manager: inert background, focus trap and Escape. */
async function showLargeNetConsent(
  config: BigNetConfig,
  signal?: AbortSignal,
  desktopWarning = true,
  downloadNotice?: string,
  inspect = inspectLargeNetDownload,
): Promise<LargeNetConsent> {
  const phone = isMemoryConstrainedDevice();
  if (signal?.aborted) return false;
  if (!phone && !desktopWarning) return 'download';
  const previousFocus = document.activeElement as HTMLElement | null;
  const previousFocusId = previousFocus?.id;
  const previousFocusLabel = previousFocus?.getAttribute('aria-label');
  const cancelled = Promise.withResolvers<null>();
  const abortLookup = () => cancelled.resolve(null);
  signal?.addEventListener('abort', abortLookup, { once: true });
  let state: LargeNetDownloadState | null = null;
  try {
    state = phone ? await Promise.race([inspect(config, signal).catch(() => null), cancelled.promise]) : null;
  } finally {
    signal?.removeEventListener('abort', abortLookup);
  }
  if (signal?.aborted) return false;
  const phoneRisk = phone && config.key !== 'lqo';
  if (phone && state?.cached && !phoneBigNetBlocked(config.key)) return 'cached';
  const name = config.key === 'bt4' ? 'BT4' : config.key === 't3' ? 't3' : 'Queen Odds';
  const bytes = state?.downloadBytes;
  const mb = typeof bytes === 'number' && Number.isFinite(bytes) && bytes > 0 ? Math.max(1, Math.round(bytes / 1_000_000)) : undefined;
  const download =
    downloadNotice ??
    (mb === undefined
      ? 'Download size is unavailable; cached after the first time.'
      : state?.mayDownloadFallback
        ? `${state.downloadBytesIncomplete ? `One model is about ${mb} MB; the other model’s size is unavailable.` : `Model downloads with fallback are estimated at ${mb} MB.`} Retries can use more data.`
        : `${phoneRisk ? `${name} is` : "It's"} a ${mb} MB download, cached after the first time.`);
  const dialog = document.createElement('dialog');
  dialog.className = 'large-net-consent';
  dialog.setAttribute('aria-modal', 'true');
  dialog.setAttribute('aria-labelledby', 'largeNetConsentTitle');
  dialog.setAttribute('aria-describedby', 'largeNetConsentDescription');
  const title = document.createElement('h2');
  title.id = 'largeNetConsentTitle';
  title.textContent = phone ? (phoneRisk ? `Use ${name} on this phone?` : 'Download Queen Odds?') : `Use Lc0 ${config.name}?`;
  const description = document.createElement('p');
  description.id = 'largeNetConsentDescription';
  description.textContent = phone
    ? phoneRisk
      ? `Large nets can crash the tab on phones.${state?.cached ? '' : ` ${download}`}`
      : download
    : bigNetLoadWarning(config);
  const actions = document.createElement('div');
  actions.className = 'large-net-consent-actions';
  const cancel = document.createElement('button');
  cancel.type = 'button';
  cancel.textContent = 'Cancel';
  const use = document.createElement('button');
  use.type = 'button';
  use.textContent = phone && !phoneRisk ? 'Download' : 'Use anyway';
  use.className = 'primary';
  actions.append(cancel, use);
  dialog.append(title, description, actions);
  document.body.append(dialog);
  return new Promise<LargeNetConsent>((resolve) => {
    let accepted = false;
    const abort = () => dialog.close();
    cancel.addEventListener('click', () => dialog.close());
    use.addEventListener('click', () => {
      accepted = true;
      dialog.close();
    });
    // Do not let the underlying board's Escape/arrow shortcuts act on a modal.
    dialog.addEventListener('keydown', (event) => {
      event.stopPropagation();
      if (event.key === 'Tab') {
        event.preventDefault();
        (document.activeElement === cancel ? use : cancel).focus();
      }
    });
    dialog.addEventListener(
      'close',
      () => {
        signal?.removeEventListener('abort', abort);
        dialog.remove();
        // Asset probes can rerender a picker while its consent dialog is open.
        const returnFocus = previousFocus?.isConnected
          ? previousFocus
          : previousFocusId
            ? document.getElementById(previousFocusId)
            : previousFocusLabel
              ? Array.from(document.querySelectorAll<HTMLElement>('[aria-label]')).find((node) => node.getAttribute('aria-label') === previousFocusLabel)
              : null;
        returnFocus?.focus({ preventScroll: true });
        if (accepted && !signal?.aborted && phoneRisk && phoneBigNetBlocked(config.key)) {
          const notice = document.getElementById('phoneLargeNetFallback');
          if (notice) notice.hidden = true;
          phoneLargeNetPreference.setEnabled(true);
        }
        resolve(accepted && !signal?.aborted && (state?.cached ? 'cached' : 'download'));
      },
      { once: true },
    );
    signal?.addEventListener('abort', abort, { once: true });
    dialog.showModal();
    cancel.focus();
  });
}
