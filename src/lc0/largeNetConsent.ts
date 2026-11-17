import { type BigNetConfig, bigNetLoadWarning } from './bt4Engine.ts';
import { isMemoryConstrainedDevice } from './deviceMemoryClass.ts';
import { PHONE_LARGE_NET_REASON, phoneLargeNetPreference, phoneLargeNetsBlocked } from './phoneLargeNetPolicy.ts';

export function showPhoneLargeNetFallback(modelOverride = false): void {
  const notice = document.getElementById('phoneLargeNetFallback');
  if (notice) {
    notice.textContent = modelOverride
      ? 'On phones, choose large nets from the engine picker, not a model URL override. Using the default small engine instead.'
      : PHONE_LARGE_NET_REASON;
    notice.hidden = false;
  }
}

/** Native modal, like Analysis's engine manager: inert background, focus trap and Escape. */
export async function requestLargeNetConsent(config: BigNetConfig, signal?: AbortSignal, desktopWarning = true): Promise<boolean> {
  const phone = isMemoryConstrainedDevice();
  if (signal?.aborted) return false;
  if (phone ? !phoneLargeNetsBlocked() : !desktopWarning) return true;
  const previousFocus = document.activeElement as HTMLElement | null;
  const previousFocusId = previousFocus?.id;
  const previousFocusLabel = previousFocus?.getAttribute('aria-label');
  const dialog = document.createElement('dialog');
  dialog.className = 'large-net-consent';
  dialog.setAttribute('aria-modal', 'true');
  dialog.setAttribute('aria-labelledby', 'largeNetConsentTitle');
  dialog.setAttribute('aria-describedby', 'largeNetConsentDescription');
  const title = document.createElement('h2');
  title.id = 'largeNetConsentTitle';
  title.textContent = phone ? 'Use large nets on this phone?' : `Use Lc0 ${config.name}?`;
  const description = document.createElement('p');
  description.id = 'largeNetConsentDescription';
  description.textContent = phone
    ? `Large nets may crash this tab on phones, even when loading just one or switching nets. Without WebGPU they are very slow (around 1–4 visits per second). Lc0 ${config.name} downloads ~${config.approxMb} MB, cached after first load. “Use anyway” enables all large nets on this device. You can turn them off near the engine picker. Arena still allows only one large net at a time.`
    : bigNetLoadWarning(config);
  const actions = document.createElement('div');
  actions.className = 'large-net-consent-actions';
  const cancel = document.createElement('button');
  cancel.type = 'button';
  cancel.textContent = 'Cancel';
  const use = document.createElement('button');
  use.type = 'button';
  use.textContent = 'Use anyway';
  use.className = 'primary';
  actions.append(cancel, use);
  dialog.append(title, description, actions);
  document.body.append(dialog);
  return new Promise<boolean>((resolve) => {
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
        if (accepted && !signal?.aborted && phone) {
          const notice = document.getElementById('phoneLargeNetFallback');
          if (notice) notice.hidden = true;
          phoneLargeNetPreference.setEnabled(true);
        }
        resolve(accepted && !signal?.aborted);
      },
      { once: true },
    );
    signal?.addEventListener('abort', abort, { once: true });
    dialog.showModal();
    cancel.focus();
  });
}
