/** Button disclosure shared by Copy and the mobile board tools. Native Tab order, no menu trap. */
export function mountAnalysisDisclosure(options: {
  root: HTMLElement;
  trigger: HTMLElement;
  panel: HTMLElement;
  signal: AbortSignal;
  enabled?: () => boolean;
  closeOnAction?: (button: HTMLElement) => boolean;
}) {
  const { root, trigger, panel, signal } = options;
  const enabled = options.enabled ?? (() => true);
  const doc = root.ownerDocument;
  let open = false;
  const close = (restoreFocus = false) => {
    open = false;
    panel.hidden = enabled();
    trigger.setAttribute('aria-expanded', 'false');
    if (restoreFocus) trigger.focus({ preventScroll: true });
  };
  trigger.addEventListener(
    'click',
    () => {
      if (!enabled()) return;
      open = !open;
      panel.hidden = !open;
      trigger.setAttribute('aria-expanded', String(open));
    },
    { signal },
  );
  panel.addEventListener(
    'click',
    (event) => {
      const button = (event.target as HTMLElement).closest<HTMLElement>('button');
      // Load / clipboard fallback may have moved focus outside before this bubbles.
      if (open && button && (options.closeOnAction?.(button) ?? true)) close(root.contains(doc.activeElement));
    },
    { signal },
  );
  root.addEventListener(
    'focusout',
    (event) => {
      if (!root.contains(event.relatedTarget as Node | null)) close();
    },
    { signal },
  );
  doc.addEventListener(
    'pointerdown',
    (event) => {
      if (!root.contains(event.target as Node)) close();
    },
    { signal },
  );
  doc.addEventListener(
    'keydown',
    (event) => {
      // Nested Copy gets first refusal; its Escape must not also close board tools.
      if (event.defaultPrevented || event.key !== 'Escape' || !open) return;
      event.preventDefault();
      event.stopPropagation();
      close(true);
    },
    { capture: true, signal },
  );
  close();
  return { close };
}
