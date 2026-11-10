export function moveFocusTarget(focused: string | null, available: readonly string[]): string | null {
  return focused !== null && available.includes(focused) ? focused : null;
}

/** Capture before rebuilding; restore only focus that already belonged to this scoresheet. */
export function rememberMoveFocus(list: HTMLElement, attribute: 'data-ply' | 'data-idx'): () => void {
  const active = list.ownerDocument.activeElement;
  const focused = active && list.contains(active) ? active.getAttribute(attribute) : null;
  return () => {
    const buttons = [...list.querySelectorAll<HTMLButtonElement>(`button[${attribute}]`)];
    const key = moveFocusTarget(
      focused,
      buttons.map((button) => button.getAttribute(attribute)!),
    );
    if (key !== null) buttons.find((button) => button.getAttribute(attribute) === key)?.focus({ preventScroll: true });
  };
}
