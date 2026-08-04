export function el(id: string): HTMLElement {
  const node = document.getElementById(id);
  if (!node) throw new Error(`Missing #${id}`);
  return node;
}

export function maybeEl(id: string): HTMLElement | null {
  return document.getElementById(id);
}

export function inputEl(id: string): HTMLInputElement {
  return el(id) as HTMLInputElement;
}

export function selectEl(id: string): HTMLSelectElement {
  return el(id) as HTMLSelectElement;
}

export function buttonEl(id: string): HTMLButtonElement {
  return el(id) as HTMLButtonElement;
}

export function htmlEscape(value: unknown): string {
  return String(value).replace(/[&<>"']/g, (character) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[character]!);
}
