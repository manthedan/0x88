import { htmlEscape } from './browserDom.ts';

/** One labelled diagnostics block: the label once, then one value per line. Escapes label and values. */
export function diagBlockHtml(label: string, valueLines: readonly string[]): string {
  const lines = valueLines.length ? valueLines : ['—'];
  return lines
    .map((line, i) => `<span class="diag-label">${i === 0 ? htmlEscape(label) : ''}</span><span class="diag-value">${htmlEscape(line)}</span>`)
    .join('');
}
