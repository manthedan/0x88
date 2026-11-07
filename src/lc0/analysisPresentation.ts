import { parseFen } from '../chess/board.ts';
import { uciLineToSan } from '../chess/san.ts';
import type { AnalysisLine } from './analysisFormat.ts';
import { htmlEscape } from './browserDom.ts';
import type { GameNode } from './gameTree.ts';

/** Display names only: engine/cache identities must retain their original labels. */
export function analysisEngineName(engine: string): string {
  if (engine === 'Lc0') return 'Lc0 Small';
  // The catalog shortens t3's name; worker results/progress retain the model suffix.
  if (engine === 'Lc0 t3-512') return 'Lc0 t3-512 distill';
  return engine.replace(/^SF\b/, 'Stockfish');
}

export function firstAnalysisMove(line: AnalysisLine): string {
  return line.pvSan.split(/\s+/).find(Boolean) ?? line.pvUci[0] ?? '—';
}

/** Number every White move, and the initial Black move, without altering engine SAN. */
export function analysisPvText(fen: string, line: AnalysisLine): string {
  const board = parseFen(fen);
  const sans = uciLineToSan(board, line.pvUci).split(/\s+/).filter(Boolean);
  const startPly = (board.fullmove - 1) * 2 + (board.turn === 'b' ? 1 : 0);
  return sans.map((san, index) => {
    const ply = startPly + index;
    const number = Math.floor(ply / 2) + 1;
    return ply % 2 === 0 ? `${number}. ${san}` : index === 0 ? `${number}… ${san}` : san;
  }).join(' ');
}

export function analysisConsensus(lines: AnalysisLine[]): string {
  if (!lines.length) return 'No analysis yet.';
  const moves = new Map<string, { count: number; san: string }>();
  for (const line of lines) {
    const key = line.pvUci[0];
    if (!key) continue;
    const move = moves.get(key) ?? { count: 0, san: firstAnalysisMove(line) };
    move.count++;
    moves.set(key, move);
  }
  const ranked = [...moves.values()].sort((a, b) => b.count - a.count);
  const best = ranked[0];
  const agreement = !best ? 'No best move yet.' : ranked.length === 1
    ? `Engines agree: ${best.san}`
    : best.count > lines.length / 2
      ? `${best.count}/${lines.length} engines prefer ${best.san}`
      : `Engines disagree: ${ranked.map((move) => move.san).join(' / ')}`;
  // Mate scores cannot be compared with finite centipawn scores.
  const scores = lines.flatMap((line) => line.mateIn === undefined && line.scoreCp !== undefined ? [line.scoreCp] : []);
  const spread = scores.length === lines.length && scores.length >= 2
    ? ` · ${Math.round(Math.max(...scores) - Math.min(...scores))} cp spread` : '';
  return agreement + spread;
}

/** Preserve keyboard position and full-line disclosures while other engines finish. */
export function patchAnalysisResults(container: HTMLElement, markup: string): void {
  if (container.innerHTML === markup) return;
  const active = container.contains(document.activeElement) ? document.activeElement as HTMLElement : null;
  const focusKey = active?.dataset.focusKey;
  const expanded = new Set([...container.querySelectorAll<HTMLDetailsElement>('details[open][data-result-key]')].map((node) => node.dataset.resultKey));
  container.innerHTML = markup;
  for (const detail of container.querySelectorAll<HTMLDetailsElement>('details[data-result-key]')) {
    detail.open = expanded.has(detail.dataset.resultKey);
  }
  if (focusKey) {
    const target = [...container.querySelectorAll<HTMLElement>('[data-focus-key]')].find((node) => node.dataset.focusKey === focusKey);
    target?.focus({ preventScroll: true });
  }
}

export function analysisAvailability(selected: number, ready: number, failed: number, running: boolean): { pending: number; unavailable: number } {
  const missing = Math.max(0, selected - ready);
  const pending = running ? Math.max(0, missing - failed) : 0;
  return { pending, unavailable: missing - pending };
}

/** Paired plies, including FEN games starting on Black's turn and nested branches. */
export function analysisMoveMarkup(root: GameNode, current: GameNode): string {
  const token = (node: GameNode) => {
    const before = parseFen(node.parent!.fen);
    const label = `${before.fullmove}${before.turn === 'w' ? '.' : '...'} ${node.san ?? ''}`;
    return `<button type="button" class="mv${node === current ? ' current' : ''}" data-node="${node.id}" aria-label="${htmlEscape(label)}"${node === current ? ' aria-current="step"' : ''}>${htmlEscape(node.san ?? '')}</button>`;
  };
  const branch = (first: GameNode, includeSiblings: boolean): string => {
    let html = '';
    let pair: GameNode[] = [];
    const flush = () => {
      if (!pair.length) return;
      const before = parseFen(pair[0].parent!.fen);
      html += `<div class="move-pair"><span class="move-number">${before.fullmove}.</span>${before.turn === 'b' ? '<span class="missing-ply">…</span>' : ''}${pair.map(token).join('')}</div>`;
      pair = [];
    };
    let node: GameNode | undefined = first;
    while (node) {
      const before = parseFen(node.parent!.fen);
      if (before.turn === 'w') flush();
      pair.push(node);
      const siblings = includeSiblings || node !== first ? node.parent!.children.slice(1) : [];
      if (siblings.length) {
        flush();
        html += siblings.map((variation) => `<div class="variation"><span class="variation-label">Variation</span>${branch(variation, false)}</div>`).join('');
      } else if (before.turn === 'b') flush();
      node = node.children[0];
    }
    flush();
    return html;
  };
  return root.children[0] ? branch(root.children[0], true) : '<p class="history-empty">No moves yet. Play on the board or load a PGN.</p>';
}
