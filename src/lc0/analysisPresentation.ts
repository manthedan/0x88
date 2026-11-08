import { parseFen } from '../chess/board.ts';
import { uciLineToSan } from '../chess/san.ts';
import { type AnalysisLine, engineBrushes } from './analysisFormat.ts';
import type { EngineRow } from './engineCatalog.ts';
import { htmlEscape } from './browserDom.ts';
import type { GameNode } from './gameTree.ts';

/** Fresh rows on every mount/reset; saved manual lists and profiles take precedence. */
export function defaultAnalysisEngineRows(): EngineRow[] {
  return [{ family: 'lc0', variant: 'small', strength: 400 }, { family: 'sf', variant: 'lite', strength: 12 }];
}

/**
 * Best moves win collisions, including promotion variants on the same squares.
 * When several engines share a best move it is drawn once; `shared` then lists
 * every backing engine's identity (in engine order) so the board can mark the
 * agreement instead of letting one engine's choice vanish under another's.
 */
export function analysisArrowCandidates(lines: AnalysisLine[], showAll: boolean, focusedEngine: string | null = null): { uci: string; brush: string; shared?: string[] }[] {
  const usable = lines.filter((line) => !line.error && line.pvUci[0] && (line.multipv === 1 || showAll)
    && (!focusedEngine || analysisEngineName(line.engine) === focusedEngine))
    .sort((a, b) => a.multipv - b.multipv);
  const backers = new Map<string, string[]>();
  for (const line of usable) {
    if (line.multipv !== 1) continue;
    const key = line.pvUci[0].slice(0, 4);
    backers.set(key, [...new Set([...(backers.get(key) ?? []), line.engine])]);
  }
  const seen = new Set<string>();
  return usable.flatMap((line) => {
    const uci = line.pvUci[0];
    const key = uci.slice(0, 4);
    if (seen.has(key)) return [];
    seen.add(key);
    const brushes = engineBrushes(line.engine);
    const shared = line.multipv === 1 ? backers.get(key) ?? [] : [];
    return [{ uci, brush: line.multipv === 1 ? brushes.primary : brushes.alt, ...(shared.length > 1 ? { shared } : {}) }];
  }).reverse(); // Alternatives sit underneath best moves; all brushes retain the owner's 15px width.
}

/**
 * Shared-move identity badges inside the destination square, away from the shaft.
 * Chessground's custom layer has a -3.5 -3.5 viewBox (unlike the arrow layer).
 * It already aligns this 0–100 box to the square bounds; do not re-centre it.
 * The move itself remains ONE arrow with the owner's uniform width.
 */
export function sharedArrowMarkerSvg(engines: string[], corner: [number, number] = [1, -1]): string {
  if (!engines.length) return '';
  // Keep the marker in the top corner even with many engine variants. Full ownership
  // remains in the title and agreement strip; a fourth tile summarises overflow.
  const visible: (string | null)[] = engines.length > 4 ? [...engines.slice(0, 3), null] : engines;
  const columns = Math.min(2, visible.length);
  const rows = Math.ceil(visible.length / columns);
  const step = 24;
  const edge = 14; // centre of the outermost badge, measured from the square's edge
  const xs = Array.from({ length: columns }, (_, col) => corner[0] > 0 ? 100 - edge - col * step : edge + col * step);
  const ys = Array.from({ length: rows }, (_, row) => corner[1] > 0 ? 100 - edge - row * step : edge + row * step);
  const badges = visible.map((engine, index) => {
    const x = xs[index % columns];
    const y = ys[Math.floor(index / columns)];
    const color = engine === null ? '#393936' : engineBrushes(engine).swatch;
    const label = engine === null ? `+${engines.length - 3}` : engineMonogram(engine);
    return `<rect x="${x - 12}" y="${y - 10}" width="24" height="20" rx="4" fill="${color}" stroke="#fbf8f1" stroke-width="2"/><text x="${x}" y="${y + 4}" text-anchor="middle" fill="#fff" font-family="system-ui,sans-serif" font-size="12" font-weight="750">${htmlEscape(label)}</text>`;
  }).join('');
  return `<title>Shared move: ${htmlEscape(engines.map(analysisEngineName).join(' + '))}</title>${badges}`;
}

/**
 * Top corner of the destination square on the side away from the incoming shaft (screen space).
 * The chosen top corner is never on the shaft's path, and the arrowhead ends at the square centre,
 * so the badges stay clear of the arrow; top corners are also the freest part of a captured piece.
 */
export function sharedMarkerCorner(uci: string, orientation: 'white' | 'black'): [number, number] {
  const travel = Math.sign(uci.charCodeAt(2) - uci.charCodeAt(0)) * (orientation === 'black' ? -1 : 1);
  return [travel < 0 ? -1 : 1, -1];
}

/** Cycle only through usable bar sources; a pending/failed engine must not trap the control. */
export function nextAnalysisEvalSource(engines: string[], lines: AnalysisLine[], current: string | null): string | null {
  const ready = engines.filter((name) => lines.some((line) => analysisEngineName(line.engine) === name && line.multipv === 1 && !line.error));
  return ready[(ready.indexOf(current ?? '') + 1) % ready.length] ?? null;
}

export interface AnalysisMoveGroup {
  uci: string;
  san: string;
  engines: string[];
}

/** Engines' best moves grouped by move, most-backed first (ties keep engine order). */
export function analysisMoveGroups(bestLines: AnalysisLine[]): AnalysisMoveGroup[] {
  const groups = new Map<string, AnalysisMoveGroup>();
  for (const line of bestLines) {
    const uci = line.pvUci[0];
    if (!uci) continue;
    const group = groups.get(uci) ?? { uci, san: firstAnalysisMove(line), engines: [] };
    group.engines.push(line.engine);
    groups.set(uci, group);
  }
  return [...groups.values()].sort((a, b) => b.engines.length - a.engines.length);
}

/** Spread (pawns) at or above which differing best moves are worth calling a real disagreement. */
export const STRONG_DISAGREEMENT_CP = 100;

/**
 * One-line verdict for the comparison callout: agreement is stated, never implied, and the moves
 * are named. Different top moves with close evals are not a "disagreement"; that word is kept for
 * a large eval spread (`spreadCp`, unsigned centipawns, when every score is finite).
 */
export function analysisAgreementHeadline(groups: AnalysisMoveGroup[], total: number, spreadCp?: number): string {
  const [top, next] = groups;
  if (!top) return 'No best move yet';
  if (groups.length === 1) return total > 2 ? `All ${total} engines agree on ${top.san}` : `Engines agree on ${top.san}`;
  if (top.engines.length >= 2 && next.engines.length < top.engines.length) return `${top.engines.length} of ${total} agree on ${top.san}`;
  const moves = groups.slice(0, 3).map((group) => group.san).join(' / ') + (groups.length > 3 ? ` +${groups.length - 3}` : '');
  if (top.engines.length >= 2) return `Split: ${moves}`;
  return `${spreadCp !== undefined && spreadCp >= STRONG_DISAGREEMENT_CP ? 'Engines disagree' : 'Different top moves'}: ${moves}`;
}

/** Visible units; White-perspective calculation is explained in the callout tooltip and Compare. */
export function analysisSpreadText(spreadCp: number): { text: string; level: 'close' | 'apart' | 'wide' } {
  const pawns = (spreadCp / 100).toFixed(2);
  return { text: `Eval spread: ${pawns} pawns`, level: spreadCp < 50 ? 'close' : spreadCp >= STRONG_DISAGREEMENT_CP ? 'wide' : 'apart' };
}

/** A pasted position is a FEN when it has eight ranks; anything else is treated as PGN. */
export function looksLikeFen(text: string): boolean {
  return /^\s*([pnbrqkPNBRQK1-8]{1,8}\/){7}[pnbrqkPNBRQK1-8]{1,8}(\s|$)/.test(text);
}

/** Short, collision-free engine labels for dense chips ("Lc0", "Stockfish"). */
export function analysisShortEngineNames(engines: string[]): Map<string, string> {
  const short = (engine: string) => analysisEngineName(engine).split(' ')[0];
  const counts = new Map<string, number>();
  for (const engine of new Set(engines)) counts.set(short(engine), (counts.get(short(engine)) ?? 0) + 1);
  return new Map(engines.map((engine) => [engine, (counts.get(short(engine)) ?? 0) > 1 ? analysisEngineName(engine) : short(engine)]));
}

const ENGINE_MONOGRAMS: [RegExp, string][] = [
  [/^lc0/i, 'L0'], [/^(sf|stockfish)\b/i, 'SF'], [/^plentychess/i, 'PC'], [/^centipawn/i, 'CP'], [/^maia/i, 'MA'],
];

/** Two-character uppercase monogram for the engine's colour avatar (one rule for every engine). */
export function engineMonogram(engine: string): string {
  const name = analysisEngineName(engine).trim();
  const known = ENGINE_MONOGRAMS.find(([pattern]) => pattern.test(name));
  if (known) return known[1];
  const letters = name.replace(/[^A-Za-z0-9]/g, '');
  return letters ? letters.slice(0, 2).toUpperCase() : '?';
}

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
  const scrollPositions = new Map([...container.querySelectorAll<HTMLElement>('[data-result-scroll-key]')]
    .map((node) => [node.dataset.resultScrollKey, node.scrollLeft]));
  const expanded = new Set([...container.querySelectorAll<HTMLDetailsElement>('details[open][data-result-key]')].map((node) => node.dataset.resultKey));
  container.innerHTML = markup;
  for (const detail of container.querySelectorAll<HTMLDetailsElement>('details[data-result-key]')) {
    detail.open = expanded.has(detail.dataset.resultKey);
  }
  for (const node of container.querySelectorAll<HTMLElement>('[data-result-scroll-key]')) {
    node.scrollLeft = scrollPositions.get(node.dataset.resultScrollKey) ?? 0;
  }
  if (focusKey) {
    const target = [...container.querySelectorAll<HTMLElement>('[data-focus-key]')].find((node) => node.dataset.focusKey === focusKey);
    target?.focus({ preventScroll: true });
    const scroller = target?.closest<HTMLElement>('[data-result-scroll-key]');
    if (target && scroller) {
      const item = target.getBoundingClientRect();
      const view = scroller.getBoundingClientRect();
      if (item.left < view.left) scroller.scrollLeft += item.left - view.left;
      else if (item.right > view.right) scroller.scrollLeft += item.right - view.right;
    }
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
  return root.children[0] ? branch(root.children[0], true) : '<div class="history-empty"><p><strong>No moves yet.</strong> Play a move on the board, or paste a game or position below.</p><textarea class="quick-pgn" data-quick-pgn rows="3" spellcheck="false" aria-label="Paste a PGN game or a FEN position" placeholder="Paste a PGN game or a FEN position"></textarea><div class="quick-pgn-actions"><button type="button" class="quick-pgn-load" data-quick-pgn-load>Load</button><span class="quick-pgn-hint">PGN or FEN, detected automatically</span></div></div>';
}
