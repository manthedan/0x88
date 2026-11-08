import { parseFen, START_FEN } from '../chess/board.ts';
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
  }).reverse(); // Alternatives sit underneath best moves; all brushes share the owner's arrow width.
}

/** Analysis-only chip fills: keep arrow colours intact and white monograms AA-readable. */
export function analysisChipColor(engine: string): string {
  const swatch = engineBrushes(engine).swatch;
  return ({ '#7a4fd6': '#6941b8', '#c45a11': '#a84b0a', '#007a87': '#006975' } as Record<string, string>)[swatch] ?? swatch;
}

export function sharedArrowMarkerText(engines: string[]): string {
  const initials = engines.map(engineMonogram).join(' ');
  return engines.length >= 4 && initials.length * 13.2 + 16 > 100 ? String(engines.length) : initials;
}

/** A single, pointer-transparent pill, in square-local units relative to the arrow's origin. */
export function sharedArrowMarkerSvg(engines: string[], position: [number, number] = [50, 50]): string {
  if (!engines.length) return '';
  const [x, y] = position;
  const text = sharedArrowMarkerText(engines);
  const width = text.length * 13.2 + 16;
  const label = htmlEscape(`${engines.length} engines share this move: ${engines.map(analysisEngineName).join(', ')}`);
  // 22 square-local units are 10.5px on the 382px mobile board, including three-engine pills.
  return `<g class="shared-move-marker" role="img" tabindex="0" aria-label="${label}" style="pointer-events:none"><title>${label}</title><rect x="${x - width / 2}" y="${y - 15}" width="${width}" height="30" rx="8" fill="${analysisChipColor(engines[0])}" stroke="#fbf8f1" stroke-width="2"/><text x="${x}" y="${y + 8}" text-anchor="middle" fill="#fff" font-family="ui-monospace,monospace" font-size="22" font-weight="750">${htmlEscape(text)}</text></g>`;
}

/**
 * Prefer empty space beside the tail/midpoint, never the destination piece or arrowhead.
 * Score the pill's full bounds against occupied piece centres, not just its anchor point.
 * Return origin-square coordinates for Chessground's customSvg center='orig'.
 */
export function sharedMarkerPosition(uci: string, orientation: 'white' | 'black', fen: string, engines: string[]): [number, number] {
  const screen = (file: number, rank: number): [number, number] => orientation === 'white' ? [file + .5, 7.5 - rank] : [7.5 - file, rank + .5];
  const from = screen(uci.charCodeAt(0) - 97, Number(uci[1]) - 1);
  const to = screen(uci.charCodeAt(2) - 97, Number(uci[3]) - 1);
  const dx = to[0] - from[0], dy = to[1] - from[1], length = Math.hypot(dx, dy) || 1;
  const halfWidth = (sharedArrowMarkerText(engines).length * 13.2 + 18) / 200;
  const halfHeight = .16;
  const pieces = parseFen(fen).squares.flatMap((piece, index) => piece ? [screen(index % 8, Math.floor(index / 8))] : []);
  const overlap = (x: number, y: number, piece: [number, number], radius: number) =>
    Math.max(0, Math.min(x + halfWidth, piece[0] + radius) - Math.max(x - halfWidth, piece[0] - radius))
    * Math.max(0, Math.min(y + halfHeight, piece[1] + radius) - Math.max(y - halfHeight, piece[1] - radius));
  const candidates = [.35, .5, .15].flatMap((fraction) => [.55, -.55, .8, -.8, 1.05, -1.05].map((offset) => {
    const x = Math.max(halfWidth + .02, Math.min(8 - halfWidth - .02, from[0] + dx * fraction - dy / length * offset));
    const y = Math.max(halfHeight + .02, Math.min(8 - halfHeight - .02, from[1] + dy * fraction + dx / length * offset));
    const score = overlap(x, y, to, .6) * 1000 + pieces.reduce((sum, piece) => sum + overlap(x, y, piece, .43), 0) * 100
      + Math.abs(offset) + Math.abs(fraction - .35);
    return { x, y, score };
  })).sort((a, b) => a.score - b.score);
  return [50 + (candidates[0].x - from[0]) * 100, 50 + (candidates[0].y - from[1]) * 100];
}

/** Only the untouched standard board needs the first-visit mobile shortcut. */
export function isEmptyAnalysisGame(root: GameNode): boolean {
  return root.fen === START_FEN && root.children.length === 0;
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

/** Verdict first: a tie at the top is a split, never a winner; otherwise say how many engines back the move. */
export function analysisAgreementHeadline(groups: AnalysisMoveGroup[], total: number): string {
  const top = groups[0];
  if (!top) return 'No best move yet';
  if (total <= 1) return `Best move: ${top.san}`;
  const tied = groups.filter((group) => group.engines.length === top.engines.length);
  if (tied.length > 1) return `Engines split: ${tied.map((group) => group.san).join(' / ')}`;
  if (top.engines.length === total) return total === 2 ? `Both engines pick ${top.san}` : `All ${total} engines pick ${top.san}`;
  return `${top.engines.length} of ${total} engines pick ${top.san}`;
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

/** Wrapped moves are clipped as whole buttons; the complete PV remains in Preview. */
export function fitAnalysisPvRows(container: HTMLElement): void {
  for (const pv of container.querySelectorAll<HTMLElement>('.engine-lines .pv')) {
    const top = pv.getBoundingClientRect().top;
    let clipped = false;
    for (const move of pv.querySelectorAll<HTMLButtonElement>('.pv-move')) {
      const outside = move.getBoundingClientRect().top >= top + pv.clientHeight;
      move.inert = outside;
      move.setAttribute('aria-hidden', String(outside));
      clipped ||= outside;
    }
    pv.parentElement?.classList.toggle('pv-clipped', clipped);
  }
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
  fitAnalysisPvRows(container);
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
  return root.children[0] ? branch(root.children[0], true) : '<div class="history-empty"><p><strong>No moves yet.</strong> Play a move on the board, or paste a game or position below.</p><label for="quickPgn">PGN game or FEN position</label><textarea id="quickPgn" class="quick-pgn" data-quick-pgn rows="3" spellcheck="false" placeholder="Paste your game or position"></textarea><div class="quick-pgn-actions"><button type="button" class="quick-pgn-load" data-quick-pgn-load>Load</button><span class="quick-pgn-hint">PGN or FEN, detected automatically</span></div></div>';
}
