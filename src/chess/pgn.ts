import { type GameNode, GameTree } from '../lc0/gameTree.ts';
import { type BoardState, parseFen, START_FEN } from './board.ts';
import { type Move } from './moveCodec.ts';
import { legalMoves } from './movegen.ts';
import { moveToSan } from './san.ts';

export type PgnWarningReason = 'invalid-move' | 'null-move' | 'after-result';

/** Where tolerant parsing stopped a line. Strict parsing throws instead. */
export interface PgnWarning {
  /** Ply the rejected move would have had (root = 0). */
  ply: number;
  /** Move number as PGN prints it before that move, e.g. `12.` or `12...`. */
  label: string;
  token: string;
  reason: PgnWarningReason;
  /** True when the stopped line is a variation; the parent line carried on. */
  variation: boolean;
}

export interface PgnGame {
  tags: Record<string, string>;
  tree: GameTree;
  result: string;
  /** An explicit valid result, rather than the parser's default for unfinished input. */
  hasResult: boolean;
  /** Lines tolerant parsing stopped early; always empty for strict parsing. */
  warnings: PgnWarning[];
}

/** Strip check/mate/annotation glyphs and normalize castling so SAN compares cleanly. */
function normalizeSan(san: string): string {
  return san
    .replace(/[+#?!]+$/g, '')
    .replace(/e\.p\.$/i, '')
    .replace(/^0-0-0/, 'O-O-O')
    .replace(/^0-0/, 'O-O');
}

/** Resolve a SAN token to a legal move in the given position, or null. */
export function sanToMove(board: BoardState, san: string): Move | null {
  const target = normalizeSan(san);
  for (const move of legalMoves(board)) {
    if (normalizeSan(moveToSan(board, move)) === target) return move;
  }
  return null;
}

/** Split a multi-game PGN string into per-game chunks (tags + movetext). */
export function splitPgnGames(text: string): string[] {
  text = text.replace(/\r\n?/g, '\n');
  const games: string[] = [];
  let start = 0;
  let i = 0;
  let depth = 0;
  let sawMovetext = false;
  let ended = false;
  const finish = (at: number) => {
    const game = text.slice(start, at).trim();
    if (game) games.push(game);
    start = at;
    sawMovetext = false;
    ended = false;
  };
  while (i < text.length) {
    const ch = text[i];
    if (/\s/.test(ch)) {
      i++;
      continue;
    }
    if (ch === '{') {
      const end = text.indexOf('}', i + 1);
      i = end === -1 ? text.length : end + 1;
      continue;
    }
    if (ch === ';' || isEscapeLine(text, i)) {
      const end = text.indexOf('\n', i + 1);
      i = end === -1 ? text.length : end + 1;
      continue;
    }
    // A top-level result terminates a game even when the next game has no tags.
    // Leave trailing comments with the preceding game rather than making empty games.
    if (!depth && (ended || (ch === '[' && sawMovetext))) finish(i);
    if (ch === '[') {
      i++;
      let quoted = false;
      while (i < text.length) {
        const next = text[i++];
        if (next === '\\' && quoted) i++;
        else if (next === '"') quoted = !quoted;
        else if (next === ']' && !quoted) break;
      }
      continue;
    }
    if (ch === '(') {
      depth++;
      i++;
      continue;
    }
    if (ch === ')') {
      depth = Math.max(0, depth - 1);
      i++;
      continue;
    }
    if (ch === '$') {
      i++;
      while (/\d/.test(text[i] ?? '')) i++;
      continue;
    }
    let j = i;
    while (j < text.length && !/[\s(){};$\[]/.test(text[j])) j++;
    if (j === i) {
      i++;
      continue;
    } // Malformed punctuation is rejected by the parser, never a loop.
    const word = stripMoveNumber(text.slice(i, j));
    sawMovetext = true;
    if (!depth && RESULTS.has(word)) ended = true;
    i = j;
  }
  finish(text.length);
  return games;
}

function parseTags(text: string): Record<string, string> {
  const tags: Record<string, string> = {};
  const re = /\[(\w+)\s+"((?:\\.|[^"\\])*)"\]/g;
  let match = re.exec(text);
  while (match) {
    tags[match[1]] = match[2].replace(/\\(["\\])/g, '$1');
    match = re.exec(text);
  }
  return tags;
}

type Token =
  | { type: 'move'; value: string }
  | { type: 'open' }
  | { type: 'close' }
  | { type: 'comment'; value: string }
  | { type: 'nag'; value: string }
  | { type: 'result'; value: string };

const RESULTS = new Set(['1-0', '0-1', '1/2-1/2', '*']);
const NULL_MOVES = new Set(['--', 'Z0', '0000', '@@@@']);
/** Standalone annotation glyphs some tools print as words; mapped to their standard NAGs. */
const GLYPH_NAGS: Record<string, number> = {
  '!': 1,
  '?': 2,
  '!!': 3,
  '??': 4,
  '!?': 5,
  '?!': 6,
  '□': 7,
  '=': 10,
  '∞': 13,
  '+=': 14,
  '+/=': 14,
  '⩲': 14,
  '=+': 15,
  '=/+': 15,
  '⩱': 15,
  '+/-': 16,
  '±': 16,
  '-/+': 17,
  '∓': 17,
  '+-': 18,
  '-+': 19,
};

/** `%` in the first column escapes the rest of the line (PGN standard 6). */
function isEscapeLine(text: string, i: number): boolean {
  return text[i] === '%' && (i === 0 || text[i - 1] === '\n');
}

/** Remove a move number such as `12.`, `12...` or `12…`, possibly fused to the SAN. */
function stripMoveNumber(word: string): string {
  return word.replace(/^\d+(?:\.+|…)/, '');
}

/** Words that carry no move: ellipses, detached check marks, and a detached `e.p.`. */
function isIgnorableWord(word: string): boolean {
  return !word || /^(?:\.+|…|[+#]+|e\.p\.)$/i.test(word);
}

/** Tokenize movetext, handling {comments}, (variations), $NAGs, move numbers, and results. */
function tokenizeMovetext(movetext: string, strict = false): Token[] {
  const tokens: Token[] = [];
  let i = 0;
  const n = movetext.length;
  while (i < n) {
    const ch = movetext[i];
    if (/\s/.test(ch)) {
      i++;
      continue;
    }
    if (ch === '{') {
      const end = movetext.indexOf('}', i);
      if (end === -1 && strict) throw new Error('Unclosed PGN comment');
      const stop = end === -1 ? n : end;
      tokens.push({ type: 'comment', value: movetext.slice(i + 1, stop).trim() });
      i = stop + 1;
      continue;
    }
    if (ch === '}') throw new Error('Unmatched closing PGN comment brace');
    if (ch === '(') {
      tokens.push({ type: 'open' });
      i++;
      continue;
    }
    if (ch === ')') {
      tokens.push({ type: 'close' });
      i++;
      continue;
    }
    if (ch === ';' || isEscapeLine(movetext, i)) {
      const end = movetext.indexOf('\n', i);
      i = end === -1 ? n : end + 1;
      continue;
    }
    if (ch === '$') {
      let j = i + 1;
      while (j < n && /\d/.test(movetext[j])) j++;
      tokens.push({ type: 'nag', value: movetext.slice(i, j) });
      i = j;
      continue;
    }
    // A whitespace/paren/comment-delimited word: move number, SAN, or result.
    let j = i;
    while (j < n && !/[\s(){};$]/.test(movetext[j])) j++;
    let word = movetext.slice(i, j);
    i = j;
    if (RESULTS.has(word)) {
      tokens.push({ type: 'result', value: word });
      continue;
    }
    // Strip a leading move number like "12." or "12..." possibly fused to the SAN.
    word = stripMoveNumber(word);
    if (isIgnorableWord(word)) continue;
    if (RESULTS.has(word)) {
      tokens.push({ type: 'result', value: word });
      continue;
    }
    if (GLYPH_NAGS[word]) {
      tokens.push({ type: 'nag', value: `$${GLYPH_NAGS[word]}` });
      continue;
    }
    tokens.push({ type: 'move', value: word });
  }
  return tokens;
}

/** Parse a single game's PGN text into a variation tree. */
export function parsePgnGame(text: string, options: { strict?: boolean; mainlineOnly?: boolean } = {}): PgnGame {
  text = text.replace(/\r\n?/g, '\n');
  const tags = parseTags(text);
  const startFen = tags.FEN || START_FEN;
  const tree = new GameTree(startFen);
  // Movetext is everything after the last tag line.
  const movetext = text.replace(/\[(\w+)\s+"((?:\\.|[^"\\])*)"\]\s*/g, '');
  const tokens = tokenizeMovetext(movetext, options.strict);

  let last: GameNode = tree.root;
  const stack: GameNode[] = [];
  let result = tags.Result || '*';
  let hasResult = RESULTS.has(tags.Result);
  const warnings: PgnWarning[] = [];
  // Tolerant mode never skips a token and plays on from the wrong position: it stops the
  // current line (keeping what parsed) and resumes after that variation's `)`, if any.
  let stoppedAt = -1; // stack depth of the stopped line; -1 while parsing
  let stoppedNesting = 0;
  let ended = false; // a top-level result was read; a following move is another game
  const stop = (token: string, reason: PgnWarningReason) => {
    const board = parseFen(last.fen);
    const label = `${board.fullmove}${board.turn === 'w' ? '.' : '...'}`;
    warnings.push({ ply: last.ply + 1, label, token, reason, variation: stack.length > 0 });
    stoppedAt = stack.length;
    stoppedNesting = 0;
  };
  for (const token of tokens) {
    if (stoppedAt >= 0) {
      if (token.type === 'open') stoppedNesting++;
      else if (token.type === 'close' && stoppedNesting) stoppedNesting--;
      else if (token.type === 'close' && stoppedAt > 0) {
        stoppedAt = -1;
        last = stack.pop() ?? tree.root;
      }
      continue;
    }
    if (token.type === 'open') {
      stack.push(last);
      if (!options.mainlineOnly) last = last.parent ?? tree.root;
      continue;
    }
    if (token.type === 'close') {
      if (options.strict && !stack.length) throw new Error('Unmatched closing PGN variation');
      // A stray `)` must not restart the line from the root position.
      last = stack.pop() ?? last;
      continue;
    }
    // Aggregation must use the literal mainline, not child insertion order when
    // a RAV shares moves with its parent line. The authoritative PGN retains RAVs.
    if (options.mainlineOnly && stack.length) continue;
    if (token.type === 'result') {
      if (!stack.length) {
        result = token.value;
        hasResult = true;
        ended = true;
      }
      continue;
    }
    if (token.type === 'nag') {
      continue;
    }
    if (token.type === 'comment') {
      if (last !== tree.root) last.comment = last.comment ? `${last.comment} ${token.value}` : token.value;
      continue;
    }
    // move
    if (!options.strict && ended && !stack.length) {
      stop(token.value, 'after-result');
      continue;
    }
    tree.current = last;
    const move = sanToMove(parseFen(last.fen), token.value);
    if (!move) {
      if (options.strict) throw new Error(`Invalid PGN move: ${token.value}`);
      // Interactive single-game loading remains tolerant, but never fabricates a game.
      stop(token.value, NULL_MOVES.has(token.value) ? 'null-move' : 'invalid-move');
      continue;
    }
    last = tree.addMove(move);
  }
  if (options.strict && stack.length) throw new Error('Unclosed PGN variation');
  tree.current = tree.root;
  return { tags, tree, result, hasResult, warnings };
}

const WARNING_TEXT: Record<PgnWarningReason, string> = {
  'invalid-move': 'not a legal move here',
  'null-move': 'null moves are not supported',
  'after-result': 'moves after the game result',
};

/** One-line explanation of where and why a tolerant parse stopped a line. */
export function describePgnWarning(warning: PgnWarning): string {
  return `${warning.variation ? 'A variation stopped' : 'Stopped'} at move ${warning.label}: ${warning.token} (${WARNING_TEXT[warning.reason]})`;
}

/**
 * Interactive loading of pasted text: the first game only, with the number of games
 * found so the caller can say the rest were ignored. Text that yields neither tags nor
 * moves is rejected rather than loaded as an empty game.
 */
export function parseFirstPgnGame(text: string): { game: PgnGame; gameCount: number } {
  const chunks = splitPgnGames(text);
  if (!chunks.length) throw new Error('No PGN found');
  const game = parsePgnGame(chunks[0]);
  if (!game.tree.root.children.length && !Object.keys(game.tags).length) {
    throw new Error(game.warnings.length ? `No PGN moves found. ${describePgnWarning(game.warnings[0])}` : 'No PGN tags or moves found');
  }
  return { game, gameCount: chunks.length };
}

/** Parse every game in a multi-game PGN string. */
export function parsePgnGames(text: string): PgnGame[] {
  return splitPgnGames(text).map((text) => parsePgnGame(text));
}

function serializeNode(node: GameNode, withNumber: boolean): string {
  const board = parseFen(node.parent!.fen);
  const number = board.turn === 'w' ? `${board.fullmove}. ` : withNumber ? `${board.fullmove}... ` : '';
  const comment = node.comment ? ` {${node.comment}}` : '';
  return `${number}${node.san}${comment}`;
}

/** Serialize a variation tree to PGN movetext (mainline with parenthesized variations). */
export function serializeMovetext(tree: GameTree): string {
  const out: string[] = [];
  let node: GameNode | undefined = tree.root.children[0];
  // A move number must be re-printed after a variation or for any black move that starts a token run.
  let forceNumber = true;
  while (node) {
    out.push(serializeNode(node, forceNumber));
    forceNumber = false;
    const variations = node.parent!.children.slice(1);
    for (const variation of variations) {
      out.push('(');
      let cursor: GameNode | undefined = variation;
      let varForce = true;
      while (cursor) {
        out.push(serializeNode(cursor, varForce));
        varForce = false;
        cursor = cursor.children[0];
      }
      out.push(')');
      forceNumber = true; // resume mainline with an explicit move number
    }
    node = node.children[0];
  }
  return out.join(' ').replace(/\(\s/g, '(').replace(/\s\)/g, ')');
}

/** Serialize a full PGN (tags + movetext + result). */
export function gameTreeToPgn(tree: GameTree, tags: Record<string, string> = {}, result = '*'): string {
  const tagPairs = { Event: '?', Site: '?', Date: '????.??.??', Round: '?', White: '?', Black: '?', Result: result, ...tags };
  const header = Object.entries(tagPairs)
    .map(
      ([k, v]) =>
        `[${k} "${v
          .replace(/\\/g, '\\\\')
          .replace(/"/g, '\\"')
          .replace(/[\r\n]+/g, ' ')}"]`,
    )
    .join('\n');
  const movetext = serializeMovetext(tree);
  return `${header}\n\n${movetext} ${result}`.trim();
}
