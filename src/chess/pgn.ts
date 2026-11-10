import { type GameNode, GameTree } from '../lc0/gameTree.ts';
import { type BoardState, type PieceRole, parseFen, START_FEN, squareName } from './board.ts';
import { type Move } from './moveCodec.ts';
import { legalMoves } from './movegen.ts';

export type PgnWarningReason =
  | 'invalid-move'
  | 'ambiguous-move'
  | 'null-move'
  | 'after-result'
  | 'tag-pair'
  | 'unclosed-command'
  | 'unmatched-comment'
  | 'variation-depth';

// Parsing is iterative. Bound hostile external nesting while allowing every shape
// within the workspace's 5000-move budget, including branches merged during import.
const MAX_VARIATION_DEPTH = 5000;

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

const SAN_ROLES: Record<string, PieceRole> = { P: 'p', N: 'n', B: 'b', R: 'r', Q: 'q', K: 'k' };
// [piece][from file][from rank][x or : or -][target][=promotion]. A lower-case letter is
// always a file, so `bxc3` is a pawn capture and `Bxc3` a bishop capture.
const SAN_PATTERN = /^([PNBRQK])?([a-h])?([1-8])?([x:-])?([a-h][1-8])(?:=?([NBRQnbrq]))?$/;

/**
 * Resolve a move token by parsing it and filtering the legal moves, rather than by
 * printing SAN for every legal move. Accepts canonical SAN and the lenient forms that
 * still name exactly one legal move: `0-0`, `a8Q`, over- or unnecessarily disambiguated
 * `Ng1f3`/`Nbc6`, and long algebraic or UCI `e2e4`/`e2-e4`/`e7e8q`. Trailing `+ # ! ?`
 * and `e.p.` are ignored. A capture mark must match a capture; a missing promotion
 * piece is ambiguous.
 */
function resolveSan(board: BoardState, token: string): { move: Move } | { error: 'ambiguous' | 'invalid' } {
  const san = token
    .replace(/[+#?!]+$/, '')
    .replace(/e\.p\.$/i, '')
    .replace(/[+#]+$/, '');
  const castle = /^(?:O-O|0-0|o-o)(-O|-0|-o)?$/.exec(san);
  const parsed = castle ? null : SAN_PATTERN.exec(san);
  if (!castle && !parsed) return { error: 'invalid' };
  const candidates = legalMoves(board).filter((move) => {
    const role = board.squares[move.from]?.[1];
    if (castle) return role === 'k' && Math.abs(move.to - move.from) === 2 && move.to % 8 === (castle[1] ? 2 : 6);
    const [, piece, fromFile, fromRank, mark, to, promotion] = parsed!;
    // Without a piece letter, a full from-square is long algebraic for any piece; otherwise a pawn.
    if (piece ? role !== SAN_ROLES[piece] : !(fromFile && fromRank) && role !== 'p') return false;
    if (squareName(move.to) !== to) return false;
    const from = squareName(move.from);
    if ((fromFile && from[0] !== fromFile) || (fromRank && from[1] !== fromRank)) return false;
    // A pawn token without a from-file (`d6`) is a push, never a capture (not even en passant).
    if (role === 'p' && !fromFile && from[0] !== to[0]) return false;
    if (promotion && move.promotion !== promotion.toLowerCase()) return false;
    if (mark === 'x' || mark === ':') {
      const enPassant = role === 'p' && move.to === board.epSquare && from[0] !== to[0];
      if (!board.squares[move.to] && !enPassant) return false;
    }
    return true;
  });
  if (candidates.length === 1) return { move: candidates[0] };
  return { error: candidates.length ? 'ambiguous' : 'invalid' };
}

/** Resolve a SAN (or long algebraic) token to a legal move in the given position, or null. */
export function sanToMove(board: BoardState, san: string): Move | null {
  const resolved = resolveSan(board, san);
  return 'move' in resolved ? resolved.move : null;
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
    // A `[%clk ...]` command outside a comment is not the next game's tag section.
    if (!depth && (ended || (ch === '[' && text[i + 1] !== '%' && sawMovetext))) finish(i);
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

const TAG_PAIR = /\[\s*(\w+)\s+"((?:\\.|[^"\\])*)"\s*\]/y;
// Some sites write unescaped quotes inside a value; accept a whole `[Name "..."]` line.
const LOOSE_TAG_LINE = /\[\s*(\w+)\s+"(.*)"\s*\][ \t]*(?=\n|$)/y;

/**
 * Read tag pairs only from the header that precedes the movetext, so a `[Site "y"]`
 * or `[FEN "..."]` quoted in a comment, or a following game's tags, cannot change this
 * game. Escape lines and `;` comments are skipped; `{...}` comments are kept for the movetext.
 */
function parseHeader(text: string): { tags: Record<string, string>; movetext: string } {
  const tags: Record<string, string> = {};
  const comments: string[] = [];
  let i = 0;
  while (i < text.length) {
    const ch = text[i];
    if (/\s/.test(ch)) {
      i++;
      continue;
    }
    if (ch === ';' || isEscapeLine(text, i)) {
      const end = text.indexOf('\n', i);
      i = end === -1 ? text.length : end + 1;
      continue;
    }
    if (ch === '{') {
      const end = text.indexOf('}', i);
      if (end === -1) break;
      comments.push(text.slice(i, end + 1));
      i = end + 1;
      continue;
    }
    if (ch !== '[') break;
    let match: RegExpExecArray | null = null;
    for (const pattern of [TAG_PAIR, LOOSE_TAG_LINE]) {
      pattern.lastIndex = i;
      match = pattern.exec(text);
      if (match) {
        i = pattern.lastIndex;
        break;
      }
    }
    if (!match) break;
    tags[match[1]] = match[2].replace(/\\(["\\])/g, '$1');
  }
  return { tags, movetext: [...comments, text.slice(i)].join(' ') };
}

type Token =
  | { type: 'move'; value: string }
  | { type: 'open' }
  | { type: 'close' }
  | { type: 'comment'; value: string }
  | { type: 'nag'; value: string }
  | { type: 'tag'; value: string }
  | { type: 'error'; value: string; reason: 'unclosed-command' | 'unmatched-comment' }
  | { type: 'result'; value: string };

const RESULTS = new Set(['1-0', '0-1', '1/2-1/2', '*']);
const NULL_MOVES = new Set(['--', 'Z0', '0000', '@@@@']);
/** Standalone annotation glyphs some tools print as words; mapped to their standard NAGs. */
const GLYPH_NAGS: ReadonlyMap<string, number> = new Map(
  Object.entries({
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
  }),
);

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
    if (ch === '}') {
      if (strict) throw new Error('Unmatched closing PGN comment brace');
      tokens.push({ type: 'error', value: ch, reason: 'unmatched-comment' });
      i++;
      continue;
    }
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
    // A tag pair after the header (usually the next game), or a stray `[%clk ...]` command.
    if (ch === '[' && /^\[(?:\s*\w+\s+"|%)/.test(movetext.slice(i, i + 64))) {
      let j = i + 1;
      let quoted = false;
      let closed = false;
      while (j < n) {
        const next = movetext[j++];
        if (next === '\\' && quoted) j++;
        else if (next === '"') quoted = !quoted;
        else if (next === ']' && !quoted) {
          closed = true;
          break;
        }
      }
      if (!closed) {
        if (strict) throw new Error('Unclosed PGN bracket command');
        tokens.push({ type: 'error', value: movetext.slice(i), reason: 'unclosed-command' });
        break;
      }
      if (movetext[i + 1] !== '%') tokens.push({ type: 'tag', value: movetext.slice(i, j) });
      i = j;
      continue;
    }
    // A whitespace/paren/comment-delimited word: move number, SAN, or result.
    let j = i + 1; // Consume at least one character, including an unrecognised opening bracket.
    while (j < n && !/[\s(){};$\[]/.test(movetext[j])) j++;
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
    const nag = GLYPH_NAGS.get(word);
    if (nag) {
      tokens.push({ type: 'nag', value: `$${nag}` });
      continue;
    }
    tokens.push({ type: 'move', value: word });
    // A suffix glyph (`Nf6!?`) is the move's NAG; SAN resolution ignores it.
    const glyphNag = GLYPH_NAGS.get(/[!?]{1,2}$/.exec(word)?.[0] ?? '');
    if (glyphNag) tokens.push({ type: 'nag', value: `$${glyphNag}` });
  }
  return tokens;
}

/** Parse a single game's PGN text into a variation tree. */
export function parsePgnGame(text: string, options: { strict?: boolean; mainlineOnly?: boolean } = {}): PgnGame {
  text = text.replace(/\r\n?/g, '\n');
  const { tags, movetext } = parseHeader(text);
  const startFen = tags.FEN || START_FEN;
  const tree = new GameTree(startFen);
  const tokens = tokenizeMovetext(movetext, options.strict);

  let last: GameNode = tree.root;
  const stack: Array<{ resume: GameNode; beforeMove: boolean; beforeComment?: string }> = [];
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
        last = stack.pop()?.resume ?? tree.root;
      }
      continue;
    }
    if (token.type === 'open') {
      if (stack.length >= MAX_VARIATION_DEPTH) {
        if (options.strict) throw new Error(`PGN variation depth exceeds ${MAX_VARIATION_DEPTH}`);
        stop('(', 'variation-depth');
        stoppedNesting = 1;
        continue;
      }
      stack.push({ resume: last, beforeMove: true });
      if (!options.mainlineOnly) last = last.parent ?? tree.root;
      continue;
    }
    if (token.type === 'close') {
      if (options.strict && !stack.length) throw new Error('Unmatched closing PGN variation');
      // A stray `)` must not restart the line from the root position.
      last = stack.pop()?.resume ?? last;
      continue;
    }
    // Aggregation must use the literal mainline, not child insertion order when
    // a RAV shares moves with its parent line. The authoritative PGN retains RAVs.
    if (options.mainlineOnly && stack.length) continue;
    if (token.type === 'error') {
      stop(token.value, token.reason);
      continue;
    }
    if (token.type === 'result') {
      if (!stack.length) {
        result = token.value;
        hasResult = true;
        ended = true;
      }
      continue;
    }
    if (token.type === 'tag') {
      if (options.strict) throw new Error(`PGN tag pair inside the movetext: ${token.value}`);
      stop(token.value, 'tag-pair');
      continue;
    }
    if (token.type === 'nag') {
      const nag = Number(token.value.slice(1));
      if (last !== tree.root && token.value.length > 1 && nag <= 255 && !last.nags?.includes(nag)) last.nags = [...(last.nags ?? []), nag];
      continue;
    }
    if (token.type === 'comment') {
      const variation = stack.at(-1);
      if (variation?.beforeMove) {
        variation.beforeComment = variation.beforeComment ? `${variation.beforeComment} ${token.value}` : token.value;
      } else {
        // Only top-level pre-move comments belong to the game introduction.
        last.comment = last.comment ? `${last.comment} ${token.value}` : token.value;
      }
      continue;
    }
    // move
    if (!options.strict && ended && !stack.length) {
      stop(token.value, 'after-result');
      continue;
    }
    tree.current = last;
    const resolved = resolveSan(parseFen(last.fen), token.value);
    if (!('move' in resolved)) {
      if (options.strict) throw new Error(`Invalid PGN move: ${token.value}`);
      // Interactive single-game loading remains tolerant, but never fabricates a game.
      stop(token.value, resolved.error === 'ambiguous' ? 'ambiguous-move' : NULL_MOVES.has(token.value) ? 'null-move' : 'invalid-move');
      continue;
    }
    last = tree.addMove(resolved.move);
    const variation = stack.at(-1);
    if (variation?.beforeMove) {
      if (variation.beforeComment) last.beforeComment = last.beforeComment ? `${last.beforeComment} ${variation.beforeComment}` : variation.beforeComment;
      variation.beforeMove = false;
    }
  }
  if (options.strict && stack.length) throw new Error('Unclosed PGN variation');
  tree.current = tree.root;
  return { tags, tree, result, hasResult, warnings };
}

const WARNING_TEXT: Record<PgnWarningReason, string> = {
  'invalid-move': 'not a legal move here',
  'ambiguous-move': 'more than one legal move matches',
  'null-move': 'null moves are not supported',
  'after-result': 'moves after the game result',
  'tag-pair': 'a tag pair inside the movetext, probably another game',
  'unclosed-command': 'unclosed bracket command',
  'unmatched-comment': 'unmatched closing comment brace',
  'variation-depth': `variation depth exceeds ${MAX_VARIATION_DEPTH}`,
};

/** One-line explanation of where and why a tolerant parse stopped a line. */
export function describePgnWarning(warning: PgnWarning): string {
  const token = warning.token.length > 60 ? `${warning.token.slice(0, 60)}…` : warning.token;
  return `${warning.variation ? 'A variation stopped' : 'Stopped'} at move ${warning.label}: ${token} (${WARNING_TEXT[warning.reason]})`;
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

// PGN has no escape for a closing comment brace; keep the text readable and reparsable.
function serializeComment(comment: string): string {
  return `{${comment.replace(/}/g, ']')}}`;
}

function serializeNode(node: GameNode, withNumber: boolean): string {
  const board = parseFen(node.parent!.fen);
  const number = board.turn === 'w' ? `${board.fullmove}. ` : withNumber ? `${board.fullmove}... ` : '';
  const nags = node.nags?.length ? ` ${node.nags.map((nag) => `$${nag}`).join(' ')}` : '';
  const comment = node.comment ? ` ${serializeComment(node.comment)}` : '';
  const beforeComment = node.beforeComment ? `${serializeComment(node.beforeComment)} ` : '';
  return `${beforeComment}${number}${node.san}${nags}${comment}`;
}

/** Tokens for the line continuing from `from`: its mainline, with each variation nested where it branches. */
function serializeLine(from: GameNode, forceNumber: boolean): string {
  const out: string[] = [];
  const pending: Array<string | { from: GameNode; force: boolean }> = [{ from, force: forceNumber }];
  while (pending.length) {
    const task = pending.pop()!;
    if (typeof task === 'string') {
      out.push(task);
      continue;
    }
    const [main, ...variations] = task.from.children;
    if (!main) continue;
    out.push(serializeNode(main, task.force));
    // A black move number is re-printed after a comment or a variation.
    if (main.children.length) pending.push({ from: main, force: variations.length > 0 || !!main.comment }, ' ');
    // Reverse the pending work so each variation precedes the mainline continuation.
    for (let i = variations.length - 1; i >= 0; i--) {
      const variation = variations[i];
      pending.push(')');
      if (variation.children.length) pending.push({ from: variation, force: !!variation.comment }, ' ');
      pending.push(serializeNode(variation, true), ' (');
    }
  }
  return out.join('');
}

/** Serialize every branch without using the call stack for nested variations. */
export function serializeMovetext(tree: GameTree): string {
  const intro = tree.root.comment ? [serializeComment(tree.root.comment)] : [];
  return [...intro, serializeLine(tree.root, true)].filter(Boolean).join(' ');
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
