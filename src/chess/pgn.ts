import { type GameNode, GameTree } from '../lc0/gameTree.ts';
import { type BoardState, parseFen, START_FEN } from './board.ts';
import { type Move } from './moveCodec.ts';
import { legalMoves } from './movegen.ts';
import { moveToSan } from './san.ts';

export interface PgnGame {
  tags: Record<string, string>;
  tree: GameTree;
  result: string;
  /** An explicit valid result, rather than the parser's default for unfinished input. */
  hasResult: boolean;
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
    if (/\s/.test(ch)) { i++; continue; }
    if (ch === '{') {
      const end = text.indexOf('}', i + 1);
      i = end === -1 ? text.length : end + 1;
      continue;
    }
    if (ch === ';') {
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
    if (ch === '(') { depth++; i++; continue; }
    if (ch === ')') { depth = Math.max(0, depth - 1); i++; continue; }
    if (ch === '$') { i++; while (/\d/.test(text[i] ?? '')) i++; continue; }
    let j = i;
    while (j < text.length && !/[\s(){};$\[]/.test(text[j])) j++;
    if (j === i) { i++; continue; } // Malformed punctuation is rejected by the parser, never a loop.
    const word = text.slice(i, j).replace(/^\d+\.(\.\.)?/, '');
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
    if (ch === ';') {
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
    word = word.replace(/^\d+\.(\.\.)?/, '');
    if (!word || word === '...') continue;
    if (RESULTS.has(word)) {
      tokens.push({ type: 'result', value: word });
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
  for (const token of tokens) {
    if (token.type === 'open') {
      stack.push(last);
      if (!options.mainlineOnly) last = last.parent ?? tree.root;
      continue;
    }
    if (token.type === 'close') {
      if (options.strict && !stack.length) throw new Error('Unmatched closing PGN variation');
      last = stack.pop() ?? tree.root;
      continue;
    }
    // Aggregation must use the literal mainline, not child insertion order when
    // a RAV shares moves with its parent line. The authoritative PGN retains RAVs.
    if (options.mainlineOnly && stack.length) continue;
    if (token.type === 'result') {
      if (!stack.length) {
        result = token.value;
        hasResult = true;
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
    tree.current = last;
    const move = sanToMove(parseFen(last.fen), token.value);
    if (!move) {
      if (options.strict) throw new Error(`Invalid PGN move: ${token.value}`);
      continue; // Interactive single-game loading remains tolerant.
    }
    last = tree.addMove(move);
  }
  if (options.strict && stack.length) throw new Error('Unclosed PGN variation');
  tree.current = tree.root;
  return { tags, tree, result, hasResult };
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
    .map(([k, v]) => `[${k} "${v.replace(/\\/g, '\\\\').replace(/"/g, '\\"').replace(/[\r\n]+/g, ' ')}"]`)
    .join('\n');
  const movetext = serializeMovetext(tree);
  return `${header}\n\n${movetext} ${result}`.trim();
}
