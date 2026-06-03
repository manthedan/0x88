import { parseFen, START_FEN, type BoardState } from './board.ts';
import { legalMoves } from './movegen.ts';
import { type Move } from './moveCodec.ts';
import { moveToSan } from './san.ts';
import { GameTree, type GameNode } from '../lc0/gameTree.ts';

export interface PgnGame {
  tags: Record<string, string>;
  tree: GameTree;
  result: string;
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
  const lines = text.replace(/\r\n?/g, '\n').split('\n');
  const games: string[] = [];
  let current: string[] = [];
  let sawMovetext = false;
  for (const line of lines) {
    const isTag = /^\s*\[/.test(line);
    // A tag line that starts a new game after we've already seen movetext closes the previous game.
    if (isTag && sawMovetext) {
      games.push(current.join('\n').trim());
      current = [];
      sawMovetext = false;
    }
    if (!isTag && line.trim()) sawMovetext = true;
    current.push(line);
  }
  if (current.join('').trim()) games.push(current.join('\n').trim());
  return games.filter(Boolean);
}

function parseTags(text: string): Record<string, string> {
  const tags: Record<string, string> = {};
  const re = /\[(\w+)\s+"([^"]*)"\]/g;
  let match: RegExpExecArray | null;
  while ((match = re.exec(text))) tags[match[1]] = match[2];
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
function tokenizeMovetext(movetext: string): Token[] {
  const tokens: Token[] = [];
  let i = 0;
  const n = movetext.length;
  while (i < n) {
    const ch = movetext[i];
    if (/\s/.test(ch)) { i++; continue; }
    if (ch === '{') {
      const end = movetext.indexOf('}', i);
      const stop = end === -1 ? n : end;
      tokens.push({ type: 'comment', value: movetext.slice(i + 1, stop).trim() });
      i = stop + 1;
      continue;
    }
    if (ch === '(') { tokens.push({ type: 'open' }); i++; continue; }
    if (ch === ')') { tokens.push({ type: 'close' }); i++; continue; }
    if (ch === ';') { const end = movetext.indexOf('\n', i); i = end === -1 ? n : end + 1; continue; }
    if (ch === '$') {
      let j = i + 1;
      while (j < n && /\d/.test(movetext[j])) j++;
      tokens.push({ type: 'nag', value: movetext.slice(i, j) });
      i = j;
      continue;
    }
    // A whitespace/paren/comment-delimited word: move number, SAN, or result.
    let j = i;
    while (j < n && !/[\s(){};]/.test(movetext[j])) j++;
    let word = movetext.slice(i, j);
    i = j;
    if (RESULTS.has(word)) { tokens.push({ type: 'result', value: word }); continue; }
    // Strip a leading move number like "12." or "12..." possibly fused to the SAN.
    word = word.replace(/^\d+\.(\.\.)?/, '');
    if (!word || word === '...') continue;
    if (RESULTS.has(word)) { tokens.push({ type: 'result', value: word }); continue; }
    tokens.push({ type: 'move', value: word });
  }
  return tokens;
}

/** Parse a single game's PGN text into a variation tree. */
export function parsePgnGame(text: string): PgnGame {
  const tags = parseTags(text);
  const startFen = tags.FEN || START_FEN;
  const tree = new GameTree(startFen);
  // Movetext is everything after the last tag line.
  const movetext = text.replace(/\[(\w+)\s+"([^"]*)"\]\s*/g, '');
  const tokens = tokenizeMovetext(movetext);

  let last: GameNode = tree.root;
  const stack: GameNode[] = [];
  let result = tags.Result || '*';
  for (const token of tokens) {
    if (token.type === 'open') { stack.push(last); last = last.parent ?? tree.root; continue; }
    if (token.type === 'close') { last = stack.pop() ?? tree.root; continue; }
    if (token.type === 'result') { result = token.value; continue; }
    if (token.type === 'nag') { continue; }
    if (token.type === 'comment') { if (last !== tree.root) last.comment = last.comment ? `${last.comment} ${token.value}` : token.value; continue; }
    // move
    tree.current = last;
    const move = sanToMove(parseFen(last.fen), token.value);
    if (!move) continue; // skip unparseable tokens rather than abort the whole game
    last = tree.addMove(move);
  }
  tree.current = tree.root;
  return { tags, tree, result };
}

/** Parse every game in a multi-game PGN string. */
export function parsePgnGames(text: string): PgnGame[] {
  return splitPgnGames(text).map(parsePgnGame);
}

function serializeNode(node: GameNode, withNumber: boolean): string {
  const board = parseFen(node.parent!.fen);
  const number = board.turn === 'w'
    ? `${board.fullmove}. `
    : withNumber ? `${board.fullmove}... ` : '';
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
  const header = Object.entries(tagPairs).map(([k, v]) => `[${k} "${v}"]`).join('\n');
  const movetext = serializeMovetext(tree);
  return `${header}\n\n${movetext} ${result}`.trim();
}
