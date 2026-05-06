#!/usr/bin/env node
import { spawn } from 'node:child_process';
import { createReadStream, mkdirSync, writeFileSync } from 'node:fs';
import { createInterface } from 'node:readline';
import { dirname } from 'node:path';
import { parseFen, boardToFen, START_FEN } from '../src/chess/board.ts';
import { legalMoves, makeMove, inCheck } from '../src/chess/movegen.ts';
import { moveToUci } from '../src/chess/moveCodec.ts';

function arg(name, fallback = undefined) {
  const prefix = `${name}=`;
  const inline = process.argv.find((value) => value.startsWith(prefix));
  if (inline) return inline.slice(prefix.length);
  const i = process.argv.indexOf(name);
  return i >= 0 ? process.argv[i + 1] : fallback;
}

async function* streamGames(path) {
  const input = path.endsWith('.zst') ? spawn('zstd', ['-dc', path], { stdio: ['ignore', 'pipe', 'pipe'] }) : null;
  let stderr = '';
  if (input?.stderr) input.stderr.setEncoding('utf8').on('data', (chunk) => { stderr += chunk; });
  const stream = input ? input.stdout : createReadStream(path);
  const rl = createInterface({ input: stream, crlfDelay: Infinity });
  let lines = [];
  for await (const line of rl) {
    if (line.startsWith('[Event ') && lines.length) {
      const game = lines.join('\n').trim();
      if (game) yield game;
      lines = [];
    }
    lines.push(line);
  }
  if (lines.length) {
    const game = lines.join('\n').trim();
    if (game) yield game;
  }
  if (input) {
    const code = await new Promise((resolve) => input.on('close', resolve));
    if (code !== 0) throw new Error(`zstd -dc failed for ${path}: ${stderr}`);
  }
}

function parseTags(game) {
  const tags = {};
  for (const line of game.split('\n')) {
    const m = line.match(/^\[([^ ]+) "(.*)"\]$/);
    if (m) tags[m[1]] = m[2];
  }
  return tags;
}

function moveText(game) {
  return game.split('\n').filter((line) => !line.startsWith('[')).join(' ');
}

function stripVariations(s) {
  let out = '', depth = 0;
  for (const ch of s) {
    if (ch === '(') { depth++; continue; }
    if (ch === ')') { depth = Math.max(0, depth - 1); continue; }
    if (depth === 0) out += ch;
  }
  return out;
}

function sanTokens(text) {
  return stripVariations(text)
    .replace(/\{[^}]*\}/g, ' ')
    .replace(/;[^\n]*/g, ' ')
    .split(/\s+/)
    .map((t) => t.trim())
    .filter(Boolean)
    .filter((t) => !/^\d+\.{1,3}$/.test(t))
    .filter((t) => !/^\$\d+$/.test(t))
    .filter((t) => !/^(1-0|0-1|1\/2-1\/2|\*)$/.test(t));
}

function resultWdl(result, turn) {
  if (result === '1/2-1/2') return [0, 1, 0];
  if (result === '1-0') return turn === 'w' ? [1, 0, 0] : [0, 0, 1];
  if (result === '0-1') return turn === 'b' ? [1, 0, 0] : [0, 0, 1];
  return [0, 1, 0];
}

function pieceRole(board, move) {
  const piece = board.squares[move.from];
  if (!piece) throw new Error(`No piece on ${move.from}`);
  return piece[1];
}

function sanMatches(board, move, rawSan) {
  let san = rawSan.replace(/[!?]+/g, '').replace(/[+#]+$/g, '');
  san = san.replace(/^0-0/, 'O-O');
  const uci = moveToUci(move);
  const role = pieceRole(board, move);
  if (san === 'O-O' || san === 'O-O-O') return role === 'k' && (uci === 'e1g1' || uci === 'e8g8' || uci === 'e1c1' || uci === 'e8c8') && (san === 'O-O' ? uci[2] === 'g' : uci[2] === 'c');
  const promo = san.match(/=([NBRQ])$/)?.[1]?.toLowerCase();
  san = san.replace(/=([NBRQ])$/, '');
  if ((move.promotion ?? '') !== (promo ?? '')) return false;
  const dest = san.slice(-2);
  if (uci.slice(2, 4) !== dest) return false;
  const pieceLetter = /^[NBRQK]/.test(san) ? san[0] : '';
  const wantedRole = pieceLetter ? pieceLetter.toLowerCase() : 'p';
  if (role !== wantedRole) return false;
  const capture = san.includes('x');
  const targetOccupied = board.squares[move.to] !== null;
  const epCapture = role === 'p' && board.epSquare === move.to && board.squares[move.to] === null && (move.from % 8) !== (move.to % 8);
  if (capture !== (targetOccupied || epCapture)) return false;
  const core = san.slice(pieceLetter ? 1 : 0, -2).replace('x', '');
  if (core.length === 1) {
    const c = core[0];
    if (/[a-h]/.test(c) && uci[0] !== c) return false;
    if (/[1-8]/.test(c) && uci[1] !== c) return false;
  } else if (core.length === 2) {
    if (uci.slice(0, 2) !== core) return false;
  }
  return true;
}

function parseSanMove(board, san) {
  const matches = legalMoves(board).filter((move) => sanMatches(board, move, san));
  if (matches.length !== 1) throw new Error(`SAN ${san} matched ${matches.length} legal moves in ${boardToFen(board)}`);
  return matches[0];
}

const input = arg('--pgn');
const out = arg('--out', 'data/lichess_training.jsonl');
const maxGames = Number(arg('--max-games', '100'));
const maxPositions = Number(arg('--max-positions', '10000'));
const minElo = Number(arg('--min-elo', '1800'));
const teacher = arg('--teacher', 'lichess_pgn');
const noEloFilter = process.argv.includes('--no-elo-filter');
const skipPlies = Number(arg('--skip-plies', '4'));
const maxPliesPerGame = Number(arg('--max-plies-per-game', '80'));
if (!input) throw new Error('Usage: node --experimental-strip-types scripts/lichess_pgn_to_training.mjs --pgn file.pgn[.zst] --out data/lichess_training.jsonl');

const rows = [];
let gamesSeen = 0, gamesAccepted = 0, parseFailures = 0;
for await (const game of streamGames(input)) {
  gamesSeen++;
  if (gamesAccepted >= maxGames || rows.length >= maxPositions) break;
  const tags = parseTags(game);
  const result = tags.Result ?? '*';
  if (!['1-0', '0-1', '1/2-1/2'].includes(result)) continue;
  if (tags.Variant && tags.Variant !== 'Standard') continue;
  const whiteElo = Number(tags.WhiteElo ?? 0), blackElo = Number(tags.BlackElo ?? 0);
  if (!noEloFilter && Math.min(whiteElo, blackElo) < minElo) continue;
  let board = parseFen(tags.FEN ?? START_FEN);
  let acceptedInGame = 0;
  try {
    for (const san of sanTokens(moveText(game))) {
      const before = boardToFen(board);
      const turn = board.turn;
      const move = parseSanMove(board, san);
      if (acceptedInGame >= skipPlies && acceptedInGame < maxPliesPerGame && rows.length < maxPositions) {
        const wdl = resultWdl(result, turn);
        rows.push({ id: `${teacher}_${String(gamesAccepted).padStart(6, '0')}_${String(acceptedInGame).padStart(3, '0')}`, fen: before, policy: { [moveToUci(move)]: 1.0 }, wdl, q: wdl[0] - wdl[2], teacher, result, white_elo: whiteElo, black_elo: blackElo });
      }
      board = makeMove(board, move);
      acceptedInGame++;
      if (inCheck(board, board.turn) && legalMoves(board).length === 0) break;
    }
    gamesAccepted++;
  } catch (err) {
    parseFailures++;
  }
}
mkdirSync(dirname(out), { recursive: true });
writeFileSync(out, rows.map((r) => JSON.stringify(r)).join('\n') + (rows.length ? '\n' : ''));
console.log(`METRIC lichess_games_seen=${gamesSeen}`);
console.log(`METRIC lichess_games_accepted=${gamesAccepted}`);
console.log(`METRIC lichess_parse_failures=${parseFailures}`);
console.log(`METRIC lichess_training_rows=${rows.length}`);
console.log(`METRIC lichess_output_bytes=${Buffer.byteLength(rows.map((r) => JSON.stringify(r)).join('\n'))}`);
