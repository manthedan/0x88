#!/usr/bin/env node
import { spawn } from 'node:child_process';
import { createReadStream, createWriteStream, mkdirSync } from 'node:fs';
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
  let input = null;
  if (path.endsWith('.zst')) input = spawn('zstd', ['-dc', path], { stdio: ['ignore', 'pipe', 'pipe'] });
  else if (path.endsWith('.zip')) input = spawn('unzip', ['-p', path], { stdio: ['ignore', 'pipe', 'pipe'] });
  let stderr = '';
  if (input?.stderr) input.stderr.setEncoding('utf8').on('data', (chunk) => { stderr += chunk; });
  const stream = input ? input.stdout : createReadStream(path);
  const rl = createInterface({ input: stream, crlfDelay: Infinity });
  let lines = [];
  let completed = false;
  try {
    for await (const line of rl) {
      if (line.startsWith('[Event ') && lines.length) {
        const game = lines.join('\n').trim();
        if (game) yield game;
        lines = [];
      }
      lines.push(line);
    }
    completed = true;
    if (lines.length) {
      const game = lines.join('\n').trim();
      if (game) yield game;
    }
  } finally {
    rl.close();
    if (input && !completed) input.kill('SIGTERM');
  }
  if (input && completed) {
    const code = await new Promise((resolve) => input.on('close', resolve));
    if (code !== 0) throw new Error(`decompress failed for ${path}: ${stderr}`);
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

function parseTimeControl(tc) {
  if (!tc || tc === '-' || tc === '?') return null;
  const m = String(tc).match(/^(\d+)\+(\d+)$/);
  if (!m) return null;
  return { initial: Number(m[1]), increment: Number(m[2]) };
}
function isBulletLike(tc) {
  const parsed = parseTimeControl(tc);
  if (!parsed) return false;
  const estimatedSeconds = parsed.initial + 40 * parsed.increment;
  return estimatedSeconds < 180;
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
const excludeBullet = !process.argv.includes('--include-bullet');
const minInitialSeconds = Number(arg('--min-initial-seconds', '0'));
const minEstimatedSeconds = Number(arg('--min-estimated-seconds', '0'));
if (!input) throw new Error('Usage: node --experimental-strip-types scripts/lichess_pgn_to_training.mjs --pgn file.pgn[.zst|.zip] --out data/lichess_training.jsonl');

mkdirSync(dirname(out), { recursive: true });
const writer = createWriteStream(out, { encoding: 'utf8' });
let rowCount = 0, outputBytes = 0;
async function writeRow(row) {
  const line = JSON.stringify(row) + '\n';
  outputBytes += Buffer.byteLength(line);
  rowCount++;
  if (!writer.write(line)) await new Promise((resolve) => writer.once('drain', resolve));
}
let gamesSeen = 0, gamesAccepted = 0, parseFailures = 0, skippedVariant = 0, skippedTimeControl = 0;
for await (const game of streamGames(input)) {
  gamesSeen++;
  if (gamesAccepted >= maxGames || rowCount >= maxPositions) break;
  const tags = parseTags(game);
  const result = tags.Result ?? '*';
  if (!['1-0', '0-1', '1/2-1/2'].includes(result)) continue;
  if (tags.Variant && tags.Variant !== 'Standard') { skippedVariant++; continue; }
  const tc = parseTimeControl(tags.TimeControl);
  if (excludeBullet && isBulletLike(tags.TimeControl)) { skippedTimeControl++; continue; }
  if (tc && minInitialSeconds > 0 && tc.initial < minInitialSeconds) { skippedTimeControl++; continue; }
  if (tc && minEstimatedSeconds > 0 && tc.initial + 40 * tc.increment < minEstimatedSeconds) { skippedTimeControl++; continue; }
  const whiteElo = Number(tags.WhiteElo ?? 0), blackElo = Number(tags.BlackElo ?? 0);
  if (!noEloFilter && Math.min(whiteElo, blackElo) < minElo) continue;
  let board = parseFen(tags.FEN ?? START_FEN);
  let acceptedInGame = 0;
  try {
    for (const san of sanTokens(moveText(game))) {
      const before = boardToFen(board);
      const turn = board.turn;
      const move = parseSanMove(board, san);
      if (acceptedInGame >= skipPlies && acceptedInGame < maxPliesPerGame && rowCount < maxPositions) {
        const wdl = resultWdl(result, turn);
        await writeRow({ id: `${teacher}_${String(gamesAccepted).padStart(6, '0')}_${String(acceptedInGame).padStart(3, '0')}`, fen: before, policy: { [moveToUci(move)]: 1.0 }, wdl, q: wdl[0] - wdl[2], teacher, result, white_elo: whiteElo, black_elo: blackElo, time_control: tags.TimeControl ?? '', variant: tags.Variant ?? 'Standard' });
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
await new Promise((resolve, reject) => writer.end((err) => err ? reject(err) : resolve()));
console.log(`METRIC lichess_games_seen=${gamesSeen}`);
console.log(`METRIC lichess_games_accepted=${gamesAccepted}`);
console.log(`METRIC lichess_parse_failures=${parseFailures}`);
console.log(`METRIC lichess_skipped_variant=${skippedVariant}`);
console.log(`METRIC lichess_skipped_time_control=${skippedTimeControl}`);
console.log(`METRIC lichess_training_rows=${rowCount}`);
console.log(`METRIC lichess_output_bytes=${outputBytes}`);
