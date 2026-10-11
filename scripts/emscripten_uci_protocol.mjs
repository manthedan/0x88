// Shared Node/browser fixed-depth protocol. No timing is run by importing this module.
export const ROTATED_FEN_SUITE = [
  ['Start position', 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'],
  ['Ruy Lopez ply 2', 'rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2'],
  ['Ruy Lopez ply 4', 'r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3'],
  ['Ruy Lopez ply 6', 'r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4'],
  ['Ruy Lopez ply 8', 'r1bqkb1r/1ppp1ppp/p1n2n2/4p3/B3P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 5'],
  ['Ruy Lopez ply 10', 'r1bqk2r/1pppbppp/p1n2n2/4p3/B3P3/5N2/PPPP1PPP/RNBQ1RK1 w kq - 4 6'],
  ['Ruy Lopez ply 12', 'r1bqk2r/2ppbppp/p1n2n2/1p2p3/B3P3/5N2/PPPP1PPP/RNBQR1K1 w kq b6 0 7'],
  ['Ruy Lopez ply 14', 'r1bqk2r/2p1bppp/p1np1n2/1p2p3/4P3/1B3N2/PPPP1PPP/RNBQR1K1 w kq - 0 8'],
  ['Ruy Lopez ply 16', 'r1bq1rk1/2p1bppp/p1np1n2/1p2p3/4P3/1BP2N2/PP1P1PPP/RNBQR1K1 w - - 1 9'],
  ['Ruy Lopez ply 18', 'r2q1rk1/1bp1bppp/p1np1n2/1p2p3/4P3/1BP2N1P/PP1P1PP1/RNBQR1K1 w - - 1 10'],
  ['Ruy Lopez ply 20', 'rn1q1rk1/1bp1bppp/p2p1n2/1p2p3/3PP3/1BP2N1P/PP3PP1/RNBQR1K1 w - - 1 11'],
  ['Ruy Lopez ply 22', 'r2q1rk1/1bpnbppp/p2p1n2/1p2p3/3PP3/1BP2N1P/PP1N1PP1/R1BQR1K1 w - - 3 12'],
  ['Ruy Lopez ply 24', 'r2q1rk1/1b1nbppp/p2p1n2/1pp1p3/P2PP3/1BP2N1P/1P1N1PP1/R1BQR1K1 w - c6 0 13'],
  ['Ruy Lopez ply 26', 'r2q1rk1/1b1nbppp/p2p1n2/1p1Pp3/P1p1P3/1BP2N1P/1P1N1PP1/R1BQR1K1 w - - 0 14'],
  ['Ruy Lopez ply 28', 'r2q1rk1/1b2bppp/p2p1n2/1pnPp3/P1p1P3/2P2N1P/1PBN1PP1/R1BQR1K1 w - - 2 15'],
  ['Ruy Lopez ply 30', 'r2q1rk1/1b1nbppp/p2p4/1pnPp3/P1p1P3/2P4P/1PBN1PPN/R1BQR1K1 w - - 4 16'],
  ['Ruy Lopez ply 32', 'r2q1rk1/1b1nbppp/p2p4/1pnPp3/P3P3/1pP4P/2BN1PPN/R1BQR1K1 w - - 0 17'],
  ['Ruy Lopez ply 34', 'r2q1rk1/1b1nbp1p/p2p2p1/1pnPp3/P3P3/1BP4P/3N1PPN/R1BQR1K1 w - - 0 18'],
  ['Ruy Lopez ply 36', 'r2qr1k1/1b1nbp1p/p2p2p1/1pnPp3/P3P3/1BP4P/1B1N1PPN/R2QR1K1 w - - 2 19'],
  ['Ruy Lopez ply 38', 'r1bqr1k1/3nbp1p/p2p2p1/1pnPp3/P3P1P1/1BP4P/1B1N1P1N/R2QR1K1 w - - 1 20'],
];

const uciMove = /^[a-h][1-8][a-h][1-8][qrbn]?$/;

export function parseBestmove(line) {
  const tokens = line.trim().split(/\s+/);
  if (tokens[0] !== 'bestmove' || ![2, 4].includes(tokens.length)
    || (!uciMove.test(tokens[1]) && !['0000', '(none)'].includes(tokens[1]))
    || (tokens.length === 4 && (tokens[2] !== 'ponder' || !uciMove.test(tokens[3])))) {
    throw new Error(`Malformed bestmove record: ${line}`);
  }
  return tokens[1];
}

export function parseInfo(line) {
  const tokens = line.trim().split(/\s+/);
  const integer = (token, signed = false) => {
    if (!(signed ? /^-?\d+$/ : /^\d+$/).test(token ?? '') || !Number.isSafeInteger(Number(token))) {
      throw new Error(`Malformed UCI info record: ${line}`);
    }
    return Number(token);
  };
  const get = (key) => {
    const index = tokens.indexOf(key);
    if (index < 0) return null;
    if (tokens.lastIndexOf(key) !== index) throw new Error(`Malformed UCI info record: ${line}`);
    return integer(tokens[index + 1]);
  };
  const scoreIndex = tokens.indexOf('score');
  let score = null;
  if (scoreIndex >= 0) {
    if (!['cp', 'mate'].includes(tokens[scoreIndex + 1]) || tokens.lastIndexOf('score') !== scoreIndex) {
      throw new Error(`Malformed UCI info record: ${line}`);
    }
    score = `${tokens[scoreIndex + 1]} ${integer(tokens[scoreIndex + 2], true)}`;
  }
  const pvIndex = tokens.indexOf('pv');
  const moves = pvIndex >= 0 ? tokens.slice(pvIndex + 1) : null;
  if (moves && (!moves.length || moves.some((move) => !uciMove.test(move)))) throw new Error(`Malformed UCI PV record: ${line}`);
  return { depth: get('depth'), nodes: get('nodes'), nps: get('nps'), timeMs: get('time'),
    score, pv: moves ? moves.join(' ') : null };
}

export function parseUciSearch(lines, depth) {
  const bestLines = lines.filter((line) => /^bestmove(?:\s|$)/.test(line));
  if (bestLines.length !== 1) throw new Error(`Incomplete UCI search: expected one bestmove, got ${bestLines.length}`);
  const bestmove = parseBestmove(bestLines[0]);
  if (!uciMove.test(bestmove)) throw new Error('Incomplete nonterminal UCI search: null bestmove');
  const beforeBest = lines.slice(0, lines.indexOf(bestLines[0]));
  // Validate every info record before selecting the final PV/count. Invalid
  // trailing fields must not disappear behind an earlier valid search line.
  const infos = beforeBest.filter((line) => /^info\s/.test(line) && !/^info\s+string\b/.test(line)).map(parseInfo);
  const info = infos.filter((record) => record.pv !== null).at(-1);
  const nodes = infos.filter((record) => record.nodes !== null).at(-1)?.nodes;
  if (!info || info.depth !== depth || !Number.isSafeInteger(nodes) || nodes <= 0 || !info.score || !info.pv) {
    throw new Error(`Incomplete depth ${depth} UCI search output`);
  }
  return { bestmove, ...info, nodes };
}

export function median(values) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

export function exactSearch(a, b) {
  // Missing output must never count as parity, even if both sides are missing it.
  return ['depth', 'bestmove', 'score', 'nodes', 'pv'].every((key) => a[key] != null && a[key] === b[key]);
}

export function summarizeCell(rows) {
  const nodesTotal = rows.reduce((sum, r) => sum + r.nodes, 0);
  const wallMs = rows.reduce((sum, r) => sum + r.wallMs, 0);
  const engineMs = rows.reduce((sum, r) => sum + r.timeMs, 0);
  return { nodesTotal, wallMs, engineMs, wallNps: nodesTotal * 1000 / wallMs,
    engineNps: nodesTotal * 1000 / engineMs, npsMedian: median(rows.map((r) => r.nps)) };
}

// These artifacts expose synchronous search. Measuring around ccall includes
// emitted UCI output but excludes TT reset / position setup / worker messages.
export function runSuite(module, stdout, { depth, hash = 16, positions = 20, warmup = true }) {
  const command = (s) => module.ccall('command', null, ['string'], [s]);
  const expect = (text, response) => {
    const start = stdout.length;
    command(text);
    if (!stdout.slice(start).includes(response)) throw Error(`Missing ${response} after ${text}`);
  };
  expect('uci', 'uciok');
  command('setoption name Threads value 1');
  command(`setoption name Hash value ${hash}`);
  expect('isready', 'readyok');
  const search = ([position, fen], targetDepth) => {
    command('ucinewgame');
    expect('isready', 'readyok');
    command(`position fen ${fen}`);
    const start = stdout.length;
    const t = performance.now();
    command(`go depth ${targetDepth}`);
    const wallMs = performance.now() - t;
    const lines = stdout.slice(start);
    const info = parseUciSearch(lines, targetDepth);
    if (info.timeMs == null || info.nps == null) {
      throw Error(`Incomplete depth ${targetDepth} output for ${position}: ${lines.join('\n')}`);
    }
    return { position, ...info, wallMs };
  };
  if (warmup) search(ROTATED_FEN_SUITE[0], depth);
  return ROTATED_FEN_SUITE.slice(0, positions).map((p) => search(p, depth));
}

// Rotate then reverse alternating rounds. Every artifact gets every slot over
// ten rounds for the five-cell matrix (current + four builds).
export function roundOrder(items, round) {
  const shift = round % items.length;
  const rotated = [...items.slice(shift), ...items.slice(0, shift)];
  return round % 2 ? rotated.reverse() : rotated;
}
