#!/usr/bin/env node
import { spawn } from 'node:child_process';
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname } from 'node:path';
import { parseFen, boardToFen } from '../src/chess/board.ts';
import { inCheck, legalMoves, makeMove } from '../src/chess/movegen.ts';
import { moveToUci } from '../src/chess/moveCodec.ts';
import { chooseMove, classicPuctPolicy, actionValuePuctPolicy, auxPuctPolicy } from '../src/search/puct.ts';
import { OnnxEvaluator } from '../src/nn/onnxEvaluator.ts';
import { SquareFormerEvaluator } from '../src/nn/squareformerEvaluator.ts';

function arg(name, fallback = undefined) {
  const prefix = `${name}=`;
  const inline = process.argv.find((value) => value.startsWith(prefix));
  if (inline) return inline.slice(prefix.length);
  const i = process.argv.indexOf(name);
  return i >= 0 ? process.argv[i + 1] : fallback;
}
function boolArg(name, fallback=false) { const v = arg(name, fallback ? '1' : '0'); return ['1','true','yes','on'].includes(String(v).toLowerCase()); }
function terminalWhiteScore(board) {
  const moves = legalMoves(board);
  if (moves.length) return null;
  if (!inCheck(board)) return 0.5;
  return board.turn === 'w' ? 0 : 1;
}
function elo(scoreRate) { const s = Math.min(0.999, Math.max(0.001, scoreRate)); return 400 * Math.log10(s / (1 - s)); }
function normalApproxCi(scoreRate, games) {
  const s = Math.min(0.999, Math.max(0.001, scoreRate));
  const se = Math.sqrt(Math.max(1e-9, s * (1 - s) / Math.max(1, games)));
  const deriv = 400 / Math.log(10) * (1 / s + 1 / (1 - s));
  return 1.96 * se * deriv;
}
function parseCandidate(s) {
  const [name, onnx, meta, mode = 'puct', avWeight = '0.25', rankWeight = '0', regretWeight = '0', riskWeight = '0', uncertaintyWeight = '0'] = s.split(':');
  if (!name || !onnx || !meta) throw new Error('Use --candidate=name:path.onnx:path.meta.json[:mode[:avWeight[:rankWeight[:regretWeight]]]]');
  if (!['puct', 'av', 'aux'].includes(mode)) throw new Error(`Bad candidate mode: ${mode}`);
  return { name, onnx, meta, mode, avWeight: Number(avWeight), rankWeight: Number(rankWeight), regretWeight: Number(regretWeight), riskWeight: Number(riskWeight), uncertaintyWeight: Number(uncertaintyWeight) };
}
function loadOpenings(path) {
  return readFileSync(path, 'utf8').split(/\r?\n/).map(l => l.trim()).filter(l => l && !l.startsWith('#')).map(l => l.split(/\s+#/)[0].trim());
}
class UciEngine {
  constructor(name, command, args=[]) { this.name = name; this.p = spawn(command, args, { stdio: ['pipe','pipe','pipe'] }); this.buf = []; this.waiters = []; this.p.stdout.setEncoding('utf8'); this.p.stdout.on('data', d => this.onData(d)); this.p.stderr.on('data', d => process.stderr.write(`[${name} stderr] ${d}`)); }
  onData(d) { for (const line of d.split(/\r?\n/)) { if (!line) continue; this.buf.push(line); for (const w of [...this.waiters]) if (w.pred(line)) { this.waiters = this.waiters.filter(x => x !== w); w.resolve(line); } } }
  send(s) { this.p.stdin.write(s + '\n'); }
  waitFor(pred, timeoutMs=30000) { const hit = this.buf.find(pred); if (hit) return Promise.resolve(hit); return new Promise((resolve, reject) => { const t=setTimeout(()=>{this.waiters=this.waiters.filter(w=>w.resolve!==resolve); reject(new Error(`${this.name} timeout`));}, timeoutMs); this.waiters.push({ pred, resolve:(v)=>{clearTimeout(t); resolve(v);} }); }); }
  async init(options={}) { this.send('uci'); await this.waitFor(l => l === 'uciok'); for (const [k,v] of Object.entries(options)) this.send(`setoption name ${k} value ${v}`); this.send('isready'); await this.waitFor(l => l === 'readyok'); }
  async bestmove(fen, search) {
    this.buf = [];
    this.send(`position fen ${fen}`);
    const go = typeof search === 'object' && search ? (search.go || `nodes ${search.nodes || 1}`) : `nodes ${search}`;
    this.send(`go ${go}`);
    const line = await this.waitFor(l => l.startsWith('bestmove '), 60000);
    return line.split(/\s+/)[1];
  }
  quit() { try { this.send('quit'); } catch {} }
}
function stockfishAnchors(path, levels, nodes, threads, hash) {
  return levels.map(level => ({ type:'uci', name:`stockfish_${level}`, command:path, search:{ go:`nodes ${nodes}`, nodes }, options:{ Threads:threads, Hash:hash, UCI_LimitStrength:'true', UCI_Elo:String(level) }}));
}
function stockfishLiteAnchors(path, levels, nodes, hash) {
  return levels.map(level => ({ type:'uci', name:`stockfish_lite_${level}`, command:path, search:{ go:`nodes ${nodes}`, nodes }, options:{ Hash:hash, UCI_LimitStrength:'true', UCI_Elo:String(level) }}));
}
function stockfishShallowAnchors(path, depths, threads, hash) {
  return depths.map(depth => ({ type:'uci', name:`stockfish_depth${depth}`, command:path, search:{ go:`depth ${depth}`, depth }, options:{ Threads:threads, Hash:hash, UCI_LimitStrength:'false' }}));
}
function stockfishLiteShallowAnchors(path, depths, hash) {
  return depths.map(depth => ({ type:'uci', name:`stockfish_lite_depth${depth}`, command:path, search:{ go:`depth ${depth}`, depth }, options:{ Hash:hash, UCI_LimitStrength:'false' }}));
}
function parseSearch(raw, defaultNodes) {
  if (!raw) return { go:`nodes ${defaultNodes}`, nodes:defaultNodes };
  if (/^\d+$/.test(raw)) return { go:`nodes ${Number(raw)}`, nodes:Number(raw) };
  const m = raw.match(/^(nodes|depth|movetime)=(\d+)$/);
  if (!m) throw new Error(`Bad UCI search spec '${raw}', use N, nodes=N, depth=N, or movetime=N`);
  return { go:`${m[1]} ${Number(m[2])}`, [m[1]]:Number(m[2]) };
}
function customUciAnchors(spec, defaultNodes) {
  if (!spec) return [];
  return spec.split(',').filter(Boolean).map(entry => {
    const [name, command, searchRaw] = entry.split('|');
    if (!name || !command) throw new Error('--uci-anchors=name|/path/to/uci|nodes=N|depth=N|movetime=N,...');
    return { type:'uci', name, command, search:parseSearch(searchRaw, defaultNodes), options:{} };
  });
}
function ptnmlFromPairScores(pairScores) {
  const c = { '0':0, '0.5':0, '1':0, '1.5':0, '2':0 };
  for (const x of pairScores) c[String(x)]++;
  return c;
}

const candidateSpec = parseCandidate(arg('--candidate', ''));
const openingsFile = arg('--openings-file', 'eval/opening_suite_uho_lite_v1.fen');
const openingStart = Number(arg('--opening-start', '0'));
const openingCountRaw = arg('--opening-count', '');
const pairs = Number(arg('--pairs', '20'));
const visits = Number(arg('--visits', '64'));
const cpuct = Number(arg('--cpuct', '1.5'));
const maxPlies = Number(arg('--max-plies', '120'));
const batchSize = Number(arg('--batch-size', '16'));
const stockfishPath = arg('--stockfish', '.local_engines/stockfish_pkg/usr/games/stockfish');
const sfLevels = arg('--stockfish-levels', '1320,1600').split(',').filter(Boolean).map(Number);
const sfNodes = Number(arg('--stockfish-nodes', '64'));
const threads = Number(arg('--threads', '1'));
const hash = Number(arg('--hash', '16'));
const out = arg('--out', 'artifacts/anchor_arena/uci_anchor_arena.json');
const includeStockfish = boolArg('--include-stockfish', true);
const includeStockfishLite = boolArg('--include-stockfish-lite', false);
const stockfishLitePath = arg('--stockfish-lite', '.local_engines/stockfish-lite-single.sh');
const sfLiteLevels = arg('--stockfish-lite-levels', sfLevels.join(',')).split(',').filter(Boolean).map(Number);
const sfLiteNodes = Number(arg('--stockfish-lite-nodes', String(sfNodes)));
const includeStockfishShallow = boolArg('--include-stockfish-shallow', false);
const sfShallowDepths = arg('--stockfish-shallow-depths', '1,2,3').split(',').filter(Boolean).map(Number);
const includeStockfishLiteShallow = boolArg('--include-stockfish-lite-shallow', false);
const sfLiteShallowDepths = arg('--stockfish-lite-shallow-depths', sfShallowDepths.join(',')).split(',').filter(Boolean).map(Number);

async function loadEvaluator(onnx, metaPath) {
  const meta = JSON.parse(readFileSync(metaPath, 'utf8'));
  return (meta.kind === 'squareformer' || meta.kind === 'squareformer_v2') ? SquareFormerEvaluator.create(onnx, meta) : OnnxEvaluator.create(onnx, meta);
}
const allOpenings = loadOpenings(openingsFile);
const openingCount = openingCountRaw === '' ? allOpenings.length - openingStart : Number(openingCountRaw);
const openings = allOpenings.slice(openingStart, openingStart + openingCount);
if (!openings.length) throw new Error(`No openings selected: start=${openingStart} count=${openingCount} total=${allOpenings.length}`);
const candidate = { type:'tiny', ...candidateSpec, evaluator: await loadEvaluator(candidateSpec.onnx, candidateSpec.meta) };
const anchors = [
  ...(includeStockfish ? stockfishAnchors(stockfishPath, sfLevels, sfNodes, threads, hash) : []),
  ...(includeStockfishLite ? stockfishLiteAnchors(stockfishLitePath, sfLiteLevels, sfLiteNodes, hash) : []),
  ...(includeStockfishShallow ? stockfishShallowAnchors(stockfishPath, sfShallowDepths, threads, hash) : []),
  ...(includeStockfishLiteShallow ? stockfishLiteShallowAnchors(stockfishLitePath, sfLiteShallowDepths, hash) : []),
  ...customUciAnchors(arg('--uci-anchors',''), sfNodes),
];
if (!anchors.length) throw new Error('No anchors configured');
for (const a of anchors) { a.engine = new UciEngine(a.name, a.command); await a.engine.init(a.options); }
async function choose(side, board, history) {
  if (side.type === 'tiny') {
    const legalUci = new Set(legalMoves(board).map(moveToUci));
    const searchPolicy = side.mode === 'aux' ? auxPuctPolicy : (side.mode === 'av' ? actionValuePuctPolicy : classicPuctPolicy);
    const r = await chooseMove(board, side.evaluator, {
      visits,
      cpuct,
      batchSize,
      historyFens: history.slice(-2).reverse(),
      searchPolicy,
      avWeight: side.avWeight,
      rankWeight: side.rankWeight,
      regretWeight: side.regretWeight,
      riskWeight: side.riskWeight,
      uncertaintyWeight: side.uncertaintyWeight,
    });
    if (!r.move) return null;
    const u = moveToUci(r.move);
    return legalUci.has(u) ? u : `ILLEGAL:${u}`;
  }
  return side.engine.bestmove(boardToFen(board), side.search || { go:`nodes ${side.nodes || 1}`, nodes:side.nodes || 1 });
}
function applyUci(board, uci) {
  for (const m of legalMoves(board)) if (moveToUci(m) === uci) return makeMove(board, m);
  return null;
}
async function playGame(anchor, tinyColor, opening, gameIndex) {
  let board = parseFen(opening), whiteScore = terminalWhiteScore(board), plies = 0, illegal = null;
  const history = [], moves = [];
  for (; whiteScore === null && plies < maxPlies; plies++) {
    const tinyToMove = board.turn === tinyColor;
    const side = tinyToMove ? candidate : anchor;
    const uci = await choose(side, board, history);
    if (!uci || uci.startsWith('ILLEGAL:')) { illegal = side.name; whiteScore = board.turn === 'w' ? 0 : 1; break; }
    const next = applyUci(board, uci);
    if (!next) { illegal = side.name; whiteScore = board.turn === 'w' ? 0 : 1; break; }
    moves.push({ ply: plies + 1, side: board.turn, engine: side.name, uci, fenBefore: boardToFen(board) });
    history.push(boardToFen(board)); board = next; whiteScore = terminalWhiteScore(board);
  }
  if (whiteScore === null) whiteScore = 0.5;
  const tinyScore = whiteScore === 0.5 ? 0.5 : ((whiteScore === 1) === (tinyColor === 'w') ? 1 : 0);
  process.stderr.write(`[anchor-arena] ${candidate.name} vs ${anchor.name} game=${gameIndex} tinyScore=${tinyScore} plies=${plies}${illegal ? ' illegal='+illegal : ''}\n`);
  return { anchor: anchor.name, white: tinyColor === 'w' ? candidate.name : anchor.name, black: tinyColor === 'w' ? anchor.name : candidate.name, opening, whiteScore, tinyScore, plies, finalFen: boardToFen(board), illegal, moves };
}
const games = [], summaries = [];
for (const anchor of anchors) {
  const pairScores = [];
  for (let p = 0; p < pairs; p++) {
    const opening = openings[p % openings.length];
    const g1 = await playGame(anchor, 'w', opening, p*2+1);
    const g2 = await playGame(anchor, 'b', opening, p*2+2);
    games.push(g1, g2); pairScores.push(g1.tinyScore + g2.tinyScore);
  }
  const ag = games.filter(g => g.anchor === anchor.name);
  const score = ag.reduce((s,g)=>s+g.tinyScore,0), n = ag.length;
  const wins = ag.filter(g=>g.tinyScore===1).length, draws = ag.filter(g=>g.tinyScore===0.5).length, losses = ag.filter(g=>g.tinyScore===0).length;
  const scoreRate = score / Math.max(1,n);
  summaries.push({ anchor: anchor.name, games:n, pairs, wins, draws, losses, scoreRate, eloDiff: elo(scoreRate), eloCi95: normalApproxCi(scoreRate,n), ptnml: ptnmlFromPairScores(pairScores), illegal: ag.filter(g=>g.illegal).length });
}
for (const a of anchors) a.engine.quit();
mkdirSync(dirname(out), { recursive:true });
const protocol = { kind:'uci_anchor_arena', candidate:{ name:candidate.name, onnx:candidate.onnx, meta:candidate.meta, mode:candidate.mode, avWeight:candidate.avWeight, rankWeight:candidate.rankWeight, regretWeight:candidate.regretWeight, riskWeight:candidate.riskWeight, uncertaintyWeight:candidate.uncertaintyWeight }, anchors:anchors.map(a=>({ name:a.name, type:a.type, command:a.command, search:a.search, nodes:a.nodes, options:a.options })), openingsFile, openingsTotal:allOpenings.length, openingStart, openingCount:openings.length, openings:openings.length, pairs, visits, cpuct, batchSize, maxPlies, stockfishNodes:sfNodes, stockfishLiteNodes:sfLiteNodes, includeStockfishLite, includeStockfishShallow, stockfishShallowDepths:sfShallowDepths, includeStockfishLiteShallow, stockfishLiteShallowDepths:sfLiteShallowDepths, threads, hash, createdUtc:new Date().toISOString() };
writeFileSync(out, JSON.stringify({ candidate:{ name:candidate.name, onnx:candidate.onnx, meta:candidate.meta }, protocol, summaries, games }, null, 2));
writeFileSync(`${out}.protocol.json`, JSON.stringify(protocol, null, 2));
for (const s of summaries) {
  console.log(`METRIC anchor_${s.anchor}_score_rate=${s.scoreRate.toFixed(6)}`);
  console.log(`METRIC anchor_${s.anchor}_elo_diff=${s.eloDiff.toFixed(3)}`);
  console.log(`METRIC anchor_${s.anchor}_elo_ci95=${s.eloCi95.toFixed(3)}`);
  console.log(`METRIC anchor_${s.anchor}_wdl=${s.wins}_${s.draws}_${s.losses}`);
  console.log(`METRIC anchor_${s.anchor}_illegal=${s.illegal}`);
}
console.log(`METRIC anchor_games=${games.length}`);
