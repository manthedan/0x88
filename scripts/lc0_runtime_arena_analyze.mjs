#!/usr/bin/env node
import { readdir, stat, writeFile } from 'node:fs/promises';
import { basename, join } from 'node:path';

function usage() {
  console.log(`Usage: node scripts/lc0_runtime_arena_analyze.mjs <report.json|report-dir>... [options]\n\nBuilds normalized LC0 browser runtime arena summaries from report JSON files.\n\nOptions:\n  --out PATH              Write full analysis JSON\n  --summary-tsv PATH      Write normalized summary TSV\n  --divergence-tsv PATH   Write LC0 same-position move divergence TSV\n  -h, --help              Show this help\n`);
}

function parseArgs(argv) {
  const args = { inputs: [], out: '', summaryTsv: '', divergenceTsv: '', help: false };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    const next = () => {
      if (i + 1 >= argv.length) throw new Error(`${arg} requires a value`);
      return argv[++i];
    };
    if (arg === '--out') args.out = next();
    else if (arg === '--summary-tsv') args.summaryTsv = next();
    else if (arg === '--divergence-tsv') args.divergenceTsv = next();
    else if (arg === '-h' || arg === '--help') args.help = true;
    else args.inputs.push(arg);
  }
  if (!args.help && args.inputs.length === 0) throw new Error('expected at least one report JSON file or directory');
  return args;
}

async function expandInputs(inputs) {
  const files = [];
  for (const input of inputs) {
    const info = await stat(input);
    if (info.isDirectory()) {
      const names = await readdir(input);
      for (const name of names) {
        if (/\.json$/i.test(name)) files.push(join(input, name));
      }
    } else {
      files.push(input);
    }
  }
  return [...new Set(files)].sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
}

function splitPgnGames(pgn) {
  return String(pgn ?? '')
    .split(/\n\s*\n(?=\[Event\s+")/g)
    .map((game) => game.trim())
    .filter(Boolean);
}

function parseTags(game) {
  const tags = {};
  for (const match of game.matchAll(/^\[([^\s]+)\s+"([^"]*)"\]$/gm)) tags[match[1]] = match[2];
  return tags;
}

function stripVariations(text) {
  let previous;
  do {
    previous = text;
    text = text.replace(/\([^()]*\)/g, ' ');
  } while (text !== previous);
  return text;
}

function moveTokens(game) {
  let text = game
    .replace(/^\[[^\n]*\]\s*$/gm, ' ')
    .replace(/\{[^}]*\}/g, ' ')
    .replace(/;[^\n]*/g, ' ')
    .replace(/\$\d+/g, ' ');
  text = stripVariations(text);
  return text
    .replace(/\d+\.(\.\.)?/g, ' ')
    .split(/\s+/)
    .map((token) => token.trim())
    .filter((token) => token && !['1-0', '0-1', '1/2-1/2', '*'].includes(token));
}

function pgnPlyStats(pgn) {
  const games = splitPgnGames(pgn).map((game, index) => {
    const tags = parseTags(game);
    const moves = moveTokens(game);
    const byColor = { white: 0, black: 0 };
    const byEngine = {};
    moves.forEach((_move, plyIndex) => {
      const color = plyIndex % 2 === 0 ? 'white' : 'black';
      byColor[color] += 1;
      const name = tags[color === 'white' ? 'White' : 'Black'] || color;
      byEngine[name] = (byEngine[name] ?? 0) + 1;
    });
    return { index: index + 1, tags, plies: moves.length, whitePlies: byColor.white, blackPlies: byColor.black, byEngine };
  });
  const totalPlies = games.reduce((sum, game) => sum + game.plies, 0);
  const byEngine = {};
  for (const game of games) for (const [name, count] of Object.entries(game.byEngine)) byEngine[name] = (byEngine[name] ?? 0) + count;
  const lc0Plies = Object.entries(byEngine).filter(([name]) => /\b(lc0|leela)\b/i.test(name)).reduce((sum, [, count]) => sum + count, 0);
  const opponentPlies = totalPlies - lc0Plies;
  return { games, totalPlies, lc0Plies, opponentPlies, byEngine };
}

function firstValue(record) {
  const values = Object.values(record ?? {});
  return values.length ? values[0] : undefined;
}

function safeDiv(n, d) {
  return Number.isFinite(n) && Number.isFinite(d) && d !== 0 ? n / d : null;
}

function rounded(value, digits = 3) {
  if (value == null || !Number.isFinite(value)) return null;
  const scale = 10 ** digits;
  return Math.round(value * scale) / scale;
}

function parseStockfishWhiteCp(output) {
  if (typeof output?.whiteCp === 'number') return output.whiteCp;
  if (typeof output?.mateInWhitePov === 'number') return null;
  const text = `${output?.shortEval ?? ''} ${output?.summary ?? ''}`;
  const mate = text.match(/\bM([+-]?\d+)\b|\bmate\s+([+-]?\d+)\b/i);
  if (mate) return null;
  const cp = text.match(/([+-]\d+(?:\.\d+)?)(?=\s*(?:·|$))/);
  return cp ? Math.round(Number(cp[1]) * 100) : null;
}

function fenTurnFromKeyOrFen(fen) {
  return String(fen ?? '').split(/\s+/)[1] || null;
}

function lc0MoveSfEvaluations(result) {
  const outputs = result.engineOutputs ?? [];
  const rows = [];
  for (let i = 0; i < outputs.length; i++) {
    const lc0 = outputs[i];
    if (lc0?.kind !== 'lc0' || !lc0.fen || !lc0.move) continue;
    const sf = outputs[i + 1];
    if (sf?.kind !== 'uci') continue;
    const whiteCp = parseStockfishWhiteCp(sf);
    if (whiteCp == null) continue;
    const turn = fenTurnFromKeyOrFen(lc0.fen);
    const lc0PerspectiveCp = turn === 'b' ? -whiteCp : whiteCp;
    rows.push({
      runtime: result.runtime,
      preFen: lc0.fen,
      positionKey: positionKey(lc0.fen),
      lc0Move: lc0.move,
      lc0Pv: lc0.pv,
      sfReply: sf.move,
      stockfishWhiteCpAfterLc0Move: whiteCp,
      stockfishLc0PerspectiveCpAfterMove: lc0PerspectiveCp,
      stockfishOutput: { shortEval: sf.shortEval, summary: sf.summary, pv: sf.pv },
      lc0Output: { summary: lc0.summary, detail: lc0.detail, pv: lc0.pv },
    });
  }
  return rows;
}

function stockfishMoveEvalSummary(result) {
  const evaluations = lc0MoveSfEvaluations(result);
  const cps = evaluations.map((row) => row.stockfishLc0PerspectiveCpAfterMove).filter(Number.isFinite);
  return {
    evaluatedLc0Moves: cps.length,
    avgStockfishLc0PerspectiveCpAfterMove: rounded(cps.reduce((sum, cp) => sum + cp, 0) / cps.length, 3),
    medianStockfishLc0PerspectiveCpAfterMove: rounded(median(cps), 3),
  };
}

function median(values) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function scoreText(matchScore) {
  return String(matchScore ?? '').match(/\s(\d+|\d+½|½)\s+–\s+(\d+|\d+½|½)\s/)?.[0]?.trim() ?? String(matchScore ?? '');
}

function positionKey(fen) {
  return String(fen ?? '').split(/\s+/).slice(0, 4).join(' ');
}

function summarizeResult(report, result, file) {
  const plyStats = pgnPlyStats(result.pgn);
  const lc0 = firstValue(result.telemetry?.lc0Tree) ?? {};
  const uci = firstValue(result.telemetry?.uci) ?? {};
  const lc0SearchSeconds = safeDiv(lc0.totalElapsedMs, 1000);
  const uciSearchSeconds = safeDiv(uci.totalElapsedMs, 1000);
  const elapsedSeconds = safeDiv(result.elapsedMs, 1000);
  const lc0Plies = plyStats.lc0Plies || lc0.searches || null;
  const opponentPlies = plyStats.opponentPlies || uci.searches || null;
  const sfEval = stockfishMoveEvalSummary(result);
  return {
    file,
    movetimeMs: report.config?.movetimeMs ?? result.configuration?.movetimeMs,
    runtime: result.runtime,
    runtimeLabel: result.runtimeLabel,
    score: scoreText(result.summary?.matchScore),
    elapsedSeconds: rounded(elapsedSeconds, 3),
    plies: plyStats,
    lc0: {
      searches: lc0.searches ?? null,
      completedVisits: lc0.completedVisits ?? null,
      evals: lc0.evalCalls ?? null,
      cacheHits: lc0.cacheHits ?? null,
      avgSearchMs: rounded(safeDiv(lc0.totalElapsedMs, lc0.searches), 3),
      searchesPerMatchPly: rounded(safeDiv(lc0.searches, plyStats.totalPlies), 4),
      searchesPerLc0Ply: rounded(safeDiv(lc0.searches, lc0Plies), 4),
      visitsPerLc0Ply: rounded(safeDiv(lc0.completedVisits, lc0Plies), 3),
      evalsPerLc0Ply: rounded(safeDiv(lc0.evalCalls, lc0Plies), 3),
      cacheHitsPerLc0Ply: rounded(safeDiv(lc0.cacheHits, lc0Plies), 3),
      searchesPerSecond: rounded(safeDiv(lc0.searches, lc0SearchSeconds), 3),
      visitsPerSecond: rounded(safeDiv(lc0.completedVisits, lc0SearchSeconds), 3),
      evalsPerSecond: rounded(safeDiv(lc0.evalCalls, lc0SearchSeconds), 3),
      backendTotalEvalMs: rounded(lc0.lastBackendTimingPerPositionMeans?.totalEvalMs, 3),
      backendReadbackMs: rounded(lc0.lastBackendTimingPerPositionMeans?.readbackSyncedMs, 3),
      ...sfEval,
    },
    opponent: {
      searches: uci.searches ?? null,
      searchesPerOpponentPly: rounded(safeDiv(uci.searches, opponentPlies), 4),
      nodes: uci.totalNodes ?? null,
      nodesPerOpponentPly: rounded(safeDiv(uci.totalNodes, opponentPlies), 3),
      nodesPerSecond: rounded(safeDiv(uci.totalNodes, uciSearchSeconds), 3),
    },
  };
}

function collectLc0Decisions(reports) {
  const byMovetime = new Map();
  for (const entry of reports) {
    const movetimeMs = entry.report.config?.movetimeMs;
    if (!byMovetime.has(movetimeMs)) byMovetime.set(movetimeMs, new Map());
    const positions = byMovetime.get(movetimeMs);
    for (const result of entry.report.results ?? []) {
      for (const output of result.engineOutputs ?? []) {
        if (output?.kind !== 'lc0' || !output.fen || !output.move) continue;
        const key = positionKey(output.fen);
        if (!key) continue;
        if (!positions.has(key)) positions.set(key, { fen: output.fen, byRuntime: {} });
        const position = positions.get(key);
        if (!position.byRuntime[result.runtime]) position.byRuntime[result.runtime] = [];
        position.byRuntime[result.runtime].push({ move: output.move, pv: output.pv, summary: output.summary, detail: output.detail });
      }
    }
  }

  const groups = [];
  for (const [movetimeMs, positions] of byMovetime.entries()) {
    let sharedPositions = 0;
    let divergentPositions = 0;
    const examples = [];
    const pairStats = new Map();
    for (const position of positions.values()) {
      const runtimeMoves = Object.fromEntries(Object.entries(position.byRuntime).map(([runtime, outputs]) => [runtime, outputs.map((o) => o.move)]));
      const runtimes = Object.keys(runtimeMoves).filter((runtime) => runtimeMoves[runtime].length > 0);
      if (runtimes.length < 2) continue;
      sharedPositions += 1;
      const firstMoves = Object.fromEntries(runtimes.map((runtime) => [runtime, runtimeMoves[runtime][0]]));
      const distinct = new Set(Object.values(firstMoves));
      for (let i = 0; i < runtimes.length; i++) {
        for (let j = i + 1; j < runtimes.length; j++) {
          const pair = [runtimes[i], runtimes[j]].sort().join(' vs ');
          const stat = pairStats.get(pair) ?? { shared: 0, same: 0, different: 0 };
          stat.shared += 1;
          if (firstMoves[runtimes[i]] === firstMoves[runtimes[j]]) stat.same += 1;
          else stat.different += 1;
          pairStats.set(pair, stat);
        }
      }
      if (distinct.size > 1) {
        divergentPositions += 1;
        if (examples.length < 50) examples.push({ fen: position.fen, movesByRuntime: firstMoves, outputsByRuntime: position.byRuntime });
      }
    }
    groups.push({
      movetimeMs,
      loggedLc0Positions: positions.size,
      sharedPositions,
      divergentPositions,
      divergenceRate: rounded(safeDiv(divergentPositions, sharedPositions), 4),
      pairAgreement: Object.fromEntries([...pairStats.entries()].map(([pair, stat]) => [pair, { ...stat, agreementRate: rounded(safeDiv(stat.same, stat.shared), 4) }])),
      examples,
    });
  }
  return groups.sort((a, b) => Number(a.movetimeMs) - Number(b.movetimeMs));
}

function collectPgnFirstDivergences(reports) {
  const byMovetime = new Map();
  for (const entry of reports) {
    const movetimeMs = entry.report.config?.movetimeMs;
    if (!byMovetime.has(movetimeMs)) byMovetime.set(movetimeMs, []);
    byMovetime.get(movetimeMs).push(...(entry.report.results ?? []).map((result) => ({
      runtime: result.runtime,
      games: splitPgnGames(result.pgn).map(moveTokens),
    })));
  }
  const groups = [];
  for (const [movetimeMs, results] of byMovetime.entries()) {
    const gameCount = Math.max(0, ...results.map((result) => result.games.length));
    const games = [];
    for (let gameIndex = 0; gameIndex < gameCount; gameIndex++) {
      const runtimeMoves = Object.fromEntries(results.filter((result) => result.games[gameIndex]).map((result) => [result.runtime, result.games[gameIndex]]));
      const runtimes = Object.keys(runtimeMoves);
      const minLen = Math.min(...Object.values(runtimeMoves).map((moves) => moves.length));
      let divergence = null;
      for (let plyIndex = 0; plyIndex < minLen; plyIndex++) {
        const moves = Object.fromEntries(runtimes.map((runtime) => [runtime, runtimeMoves[runtime][plyIndex]]));
        if (new Set(Object.values(moves)).size > 1) {
          divergence = {
            ply: plyIndex + 1,
            moveNumber: Math.floor(plyIndex / 2) + 1,
            side: plyIndex % 2 === 0 ? 'white' : 'black',
            sanByRuntime: moves,
            previousMoves: runtimeMoves[runtimes[0]].slice(Math.max(0, plyIndex - 8), plyIndex),
          };
          break;
        }
      }
      if (!divergence && new Set(Object.values(runtimeMoves).map((moves) => moves.length)).size > 1) {
        divergence = { ply: minLen + 1, moveNumber: Math.floor(minLen / 2) + 1, side: minLen % 2 === 0 ? 'white' : 'black', sanByRuntime: Object.fromEntries(runtimes.map((runtime) => [runtime, runtimeMoves[runtime][minLen] ?? '<game ended>'])), previousMoves: runtimeMoves[runtimes[0]].slice(Math.max(0, minLen - 8), minLen) };
      }
      games.push({ game: gameIndex + 1, runtimes, firstDivergence: divergence });
    }
    groups.push({ movetimeMs, games });
  }
  return groups.sort((a, b) => Number(a.movetimeMs) - Number(b.movetimeMs));
}

function collectStockfishCpLoss(reports) {
  const byMovetime = new Map();
  for (const entry of reports) {
    const movetimeMs = entry.report.config?.movetimeMs;
    if (!byMovetime.has(movetimeMs)) byMovetime.set(movetimeMs, new Map());
    const positions = byMovetime.get(movetimeMs);
    for (const result of entry.report.results ?? []) {
      for (const evaluation of lc0MoveSfEvaluations(result)) {
        if (!positions.has(evaluation.positionKey)) positions.set(evaluation.positionKey, { fen: evaluation.preFen, byRuntime: {} });
        const position = positions.get(evaluation.positionKey);
        if (!position.byRuntime[result.runtime]) position.byRuntime[result.runtime] = [];
        position.byRuntime[result.runtime].push(evaluation);
      }
    }
  }

  const groups = [];
  for (const [movetimeMs, positions] of byMovetime.entries()) {
    const byRuntime = new Map();
    const examples = [];
    let sharedEvaluatedPositions = 0;
    for (const position of positions.values()) {
      const entries = Object.entries(position.byRuntime).map(([runtime, rows]) => [runtime, rows[0]]).filter(([, row]) => row);
      if (entries.length < 2) continue;
      sharedEvaluatedPositions += 1;
      const bestCp = Math.max(...entries.map(([, row]) => row.stockfishLc0PerspectiveCpAfterMove));
      const moves = Object.fromEntries(entries.map(([runtime, row]) => [runtime, row.lc0Move]));
      const cps = Object.fromEntries(entries.map(([runtime, row]) => [runtime, row.stockfishLc0PerspectiveCpAfterMove]));
      const losses = Object.fromEntries(entries.map(([runtime, row]) => [runtime, bestCp - row.stockfishLc0PerspectiveCpAfterMove]));
      for (const [runtime, row] of entries) {
        const stat = byRuntime.get(runtime) ?? { positions: 0, losses: [], cps: [], divergentMovePositions: 0 };
        stat.positions += 1;
        stat.losses.push(bestCp - row.stockfishLc0PerspectiveCpAfterMove);
        stat.cps.push(row.stockfishLc0PerspectiveCpAfterMove);
        if (new Set(Object.values(moves)).size > 1) stat.divergentMovePositions += 1;
        byRuntime.set(runtime, stat);
      }
      if ((new Set(Object.values(moves)).size > 1 || Math.max(...Object.values(losses)) > 0) && examples.length < 50) {
        examples.push({ fen: position.fen, movesByRuntime: moves, lc0PerspectiveCpAfterMoveByRuntime: cps, relativeCpLossByRuntime: losses });
      }
    }
    groups.push({
      movetimeMs,
      sharedEvaluatedPositions,
      byRuntime: Object.fromEntries([...byRuntime.entries()].map(([runtime, stat]) => [runtime, {
        positions: stat.positions,
        avgRelativeCpLoss: rounded(stat.losses.reduce((sum, value) => sum + value, 0) / stat.losses.length, 3),
        medianRelativeCpLoss: rounded(median(stat.losses), 3),
        avgStockfishLc0PerspectiveCpAfterMove: rounded(stat.cps.reduce((sum, value) => sum + value, 0) / stat.cps.length, 3),
        divergentMovePositions: stat.divergentMovePositions,
      }])),
      examples,
    });
  }
  return groups.sort((a, b) => Number(a.movetimeMs) - Number(b.movetimeMs));
}

function summaryTsv(analysis) {
  const lossByMsRuntime = new Map();
  for (const group of analysis.stockfishCpLoss) for (const [runtime, stat] of Object.entries(group.byRuntime)) lossByMsRuntime.set(`${group.movetimeMs}\t${runtime}`, stat);
  const header = ['ms', 'runtime', 'score', 'games', 'plies', 'lc0_plies', 'elapsed_s', 'lc0_searches', 'lc0_searches_per_s', 'lc0_visits_per_s', 'lc0_evals_per_s', 'visits_per_lc0_ply', 'evals_per_lc0_ply', 'sf_eval_lc0_moves', 'avg_sf_cp_after_lc0_move', 'avg_relative_sf_cp_loss', 'sf_nodes_per_s'].join('\t');
  const rows = analysis.normalized.map((row) => {
    const loss = lossByMsRuntime.get(`${row.movetimeMs}\t${row.runtime}`) ?? {};
    return [
      row.movetimeMs,
      row.runtime,
      row.score,
      row.plies.games.length,
      row.plies.totalPlies,
      row.plies.lc0Plies,
      row.elapsedSeconds,
      row.lc0.searches,
      row.lc0.searchesPerSecond,
      row.lc0.visitsPerSecond,
      row.lc0.evalsPerSecond,
      row.lc0.visitsPerLc0Ply,
      row.lc0.evalsPerLc0Ply,
      row.lc0.evaluatedLc0Moves,
      row.lc0.avgStockfishLc0PerspectiveCpAfterMove,
      loss.avgRelativeCpLoss ?? '',
      row.opponent.nodesPerSecond,
    ].join('\t');
  });
  return `${header}\n${rows.join('\n')}\n`;
}

function divergenceTsv(analysis) {
  const header = ['ms', 'fen', 'moves_by_runtime'].join('\t');
  const rows = [];
  for (const group of analysis.decisionDivergence) {
    for (const example of group.examples) rows.push([group.movetimeMs, example.fen, Object.entries(example.movesByRuntime).map(([runtime, move]) => `${runtime}:${move}`).join(',')].join('\t'));
  }
  return `${header}\n${rows.join('\n')}\n`;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) return usage();
  const files = await expandInputs(args.inputs);
  const reports = [];
  for (const file of files) {
    const report = JSON.parse(await BunLikeReadFile(file));
    if (report.status !== 'LC0_RUNTIME_ARENA_BENCH_DONE') continue;
    reports.push({ file, report });
  }
  const normalized = reports.flatMap(({ file, report }) => (report.results ?? []).map((result) => summarizeResult(report, result, file)))
    .sort((a, b) => Number(a.movetimeMs) - Number(b.movetimeMs) || String(a.runtime).localeCompare(String(b.runtime)));
  const analysis = {
    status: 'LC0_RUNTIME_ARENA_ANALYSIS_DONE',
    generatedAt: new Date().toISOString(),
    files: reports.map((entry) => entry.file),
    normalized,
    decisionDivergence: collectLc0Decisions(reports),
    pgnFirstDivergence: collectPgnFirstDivergences(reports),
    stockfishCpLoss: collectStockfishCpLoss(reports),
    notes: [
      'Ply counts are parsed from saved PGN, so they include complete games even when engineOutputs were truncated.',
      'LC0 rates use LC0 search wall time (telemetry.lc0Tree.totalElapsedMs), not whole-match elapsed time.',
      'Stockfish cp-after-LC0-move is parsed from the opponent UCI output immediately following each saved LC0 output; cp is converted to LC0 perspective, so higher is better for LC0.',
      'Relative Stockfish cp loss compares saved runtimes that reached the same LC0 pre-move FEN; loss is best LC0-perspective cp among those runtimes minus this runtime cp.',
      'Same-position move comparisons use saved LC0 engineOutputs keyed by FEN fields 1-4; older reports may only include the last 200 outputs per runtime.',
    ],
  };
  if (args.out) await writeFile(args.out, `${JSON.stringify(analysis, null, 2)}\n`);
  if (args.summaryTsv) await writeFile(args.summaryTsv, summaryTsv(analysis));
  if (args.divergenceTsv) await writeFile(args.divergenceTsv, divergenceTsv(analysis));
  console.log(summaryTsv(analysis));
  for (const group of analysis.decisionDivergence) {
    console.log(`movetime ${group.movetimeMs}ms: ${group.divergentPositions}/${group.sharedPositions} shared logged LC0 positions had different selected moves`);
  }
  for (const group of analysis.pgnFirstDivergence) {
    for (const game of group.games) {
      const d = game.firstDivergence;
      if (!d) console.log(`movetime ${group.movetimeMs}ms game ${game.game}: no PGN divergence across runtimes`);
      else console.log(`movetime ${group.movetimeMs}ms game ${game.game}: first PGN divergence at ply ${d.ply} (${d.moveNumber}${d.side === 'black' ? '...' : '.'}) ${Object.entries(d.sanByRuntime).map(([runtime, san]) => `${runtime}:${san}`).join(' | ')}`);
    }
  }
  for (const group of analysis.stockfishCpLoss) {
    const parts = Object.entries(group.byRuntime).map(([runtime, stat]) => `${runtime} avgLoss=${stat.avgRelativeCpLoss}cp n=${stat.positions}`);
    console.log(`movetime ${group.movetimeMs}ms SF relative cp loss: ${parts.join(' | ') || 'no shared evaluated positions'}`);
  }
}

async function BunLikeReadFile(file) {
  const { readFile } = await import('node:fs/promises');
  return readFile(file, 'utf8');
}

main().catch((error) => {
  console.error(error.stack ?? error.message);
  process.exit(1);
});
