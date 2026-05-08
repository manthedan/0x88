#!/usr/bin/env node
import { readFileSync } from 'node:fs';
import { performance } from 'node:perf_hooks';
import { parseFen } from '../src/chess/board.ts';
import { makeMove, legalMoves } from '../src/chess/movegen.ts';
import { OnnxEvaluator } from '../src/nn/onnxEvaluator.ts';
import { SquareFormerEvaluator } from '../src/nn/squareformerEvaluator.ts';
import { searchRoot, classicPuctPolicy, actionValuePuctPolicy, auxPuctPolicy } from '../src/search/puct.ts';

function arg(name, fallback='') { const p=`${name}=`; const x=process.argv.find(v=>v.startsWith(p)); if (x) return x.slice(p.length); const i=process.argv.indexOf(name); return i>=0 ? process.argv[i+1] : fallback; }
async function load(model, metaPath) { const meta=JSON.parse(readFileSync(metaPath,'utf8')); return meta.kind==='squareformer' ? SquareFormerEvaluator.create(model, meta) : OnnxEvaluator.create(model, meta); }
function median(xs){const a=[...xs].sort((x,y)=>x-y); return a[Math.floor(a.length/2)] ?? 0;}
function p90(xs){const a=[...xs].sort((x,y)=>x-y); return a[Math.min(a.length-1,Math.floor(a.length*0.9))] ?? 0;}
function sampleBoards(rootFen, n){
  const out=[parseFen(rootFen)];
  for(let i=1;i<n;i++){
    const prev=out[out.length-1];
    const moves=legalMoves(prev);
    if(!moves.length){ out.push(parseFen(rootFen)); continue; }
    out.push(makeMove(prev, moves[(i*7)%moves.length]));
  }
  return out;
}

const model=arg('--model'); const meta=arg('--meta');
const fen=arg('--fen','rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1');
const visitsList=arg('--visits','8,16,32,64').split(',').filter(Boolean).map(Number);
const batchList=arg('--batches','1,2,4,8,16').split(',').filter(Boolean).map(Number);
const repeats=Number(arg('--repeats','5')); const positions=Number(arg('--positions','3')); const cpuct=Number(arg('--cpuct','1.5'));
const mode=arg('--mode','puct');
const avWeight=Number(arg('--av-weight','0.25'));
const rankWeight=Number(arg('--rank-weight','0'));
const regretWeight=Number(arg('--regret-weight','0'));
const riskWeight=Number(arg('--risk-weight','0'));
const uncertaintyWeight=Number(arg('--uncertainty-weight','0'));
if (!['puct','av','aux'].includes(mode)) throw new Error(`bad --mode ${mode}`);
const searchPolicy = mode === 'aux' ? auxPuctPolicy : (mode === 'av' ? actionValuePuctPolicy : classicPuctPolicy);
if (!model || !meta) throw new Error('usage: node --experimental-strip-types eval/puct_batch_benchmark.mjs --model m.onnx --meta m.meta.json');
const evaluator=await load(model, meta); const boards=sampleBoards(fen, positions);
// warmup
await searchRoot(boards[0], evaluator, { visits: 2, batchSize: 1, cpuct, temperature: 0, searchPolicy, avWeight, rankWeight, regretWeight, riskWeight, uncertaintyWeight });
for (const visits of visitsList) {
  for (const batchSize of batchList) {
    if (batchSize > visits && batchSize !== 1) continue;
    const times=[];
    for (let r=0;r<repeats;r++) {
      const t0=performance.now();
      for (const board of boards) await searchRoot(board, evaluator, { visits, batchSize, cpuct, temperature: 0, searchPolicy, avWeight, rankWeight, regretWeight, riskWeight, uncertaintyWeight });
      times.push((performance.now()-t0)/boards.length);
    }
    const med=median(times); const q90=p90(times); const nps=visits/(med/1000);
    console.log(`RESULT visits=${visits} batch=${batchSize} positions=${positions} repeats=${repeats} median_ms=${med.toFixed(3)} p90_ms=${q90.toFixed(3)} visits_per_s=${nps.toFixed(1)}`);
    console.log(`METRIC v${visits}_b${batchSize}_median_ms=${med.toFixed(3)}`);
  }
}
