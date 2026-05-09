#!/usr/bin/env node
import { readFileSync } from 'node:fs';
import { parseFen } from '../src/chess/board.ts';
import { legalMoves } from '../src/chess/movegen.ts';
import { moveToActionId, moveToUci } from '../src/chess/moveCodec.ts';
import { OnnxEvaluator } from '../src/nn/onnxEvaluator.ts';
import { SquareFormerEvaluator } from '../src/nn/squareformerEvaluator.ts';
import { searchRoot } from '../src/search/puct.ts';

function arg(name, fallback='') { const p=`${name}=`; const x=process.argv.find(v=>v.startsWith(p)); if (x) return x.slice(p.length); const i=process.argv.indexOf(name); return i>=0 ? process.argv[i+1] : fallback; }
async function load(model, metaPath) { const meta=JSON.parse(readFileSync(metaPath,'utf8')); return (meta.kind==='squareformer'||meta.kind==='squareformer_v2') ? SquareFormerEvaluator.create(model, meta) : OnnxEvaluator.create(model, meta); }
function valueFromWdl(wdl) { return wdl[0]-wdl[2]; }
const model=arg('--model'); const meta=arg('--meta'); const fen=arg('--fen','rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'); const visitsList=arg('--visits','1,2,4,8,16,32').split(',').map(Number); const cpuct=Number(arg('--cpuct','1.5')); const batchSize=Number(arg('--batch-size','1')); const topN=Number(arg('--top','12'));
if (!model || !meta) throw new Error('usage: node --experimental-strip-types eval/puct_probe.mjs --model m.onnx --meta m.meta.json [--fen FEN]');
const evaluator=await load(model, meta); const board=parseFen(fen); const ev=await evaluator.evaluate(board); const legal=legalMoves(board);
const policy=legal.map(m=>({uci:moveToUci(m), p:ev.policy.get(moveToActionId(m))??0})).sort((a,b)=>b.p-a.p);
const z=policy.reduce((s,x)=>s+x.p,0);
console.log(`fen ${fen}`); console.log(`wdl=${ev.wdl.map(x=>x.toFixed(4)).join(',')} value_stm=${valueFromWdl(ev.wdl).toFixed(4)} legal=${legal.length} policy_sum_legal=${z.toFixed(6)}`);
console.log('POLICY_TOP'); for (const x of policy.slice(0,topN)) console.log(`${x.uci}\tP=${x.p.toFixed(6)}`);
for (const v of visitsList) {
  const r=await searchRoot(board,evaluator,{visits:v,cpuct,batchSize,temperature:0});
  const rows=r.policy.slice().sort((a,b)=>b.visits-a.visits || b.prior-a.prior).slice(0,topN).map(e=>({uci:moveToUci(e.move),N:e.visits,P:e.prior,Q:e.q,prob:e.probability}));
  console.log(`PUCT visits=${v} selected=${r.move?moveToUci(r.move):'null'} value=${r.value.toFixed(4)} totalN=${r.visits}`);
  for (const x of rows) console.log(`${x.uci}\tN=${x.N}\tP=${x.P.toFixed(6)}\tQ=${x.Q.toFixed(4)}\tprob=${x.prob.toFixed(3)}`);
}
