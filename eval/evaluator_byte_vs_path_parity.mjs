#!/usr/bin/env node
import { readFileSync } from 'node:fs';
import { parseFen } from '../src/chess/board.ts';
import { legalMoves } from '../src/chess/movegen.ts';
import { moveToActionId } from '../src/chess/moveCodec.ts';
import { OnnxEvaluator } from '../src/nn/onnxEvaluator.ts';
import { SquareFormerEvaluator } from '../src/nn/squareformerEvaluator.ts';

function arg(name, fallback=''){ const p=`${name}=`; const x=process.argv.find(v=>v.startsWith(p)); if(x)return x.slice(p.length); const i=process.argv.indexOf(name); return i>=0?process.argv[i+1]:fallback; }
async function create(model, meta, bytes=false){ const m=JSON.parse(readFileSync(meta,'utf8')); const src=bytes?readFileSync(model):model; return m.kind==='squareformer'?SquareFormerEvaluator.create(src,m):OnnxEvaluator.create(src,m); }
const model=arg('--model','public/models/chessformer_v1_100m_e3_single.onnx');
const meta=arg('--meta','public/models/chessformer_v1_100m_e3_single.meta.json');
const tol=Number(arg('--tol','1e-7'));
const fens=[
 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
 'r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1',
 '8/P6p/8/8/8/8/p6P/8 w - - 0 1',
 'rnbqkbnr/pp3ppp/2p1p3/3pP3/3P4/2N2N2/PPP2PPP/R1BQKB1R b KQkq - 0 4',
];
const pathEval=await create(model,meta,false); const byteEval=await create(model,meta,true);
let maxPolicyDiff=0, maxWdlDiff=0, positions=0;
for(const fen of fens){ const b=parseFen(fen); const a=await pathEval.evaluate(b); const z=await byteEval.evaluate(b); positions++; for(let i=0;i<3;i++) maxWdlDiff=Math.max(maxWdlDiff,Math.abs((a.wdl[i]??0)-(z.wdl[i]??0))); for(const m of legalMoves(b)){ const id=moveToActionId(m); maxPolicyDiff=Math.max(maxPolicyDiff,Math.abs((a.policy.get(id)??0)-(z.policy.get(id)??0))); } }
console.log(`METRIC evaluator_path_byte_parity_positions=${positions}`);
console.log(`METRIC evaluator_path_byte_parity_max_policy_diff=${maxPolicyDiff}`);
console.log(`METRIC evaluator_path_byte_parity_max_wdl_diff=${maxWdlDiff}`);
if(maxPolicyDiff>tol||maxWdlDiff>tol) process.exitCode=1;
