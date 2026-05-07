#!/usr/bin/env node
import { existsSync, readFileSync } from 'node:fs';
import { parseFen } from '../src/chess/board.ts';
import { legalMoves } from '../src/chess/movegen.ts';
import { moveToActionId } from '../src/chess/moveCodec.ts';
import { OnnxEvaluator } from '../src/nn/onnxEvaluator.ts';
import { SquareFormerEvaluator } from '../src/nn/squareformerEvaluator.ts';

function arg(name, fallback=''){ const p=`${name}=`; const x=process.argv.find(v=>v.startsWith(p)); if(x)return x.slice(p.length); const i=process.argv.indexOf(name); return i>=0?process.argv[i+1]:fallback; }
const cnnModel=arg('--cnn-model',process.env.CNN_MODEL??''); const cnnMeta=arg('--cnn-meta',process.env.CNN_META??'');
const sfModel=arg('--squareformer-model',process.env.SQUAREFORMER_MODEL??'public/models/chessformer_v1_100m_e3_single.onnx'); const sfMeta=arg('--squareformer-meta',process.env.SQUAREFORMER_META??'public/models/chessformer_v1_100m_e3_single.meta.json');
const fens=[
 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
 'r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1',
 '4k3/P6p/8/8/8/8/p6P/4K3 b - - 0 1',
 'rnbqkbnr/pp3ppp/2p1p3/3pP3/3P4/2N2N2/PPP2PPP/R1BQKB1R b KQkq - 0 4',
];
function legalIdSet(fen){ return new Set(legalMoves(parseFen(fen)).map(moveToActionId)); }
let structural=0; for(const fen of fens){ const ids=legalIdSet(fen); if(!ids.size) throw new Error(`no legal ids ${fen}`); for(const id of ids) if(id<0||id>=64*64*5) throw new Error(`bad action id ${id}`); structural++; }
console.log(`METRIC adapter_policy_key_structural_positions=${structural}`);
if(cnnModel && cnnMeta && existsSync(cnnModel) && existsSync(cnnMeta) && existsSync(sfModel) && existsSync(sfMeta)){
  const cnn=await OnnxEvaluator.create(cnnModel,JSON.parse(readFileSync(cnnMeta,'utf8')));
  const sf=await SquareFormerEvaluator.create(sfModel,JSON.parse(readFileSync(sfMeta,'utf8')));
  let positions=0;
  for(const fen of fens){ const b=parseFen(fen); const expected=legalIdSet(fen); const ce=await cnn.evaluate(b), se=await sf.evaluate(b); const ck=new Set(ce.policy.keys()), sk=new Set(se.policy.keys()); for(const id of expected){ if(!ck.has(id)) throw new Error(`CNN missing legal action ${id} ${fen}`); if(!sk.has(id)) throw new Error(`SquareFormer missing legal action ${id} ${fen}`); } if(ck.size!==expected.size) throw new Error(`CNN extra/missing policy keys ${fen}`); if(sk.size!==expected.size) throw new Error(`SquareFormer extra/missing policy keys ${fen}`); positions++; }
  console.log(`METRIC cnn_squareformer_adapter_parity_positions=${positions}`);
}else{
  console.log('SKIP cnn_squareformer_adapter_parity_actual_models=1');
  console.log('METRIC cnn_squareformer_adapter_parity_positions=0');
}
