#!/usr/bin/env node
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname } from 'node:path';
import { parseFen } from '../src/chess/board.ts';
import { legalMoves } from '../src/chess/movegen.ts';
import { moveToUci, moveToActionId } from '../src/chess/moveCodec.ts';
import { chooseMove } from '../src/search/puct.ts';
import { OnnxEvaluator } from '../src/nn/onnxEvaluator.ts';
function arg(name, fallback) { const p=`${name}=`; const inline=process.argv.find(v=>v.startsWith(p)); if(inline) return inline.slice(p.length); const i=process.argv.indexOf(name); return i>=0?process.argv[i+1]:fallback; }
const model=arg('--model','artifacts/selfplay_best.onnx'), meta=arg('--meta','artifacts/selfplay_best.meta.json'), out=arg('--out','artifacts/eval/puct_sweep.json');
const visitsList=arg('--visits-list','1,4,8,16,32,64').split(',').map(Number), cpuctList=arg('--cpuct-list','0.5,0.75,1,1.25,1.5,2,2.5,3').split(',').map(Number);
const suite=[
  {id:'mate_in_1_back_rank',fen:'6k1/5ppp/8/8/8/8/8/6KQ w - - 0 1',best:['h1a8']},
  {id:'mate_in_1_rook',fen:'6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1',best:['a1a8']},
  {id:'mate_in_1_queen_file',fen:'6k1/5ppp/8/8/8/8/8/3Q2K1 w - - 0 1',best:['d1d8']},
  {id:'mate_in_1_smother_net',fen:'6k1/6pp/8/8/8/5N2/6PP/6K1 w - - 0 1',best:['f3g5']},
  {id:'win_hanging_queen',fen:'4k3/8/8/8/3q4/8/4Q3/4K3 w - - 0 1',best:['e2d3','e2d2','e2e8']},
  {id:'must_recapture_queen',fen:'4k3/8/8/8/3q4/8/3Q4/4K3 w - - 0 1',best:['d2d4']},
  {id:'recapture_rook',fen:'4k3/8/8/8/4r3/8/4R3/4K3 w - - 0 1',best:['e2e4']},
  {id:'knight_fork_king_queen',fen:'4k3/8/8/3q4/5N2/8/8/4K3 w - - 0 1',best:['f4d5']},
  {id:'bishop_skewer_queen',fen:'4k3/6q1/8/8/8/2B5/8/4K3 w - - 0 1',best:['c3g7']},
  {id:'promote_queen',fen:'6k1/P7/8/8/8/8/8/6K1 w - - 0 1',best:['a7a8q']},
  {id:'underpromotion_knight_check',fen:'6k1/P7/8/8/8/8/5PPP/6K1 w - - 0 1',best:['a7a8q','a7a8n']},
  {id:'avoid_illegal_in_check_rook',fen:'4k3/8/8/8/8/8/4r3/4K3 w - - 0 1',legalOnly:true},
  {id:'avoid_illegal_in_check_bishop',fen:'4k3/8/8/8/8/8/3b4/4K3 w - - 0 1',legalOnly:true},
  {id:'terminal_stalemate_sanity',fen:'7k/5Q2/7K/8/8/8/8/8 b - - 0 1',legalOnly:true}
];
const evaluator=await OnnxEvaluator.create(model, JSON.parse(readFileSync(meta,'utf8')));
async function policyOnly(t){const b=parseFen(t.fen), moves=legalMoves(b), legal=new Set(moves.map(moveToUci)), ev=await evaluator.evaluate(b,{historyFens:[]}); let best=moves[0], bp=-1; for(const m of moves){const p=ev.policy.get(moveToActionId(m))??0; if(p>bp){bp=p; best=m;}} const u=best?moveToUci(best):''; return {uci:u,ok:t.legalOnly?legal.has(u):legal.has(u)&&t.best.includes(u),prob:bp};}
async function evalSetting(visits,cpuct){let pass=0; const details=[]; for(const t of suite){const b=parseFen(t.fen), legal=new Set(legalMoves(b).map(moveToUci)), r=await chooseMove(b,evaluator,{visits,cpuct}), u=r.move?moveToUci(r.move):'', ok=t.legalOnly?(!!u&&legal.has(u)):(!!u&&legal.has(u)&&t.best.includes(u)); if(ok) pass++; details.push({id:t.id,move:u,ok,value:r.value,visits:r.visits});} return {visits,cpuct,pass,total:suite.length,passRate:pass/suite.length,details};}
const rows=[]; let pp=0; const pd=[]; for(const t of suite){const d=await policyOnly(t); pd.push({id:t.id,...d}); if(d.ok) pp++;} rows.push({mode:'policy',visits:0,cpuct:0,pass:pp,total:suite.length,passRate:pp/suite.length,details:pd}); console.log(`RESULT mode=policy visits=0 cpuct=0 pass=${pp}/${suite.length} rate=${(pp/suite.length).toFixed(6)}`);
for(const visits of visitsList) for(const cpuct of cpuctList){const r=await evalSetting(visits,cpuct); rows.push({mode:'puct',...r}); console.log(`RESULT mode=puct visits=${visits} cpuct=${cpuct} pass=${r.pass}/${r.total} rate=${r.passRate.toFixed(6)}`);}
mkdirSync(dirname(out),{recursive:true}); writeFileSync(out,JSON.stringify({model,meta,rows},null,2)); const best=rows.filter(r=>r.mode==='puct').sort((a,b)=>b.passRate-a.passRate||b.visits-a.visits)[0];
console.log(`METRIC puct_sweep_best_pass_rate=${best.passRate.toFixed(6)}`); console.log(`METRIC puct_sweep_best_visits=${best.visits}`); console.log(`METRIC puct_sweep_best_cpuct=${best.cpuct}`); console.log(`METRIC puct_sweep_policy_pass_rate=${(pp/suite.length).toFixed(6)}`);
