#!/usr/bin/env node
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname } from 'node:path';
import { parseFen, START_FEN, boardToFen } from '../src/chess/board.ts';
import { legalMoves, makeMove } from '../src/chess/movegen.ts';
import { moveToActionId, moveToUci } from '../src/chess/moveCodec.ts';
import { chooseMove } from '../src/search/puct.ts';
import { OnnxEvaluator } from '../src/nn/onnxEvaluator.ts';
function arg(name, fallback='') { const p=`${name}=`; const x=process.argv.find(v=>v.startsWith(p)); if(x) return x.slice(p.length); const i=process.argv.indexOf(name); return i>=0?process.argv[i+1]:fallback; }
function specs(){return arg('--models').split(',').filter(Boolean).map(e=>{const [name,onnx,meta]=e.split(':'); return {name,onnx,meta};});}
const ss=specs(); const visits=Number(arg('--visits','32')); const maxPlies=Number(arg('--max-plies','40')); const out=arg('--out','artifacts/arena_10m/opening_diagnostic.json');
const openings=[START_FEN,'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1','rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1','rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq - 0 1','rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1','rnbqkbnr/pppppppp/8/8/8/2N5/PPPPPPPP/R1BQKBNR b KQkq - 1 1'];
const models=new Map(); for(const s of ss) models.set(s.name, await OnnxEvaluator.create(s.onnx, JSON.parse(readFileSync(s.meta,'utf8'))));
const rows=[]; const byPly={};
for(const fen of openings){ let board=parseFen(fen); const hist=[]; for(let ply=0; ply<maxPlies && legalMoves(board).length; ply++){
  const evals=[]; for(const s of ss){ const ev=await models.get(s.name).evaluate(board,{historyFens:hist.slice(-2).reverse()}); const chosen=await chooseMove(board, models.get(s.name), {visits, historyFens:hist.slice(-2).reverse()}); const move=chosen.move?moveToUci(chosen.move):''; const prior=chosen.move?(ev.policy.get(moveToActionId(chosen.move))||0):0; const q=ev.wdl[0]-ev.wdl[2]; evals.push({name:s.name, move, prior, q}); }
  const row={fen:boardToFen(board), ply, phase:ply<10?'opening':ply<24?'early_mid':'mid', evals}; rows.push(row);
  const key=row.phase; byPly[key]??={positions:0, agree:{}, q:{}, prior:{}}; byPly[key].positions++;
  for(const a of evals) { byPly[key].q[a.name]=(byPly[key].q[a.name]||0)+a.q; byPly[key].prior[a.name]=(byPly[key].prior[a.name]||0)+a.prior; }
  for(let i=0;i<evals.length;i++) for(let j=i+1;j<evals.length;j++){ const k=`${evals[i].name}__${evals[j].name}`; byPly[key].agree[k]=(byPly[key].agree[k]||0)+(evals[i].move===evals[j].move?1:0); }
  const lead = evals.find(e=>e.name===arg('--lead','48x5e6')) ?? evals[0]; const mv=legalMoves(board).find(m=>moveToUci(m)===lead.move) ?? legalMoves(board)[0]; hist.push(boardToFen(board)); board=makeMove(board,mv);
}}
for(const b of Object.values(byPly)){ for(const o of [b.q,b.prior,b.agree]) for(const k of Object.keys(o)) o[k]/=b.positions; }
mkdirSync(dirname(out),{recursive:true}); writeFileSync(out,JSON.stringify({visits,maxPlies,byPly,rows},null,2));
for(const [phase,b] of Object.entries(byPly)){ console.log(`METRIC diag_${phase}_positions=${b.positions}`); for(const [k,v] of Object.entries(b.agree)) console.log(`METRIC diag_${phase}_agree_${k}=${v.toFixed(6)}`); for(const [k,v] of Object.entries(b.prior)) console.log(`METRIC diag_${phase}_chosen_prior_${k}=${v.toFixed(6)}`); for(const [k,v] of Object.entries(b.q)) console.log(`METRIC diag_${phase}_q_${k}=${v.toFixed(6)}`); }
