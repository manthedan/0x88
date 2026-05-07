#!/usr/bin/env node
import assert from 'node:assert/strict';
import { parseFen } from '../src/chess/board.ts';
import { legalMoves } from '../src/chess/movegen.ts';
import { moveToActionId, moveFromUci, moveToUci } from '../src/chess/moveCodec.ts';
import { searchRoot } from '../src/search/puct.ts';

function moveEq(a,b){return a.from===b.from&&a.to===b.to&&(a.promotion??'')===(b.promotion??'');}
function legalByUci(board, uci){const want=moveFromUci(uci); return legalMoves(board).find(m=>moveEq(m,want));}
function policyFor(board, entries){const m=new Map(); for(const [uci,p] of Object.entries(entries)){const mv=legalByUci(board,uci); if(mv) m.set(moveToActionId(mv),p);} return m;}
function evaluator(fn){return { async evaluate(board, opts={}){ return fn(board, opts); } };}
const START='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';

async function testPolicyIdentity(){
  const ev=evaluator((board)=>({ policy:policyFor(board,{g1f3:0.9,e2e4:0.1}), wdl:[0.5,0,0.5] }));
  const r=await searchRoot(parseFen(START),ev,{visits:1,temperature:0});
  assert.equal(moveToUci(r.move),'g1f3','visits=1 should select legal policy argmax');
  assert.equal(r.policy.reduce((s,x)=>s+x.prior,0).toFixed(6),'1.000000','root priors should normalize to 1');
}

async function testAllZeroPolicyUniformFallback(){
  const ev=evaluator(()=>({ policy:new Map(), wdl:[0.5,0,0.5] }));
  const r=await searchRoot(parseFen(START),ev,{visits:1,temperature:0});
  const priors=r.policy.map(x=>x.prior);
  assert.equal(priors.length,20);
  assert.ok(priors.every(p=>Math.abs(p-1/20)<1e-12),'missing/all-zero policy should fallback to uniform legal priors');
}

async function testValuePerspectiveFlip(){
  const ev=evaluator((board)=>({ policy:policyFor(board,{e2e4:1,d2d4:0.01}), wdl:[0.9,0,0.1] }));
  const r=await searchRoot(parseFen(START),ev,{visits:1,temperature:0});
  const e=r.policy.find(x=>moveToUci(x.move)==='e2e4');
  assert.ok(e, 'e2e4 edge exists');
  assert.ok(Math.abs(e.q + 0.8) < 1e-9, `child side-to-move value +0.8 should back up as parent Q -0.8, got ${e.q}`);
}

async function testTerminalNoLegalMoves(){
  const ev=evaluator(()=>({ policy:new Map(), wdl:[0.5,0,0.5] }));
  const mate=await searchRoot(parseFen('rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3'),ev,{visits:4,temperature:0});
  assert.equal(mate.move,null);
  assert.equal(mate.value,-1,'checkmated side to move should be terminal loss');
  const stale=await searchRoot(parseFen('7k/5Q2/6K1/8/8/8/8/8 b - - 0 1'),ev,{visits:4,temperature:0});
  assert.equal(stale.move,null);
  assert.equal(stale.value,0,'stalemated side to move should be draw');
}

async function testTieBreakByQThenPrior(){
  const ev=evaluator((board)=>({ policy:policyFor(board,{g1f3:0.6,b1c3:0.4}), wdl:[0.5,0,0.5] }));
  const r=await searchRoot(parseFen(START),ev,{visits:2,temperature:0});
  assert.equal(r.visits,2);
  assert.equal(moveToUci(r.move),'g1f3','equal visits/Q should tie-break by larger/equal prior preserving deterministic first high-prior edge');
}

for (const t of [testPolicyIdentity,testAllZeroPolicyUniformFallback,testValuePerspectiveFlip,testTerminalNoLegalMoves,testTieBreakByQThenPrior]) {
  await t();
  console.log(`ok ${t.name}`);
}
console.log('METRIC puct_core_tests=5');
