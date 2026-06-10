#!/usr/bin/env node
import assert from 'node:assert/strict';
import { parseFen } from '../src/chess/board.ts';
import { legalMoves } from '../src/chess/movegen.ts';
import { moveToActionId, moveFromUci, moveToUci } from '../src/chess/moveCodec.ts';
import { GumbelRootPolicy, actionValuePuctPolicy, classicPuctPolicy, searchRoot } from '../src/search/puct.ts';

function moveEq(a,b){return a.from===b.from&&a.to===b.to&&(a.promotion??'')===(b.promotion??'');}
function legalByUci(board, uci){const want=moveFromUci(uci); return legalMoves(board).find(m=>moveEq(m,want));}
function policyFor(board, entries){const m=new Map(); for(const [uci,p] of Object.entries(entries)){const mv=legalByUci(board,uci); if(mv) m.set(moveToActionId(mv),p);} return m;}
function actionValuesFor(board, entries){const m=new Map(); for(const [uci,v] of Object.entries(entries)){const mv=legalByUci(board,uci); if(mv) m.set(moveToActionId(mv),v);} return m;}
function evaluator(fn){return { async evaluate(board, opts={}){ return fn(board, opts); } };}
function batchEvaluator(fn){
  const calls={single:0,batch:0,batchBoards:0};
  return { calls, async evaluate(board, opts={}){ calls.single++; return fn(board, opts); }, async evaluateBatch(boards, opts=[]){ calls.batch++; calls.batchBoards += boards.length; return boards.map((b,i)=>fn(b, opts[i] ?? {})); } };
}
function sequenceEvaluator(fn){
  const calls={single:0,batch:0,sequence:0,sequenceBatches:0,sequenceBoards:0};
  return {
    calls,
    async evaluate(board, opts={}){ calls.single++; return fn(board, opts); },
    async evaluateBatch(boards, opts=[]){ calls.batch++; return boards.map((b,i)=>fn(b, opts[i] ?? {})); },
    async evaluateBatchSequence(batches){ calls.sequence++; calls.sequenceBatches += batches.length; calls.sequenceBoards += batches.reduce((sum,b)=>sum+b.boards.length,0); return batches.map((batch)=>batch.boards.map((b,i)=>fn(b, batch.contexts?.[i] ?? {}))); },
  };
}
function searchContext(overrides={}){
  return {
    cpuct:1.5,
    fpu:0,
    temperature:1,
    cpuctSchedule:'constant',
    cpuctBase:38739,
    cpuctFactor:3.894,
    fpuStrategy:'constant',
    fpuReduction:0.33,
    avWeight:0,
    rankWeight:0,
    regretWeight:0,
    riskWeight:0,
    uncertaintyWeight:0,
    valueWdlAuxWeight:0,
    valueWdlBlendMode:'constant',
    valueWdlBaseTemp:1,
    valueWdlAuxTemp:1,
    cpuctVarianceWeight:0,
    cpuctVarianceMaxScale:2,
    butterflyWeight:0,
    butterflyDecay:0.995,
    butterflyMaxBonus:0.25,
    ...overrides,
  };
}
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

async function testBatchedEvaluatorPath(){
  const ev=batchEvaluator((board)=>({ policy:policyFor(board,{g1f3:0.6,b1c3:0.4,e2e4:0.2,d2d4:0.1}), wdl:[0.5,0,0.5] }));
  const r=await searchRoot(parseFen(START),ev,{visits:8,batchSize:4,temperature:0});
  assert.equal(r.visits,8);
  assert.ok(ev.calls.batch > 0, 'batched PUCT should call evaluateBatch');
  assert.ok(r.move, 'batched PUCT should return a move');
}

async function testBatchedDuplicateLeafDoesNotBackupZero(){
  const ev=batchEvaluator((board)=>({ policy:policyFor(board,{e2e4:1.0}), wdl:[0.9,0,0.1] }));
  const r=await searchRoot(parseFen(START),ev,{visits:4,batchSize:4,temperature:0});
  const edge=r.policy.find((entry)=>moveToUci(entry.move)==='e2e4');
  assert.ok(edge, 'e2e4 edge exists');
  assert.equal(edge.visits,4, 'batch should converge all duplicate leaf visits onto the forced-prior edge');
  assert.ok(Math.abs(edge.q + 0.8) < 1e-9, `duplicate-leaf batched backup should preserve evaluator value, got ${edge.q}`);
}

async function testActionValuePolicyGuidesSelection(){
  const ev=evaluator((board)=>({
    policy:policyFor(board,{g1f3:0.51,e2e4:0.49}),
    actionValues:actionValuesFor(board,{g1f3:-0.5,e2e4:0.8}),
    wdl:[0.5,0,0.5],
  }));
  const classic=await searchRoot(parseFen(START),ev,{visits:1,temperature:0});
  assert.equal(moveToUci(classic.move),'g1f3','classic PUCT should still follow policy prior');
  const av=await searchRoot(parseFen(START),ev,{visits:1,temperature:0,searchPolicy:actionValuePuctPolicy,avWeight:1.0});
  assert.equal(moveToUci(av.move),'e2e4','AV-PUCT should use edge actionValuePrior to guide low-visit selection');
}

async function testGumbelRootPolicyIsBoundedToCandidateSet(){
  const ev=evaluator((board)=>({ policy:policyFor(board,{g1f3:0.40,e2e4:0.30,d2d4:0.20,b1c3:0.10}), wdl:[0.5,0,0.5] }));
  const g=await searchRoot(parseFen(START),ev,{visits:8,temperature:0,searchPolicy:new GumbelRootPolicy({candidateCount:2,seed:7})});
  const visited=g.policy.filter(e=>e.visits>0).map(e=>moveToUci(e.move));
  assert.ok(visited.length<=2, `Gumbel root should spend visits only inside candidate set, got ${visited.join(',')}`);
  assert.ok(g.move, 'Gumbel root should return a legal move');
}

async function testLowTemperatureRootPolicyStaysFinite(){
  const board=parseFen(START);
  const edges=['g1f3','e2e4'].map((uci,i)=>({ move:legalByUci(board,uci), prior:i===0?0.6:0.4, child:null, visits:i===0?10000:9000, valueSum:0, virtualVisits:0, side:'w' }));
  const policy=classicPuctPolicy.rootPolicy(edges, searchContext({temperature:0.01}), { board, historyFens:[], expanded:true, terminalValue:null, edges });
  const total=policy.reduce((sum,entry)=>sum+entry.probability,0);
  assert.ok(policy.every((entry)=>Number.isFinite(entry.probability)), `low-temperature probabilities should be finite: ${policy.map((e)=>e.probability).join(',')}`);
  assert.ok(Math.abs(total-1)<1e-12, `low-temperature probabilities should normalize, got ${total}`);
  assert.equal(moveToUci(classicPuctPolicy.chooseFinalMove(policy).move),'g1f3');
}

async function testPipelinedBackupCollisionModeDoesNotCrash(){
  const ev=sequenceEvaluator((board)=>({ policy:policyFor(board,{e2e4:1.0,e7e5:1.0,g1f3:0.5,b8c6:0.5}), wdl:[0.5,0,0.5] }));
  const r=await searchRoot(parseFen(START),ev,{visits:4,batchSize:1,batchPipelineDepth:2,batchCollisionMode:'backup',temperature:0});
  assert.equal(r.visits,4, 'pipelined backup mode should complete requested visits');
  assert.ok(r.move, 'pipelined backup mode should return a legal move');
  assert.ok((r.stats?.batchLeafCollisions ?? 0)>0, 'pipelined backup mode should observe and dedupe an in-flight leaf collision');
  assert.ok(ev.calls.sequence>0, 'test should exercise evaluateBatchSequence pipeline path');
}

const tests=[testPolicyIdentity,testAllZeroPolicyUniformFallback,testValuePerspectiveFlip,testTerminalNoLegalMoves,testTieBreakByQThenPrior,testBatchedEvaluatorPath,testBatchedDuplicateLeafDoesNotBackupZero,testActionValuePolicyGuidesSelection,testGumbelRootPolicyIsBoundedToCandidateSet,testLowTemperatureRootPolicyStaysFinite,testPipelinedBackupCollisionModeDoesNotCrash];
for (const t of tests) {
  await t();
  console.log(`ok ${t.name}`);
}
console.log(`METRIC puct_core_tests=${tests.length}`);
