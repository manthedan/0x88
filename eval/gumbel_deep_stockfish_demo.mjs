#!/usr/bin/env node
import { mkdirSync, writeFileSync, readFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { spawn } from 'node:child_process';
import { parseFen, boardToFen } from '../src/chess/board.ts';
import { legalMoves, makeMove } from '../src/chess/movegen.ts';
import { moveToActionId, moveToUci } from '../src/chess/moveCodec.ts';
import { OnnxEvaluator } from '../src/nn/onnxEvaluator.ts';
import { SquareFormerEvaluator } from '../src/nn/squareformerEvaluator.ts';
import { GumbelRootPolicy, searchRoot } from '../src/search/puct.ts';

function arg(name, fallback='') { const p=`${name}=`; const x=process.argv.find(v=>v.startsWith(p)); if (x) return x.slice(p.length); const i=process.argv.indexOf(name); return i>=0 ? process.argv[i+1] : fallback; }
function nums(name, fallback) { return String(arg(name, fallback)).split(',').map(x=>Number(x.trim())).filter(Number.isFinite); }
async function load(model, metaPath) { const meta=JSON.parse(readFileSync(metaPath,'utf8')); return (meta.kind==='squareformer'||meta.kind==='squareformer_v2') ? SquareFormerEvaluator.create(model, meta) : OnnxEvaluator.create(model, meta); }
function cpForColor(evaln, fen, color) {
  const cpStm = evaln.cpStm;
  if (cpStm === null || cpStm === undefined) return null;
  const turn = fen.split(/\s+/)[1];
  const cpWhite = turn === 'w' ? cpStm : -cpStm;
  return color === 'w' ? cpWhite : -cpWhite;
}
function valueFromWdl(wdl) { return wdl[0]-wdl[2]; }
function topPolicy(board, ev, n) { return legalMoves(board).map(move=>({move:moveToUci(move), prior:ev.policy.get(moveToActionId(move))??0})).sort((a,b)=>b.prior-a.prior).slice(0,n); }
function topSearch(r, n) { return r.policy.slice().sort((a,b)=>b.visits-a.visits||b.prior-a.prior).slice(0,n).map(e=>({move:moveToUci(e.move), visits:e.visits, prior:e.prior, q:e.q, probability:e.probability})); }

class Stockfish {
  constructor({nodes=20000, variant='lite-single'}={}) {
    this.nodes=nodes; this.buf=[]; this.waiters=[];
    this.p=spawn('node', ['scripts/uci_stockfish_js_wrapper.mjs', variant], {stdio:['pipe','pipe','pipe']});
    this.p.stdout.setEncoding('utf8'); this.p.stdout.on('data', d=>this.onData(d));
    this.p.stderr.setEncoding('utf8'); this.p.stderr.on('data', d=>process.stderr.write(`[stockfish] ${d}`));
  }
  onData(d){ for(const line of d.split(/\r?\n/)){ if(!line) continue; const w=this.waiters[0]; if(w) w.lines.push(line); else this.buf.push(line); this.check(); } }
  cmd(s){ this.p.stdin.write(`${s}\n`); }
  waitUntil(pred){ return new Promise(resolve=>{ this.waiters.push({pred,resolve,lines:[]}); this.check(); }); }
  check(){ while(this.waiters.length){ const w=this.waiters[0]; let hit=-1; for(let i=0;i<w.lines.length;i++){ if(w.pred(w.lines[i], w.lines)){ hit=i; break; } } if(hit<0) return; this.waiters.shift(); resolveLater(w.resolve, w.lines.slice(0, hit+1)); } }
  async init(){ this.cmd('uci'); await this.waitUntil(l=>l==='uciok'); this.cmd('setoption name Threads value 1'); this.cmd('setoption name Hash value 64'); this.cmd('isready'); await this.waitUntil(l=>l==='readyok'); }
  async evalFen(fen){ this.cmd('ucinewgame'); this.cmd(`position fen ${fen}`); this.cmd(`go nodes ${this.nodes}`); const lines=await this.waitUntil(l=>l.startsWith('bestmove ')); let cp=null, mate=null, bestmove=null; for(const line of lines){ if(line.startsWith('info ') && line.includes(' score ')){ const parts=line.split(/\s+/); const i=parts.indexOf('score'); if(parts[i+1]==='cp'){ cp=Number(parts[i+2]); mate=null; } else if(parts[i+1]==='mate'){ mate=Number(parts[i+2]); cp=mate>0 ? 100000-mate : -100000-mate; } } if(line.startsWith('bestmove ')) bestmove=line.split(/\s+/)[1]; } return {cpStm:cp, mate, bestmove}; }
  close(){ try{ this.cmd('quit'); }catch{} this.p.kill('SIGTERM'); }
}
function resolveLater(resolve, value){ setImmediate(()=>resolve(value)); }

const POSITIONS = [
  { name: 'start', fen: 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1' },
  { name: 'kings-pawn', fen: 'rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2' },
  { name: 'queen-gambit', fen: 'rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3 0 2' },
  { name: 'castling-choice', fen: 'r1bq1rk1/ppp2ppp/2np1n2/4p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 6 8' },
  { name: 'sharp-center', fen: 'r2qkbnr/ppp2ppp/2np4/4p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 2 6' },
  { name: 'tactical-skirmish', fen: 'r1bq1rk1/pp2bppp/2nppn2/8/2BPP3/2N2N2/PPP2PPP/R1BQ1RK1 w - - 4 8' },
];

const model=arg('--model','artifacts/top3_100m_overnight_20260509/cnn96x8_100m/e08/model.onnx');
const meta=arg('--meta','artifacts/top3_100m_overnight_20260509/cnn96x8_100m/e08/model.meta.json');
const out=arg('--out','artifacts/gumbel_demos/cnn96_e08_deep_stockfish/demo.json');
const mdOut=arg('--md-out', out.replace(/\.json$/i,'.md'));
const lowVisits=nums('--visits','16,32,64');
const deepVisits=Number(arg('--deep-visits','256'));
const seeds=nums('--seeds','1,2,3,4');
const candidateCount=Number(arg('--candidate-count','8'));
const cpuct=Number(arg('--cpuct','1.5'));
const batchSize=Number(arg('--batch-size','8'));
const sfNodes=Number(arg('--stockfish-nodes','20000'));
const topN=Number(arg('--top','8'));
mkdirSync(dirname(out), {recursive:true});
const evaluator=await load(model, meta);
const sf=new Stockfish({nodes:sfNodes, variant:arg('--stockfish-variant','lite-single')});
await sf.init();
const rows=[]; const summary=[]; const sfCache=new Map();
async function sfEvalCached(fen){ if(!sfCache.has(fen)) sfCache.set(fen, await sf.evalFen(fen)); return sfCache.get(fen); }
console.log(`DEEP_GUMBEL model=${model} low=${lowVisits.join(',')} deep=${deepVisits} seeds=${seeds.join(',')} sf_nodes=${sfNodes}`);
for(const pos of POSITIONS){
  const board=parseFen(pos.fen); const rootColor=board.turn; const ev=await evaluator.evaluate(board); const priorTop=topPolicy(board, ev, topN);
  const beforeSf=await sfEvalCached(pos.fen); const beforeRootCp=cpForColor(beforeSf, pos.fen, rootColor);
  const deep=await searchRoot(board, evaluator, {visits:deepVisits, cpuct, batchSize, temperature:0});
  const deepMove=deep.move ? moveToUci(deep.move) : 'null';
  console.log(`POSITION ${pos.name} value=${valueFromWdl(ev.wdl).toFixed(4)} sf_cp_root=${beforeRootCp} deep=${deepMove}`);
  for(const visits of lowVisits){
    const candidates=[];
    const classic=await searchRoot(board, evaluator, {visits, cpuct, batchSize, temperature:0});
    candidates.push({mode:'classic', seed:null, result:classic});
    for(const seed of seeds){
      const gpol=new GumbelRootPolicy({candidateCount, seed});
      const g=await searchRoot(board, evaluator, {visits, cpuct, batchSize, temperature:0, searchPolicy:gpol});
      candidates.push({mode:'gumbel_root', seed, result:g});
    }
    for(const c of candidates){
      const move=c.result.move; const uci=move ? moveToUci(move) : 'null';
      let afterFen=null, afterSf=null, afterRootCp=null, cpLoss=null;
      if(move){ afterFen=boardToFen(makeMove(board, move)); afterSf=await sfEvalCached(afterFen); afterRootCp=cpForColor(afterSf, afterFen, rootColor); cpLoss=(beforeRootCp ?? 0) - (afterRootCp ?? 0); }
      rows.push({position:pos.name, fen:pos.fen, visits, deepVisits, mode:c.mode, seed:c.seed, selected:uci, deepSelected:deepMove, agreesWithDeep:uci===deepMove, modelValue:c.result.value, beforeRootCp, afterRootCp, cpLoss, stockfish:{nodes:sfNodes,before:beforeSf,after:afterSf}, priorTop, searchTop:topSearch(c.result, topN), deepTop:topSearch(deep, topN)});
    }
    const rs=rows.filter(r=>r.position===pos.name && r.visits===visits);
    const classicRow=rs.find(r=>r.mode==='classic'); const gs=rs.filter(r=>r.mode==='gumbel_root');
    const mean=x=>x.reduce((a,b)=>a+b,0)/Math.max(1,x.length);
    const gAgree=gs.filter(r=>r.agreesWithDeep).length;
    summary.push({position:pos.name, visits, deepMove, classic:classicRow.selected, classicAgrees:classicRow.agreesWithDeep, classicCpLoss:classicRow.cpLoss, gumbelUnique:[...new Set(gs.map(r=>r.selected))], gumbelAgreeSeeds:gAgree, gumbelMeanCpLoss:mean(gs.map(r=>r.cpLoss).filter(Number.isFinite)), gumbelBestCpLoss:Math.min(...gs.map(r=>r.cpLoss).filter(Number.isFinite)), gumbelWorstCpLoss:Math.max(...gs.map(r=>r.cpLoss).filter(Number.isFinite))});
    const s=summary[summary.length-1];
    console.log(`  v=${visits} deep=${deepMove} classic=${s.classic} agree=${s.classicAgrees?1:0} cp_loss=${s.classicCpLoss?.toFixed?.(1)} g_unique=${s.gumbelUnique.join(',')} g_agree=${gAgree}/${gs.length} g_mean_loss=${s.gumbelMeanCpLoss.toFixed(1)}`);
  }
}
sf.close();
const outData={model,meta,lowVisits,deepVisits,seeds,candidateCount,cpuct,batchSize,stockfishNodes:sfNodes,summary,rows};
writeFileSync(out, JSON.stringify(outData,null,2));
const md=['# Gumbel deep-search + Stockfish demo','',`model: \`${model}\``, `deep classic visits: ${deepVisits}`, `stockfish nodes: ${sfNodes}`,'','| position | visits | deep | classic | classic agree | classic cp loss | gumbel unique | gumbel agree seeds | gumbel mean cp loss |','|---|---:|---|---|---:|---:|---|---:|---:|',...summary.map(s=>`| ${s.position} | ${s.visits} | ${s.deepMove} | ${s.classic} | ${s.classicAgrees?1:0} | ${Number.isFinite(s.classicCpLoss)?s.classicCpLoss.toFixed(1):''} | ${s.gumbelUnique.join(', ')} | ${s.gumbelAgreeSeeds}/${seeds.length} | ${Number.isFinite(s.gumbelMeanCpLoss)?s.gumbelMeanCpLoss.toFixed(1):''} |`),'',`JSON: \`${out}\``].join('\n');
writeFileSync(mdOut, md+'\n');
console.log(`WROTE ${out}`); console.log(`WROTE ${mdOut}`); console.log(`METRIC deep_gumbel_rows=${rows.length}`); console.log(`METRIC deep_gumbel_summary=${summary.length}`);
