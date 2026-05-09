#!/usr/bin/env node
import { spawn } from 'node:child_process';
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname } from 'node:path';
import { parseFen, boardToFen } from '../src/chess/board.ts';
import { legalMoves, makeMove } from '../src/chess/movegen.ts';
import { moveToActionId, moveToUci } from '../src/chess/moveCodec.ts';
import { OnnxEvaluator } from '../src/nn/onnxEvaluator.ts';
import { SquareFormerEvaluator } from '../src/nn/squareformerEvaluator.ts';
import { searchRoot } from '../src/search/puct.ts';

function arg(name, fallback = undefined) { const p=`${name}=`; const x=process.argv.find(v=>v.startsWith(p)); if(x)return x.slice(p.length); const i=process.argv.indexOf(name); return i>=0?process.argv[i+1]:fallback; }
function num(name, fallback) { return Number(arg(name, String(fallback))); }
async function loadEvaluator(model, metaPath) { const meta=JSON.parse(readFileSync(metaPath,'utf8')); return (meta.kind==='squareformer'||meta.kind==='squareformer_v2')?SquareFormerEvaluator.create(model,meta):OnnxEvaluator.create(model,meta); }
function cpForSide(evalStm, fen, side) { if (evalStm.cp === null || evalStm.cp === undefined) return null; return fen.split(' ')[1] === side ? evalStm.cp : -evalStm.cp; }
function material(fen) { const v={p:1,n:3,b:3,r:5,q:9}; const out={w:0,b:0,wq:0,bq:0}; for (const c of fen.split(' ')[0]) if (/[a-zA-Z]/.test(c)) { const s=c===c.toUpperCase()?'w':'b'; const p=c.toLowerCase(); out[s]+=v[p]??0; if(p==='q') out[`${s}q`]++; } return out; }
class Stockfish {
  constructor(path) {
    this.p=spawn(path,[],{stdio:['pipe','pipe','pipe']}); this.partial=''; this.lines=[]; this.waiters=[];
    this.p.stdout.setEncoding('utf8'); this.p.stderr.setEncoding('utf8');
    this.p.stdout.on('data', d => this.onData(d));
  }
  onData(d) {
    this.partial += d;
    const parts = this.partial.split(/\r?\n/); this.partial = parts.pop() ?? '';
    for (const line of parts) { if (!line) continue; this.lines.push(line); for (const w of [...this.waiters]) if (w.pred(line)) { this.waiters=this.waiters.filter(x=>x!==w); clearTimeout(w.t); w.resolve(line); } }
  }
  send(s) { this.p.stdin.write(s+'\n'); }
  async wait(pred, timeout=30000) { const hit=this.lines.find(pred); if(hit) return hit; return new Promise((resolve,reject)=>{ const w={pred,resolve,t:null}; w.t=setTimeout(()=>{this.waiters=this.waiters.filter(x=>x!==w); reject(new Error('stockfish timeout'));},timeout); this.waiters.push(w); }); }
  async init(threads=1, hash=64) { this.send('uci'); await this.wait(l=>l==='uciok'); this.send(`setoption name Threads value ${threads}`); this.send(`setoption name Hash value ${hash}`); this.send('isready'); await this.wait(l=>l==='readyok'); }
  async eval(fen, nodes) { this.lines=[]; this.send('ucinewgame'); this.send(`position fen ${fen}`); this.send(`go nodes ${nodes}`); let cp=null, mate=null, bestmove=null; while(true){ const line=await this.wait(l=>l.startsWith('info ')||l.startsWith('bestmove '),60000); this.lines=this.lines.filter(x=>x!==line); if(line.startsWith('info ') && line.includes(' score ')){ const parts=line.split(/\s+/); const i=parts.indexOf('score'); if(parts[i+1]==='cp'){ cp=Number(parts[i+2]); mate=null; } else if(parts[i+1]==='mate'){ mate=Number(parts[i+2]); cp=100000*(mate>0?1:-1); } } else if(line.startsWith('bestmove ')){ bestmove=line.split(/\s+/)[1]; break; } } return { cp, mate, bestmove }; }
  close(){ try{this.send('quit')}catch{} }
}
function applyUci(board, uci) { for (const m of legalMoves(board)) if (moveToUci(m) === uci) return makeMove(board, m); return null; }

const model=arg('--model'), meta=arg('--meta'), pos=arg('--positions-json'), out=arg('--out','artifacts/diagnostics/puct_override_stockfish_delta.json');
const visits=num('--visits',512), limit=num('--limit',100), nodes=num('--stockfish-nodes',2000), stockfish=arg('--stockfish','.local_engines/stockfish_pkg/usr/games/stockfish');
if(!model||!meta||!pos) throw new Error('usage --model --meta --positions-json --out [--visits 512] [--limit 100]');
const data=JSON.parse(readFileSync(pos,'utf8')); const fens=(data.positions??data).map(x=>x.fen??x.fen_before).filter(Boolean).slice(0,limit);
const evaluator=await loadEvaluator(model,meta); const sf=new Stockfish(stockfish); await sf.init();
const rows=[]; let overrides=0, puctBetter=0, puctWorse=0, puctEqual=0, cpSum=0, cpN=0, queenRiskDelta=0, materialDeltaSum=0;
for (let i=0;i<fens.length;i++) {
  const fen=fens[i]; const board=parseFen(fen); const side=board.turn;
  const ev=await evaluator.evaluate(board); const legal=legalMoves(board);
  const policy=legal.map(m=>({move:m,uci:moveToUci(m),prior:Math.max(0,ev.policy.get(moveToActionId(m))??0)})).sort((a,b)=>b.prior-a.prior);
  const total=policy.reduce((s,x)=>s+x.prior,0)||1; for(const x of policy) x.prior/=total;
  const ptop=policy[0]; const root=await searchRoot(board,evaluator,{visits,temperature:0}); const puctUci=root.move?moveToUci(root.move):'';
  const puctEntry=root.policy.find(x=>moveToUci(x.move)===puctUci); const policyEntry=root.policy.find(x=>moveToUci(x.move)===ptop.uci);
  if (puctUci && puctUci !== ptop.uci) {
    overrides++;
    const afterPolicy=applyUci(board,ptop.uci); const afterPuct=applyUci(board,puctUci); if(!afterPolicy||!afterPuct) continue;
    const fp=boardToFen(afterPolicy), fu=boardToFen(afterPuct); const ep=await sf.eval(fp,nodes), eu=await sf.eval(fu,nodes);
    const cpPolicy=cpForSide(ep,fp,side), cpPuct=cpForSide(eu,fu,side); const cpDelta=(cpPolicy==null||cpPuct==null)?null:cpPuct-cpPolicy;
    if(cpDelta!=null){ cpSum+=cpDelta; cpN++; if(cpDelta>25)puctBetter++; else if(cpDelta<-25)puctWorse++; else puctEqual++; }
    const mb=material(fen), mp=material(fp), mu=material(fu); const matPolicy=(mp[side]-mb[side])-(mp[side==='w'?'b':'w']-mb[side==='w'?'b':'w']); const matPuct=(mu[side]-mb[side])-(mu[side==='w'?'b':'w']-mb[side==='w'?'b':'w']); materialDeltaSum += matPuct - matPolicy;
    const qBefore=mb[`${side}q`], qPolicy=mp[`${side}q`]<qBefore, qPuct=mu[`${side}q`]<qBefore; queenRiskDelta += (qPuct?1:0)-(qPolicy?1:0);
    rows.push({ fen, side, policy_top_move:ptop.uci, puct_move:puctUci, policy_prior_policy_top:ptop.prior, policy_prior_puct_move:policy.find(x=>x.uci===puctUci)?.prior??0, root_Q_policy_top:policyEntry?.q??null, root_Q_puct_move:puctEntry?.q??null, visits_policy_top:policyEntry?.visits??0, visits_puct_move:puctEntry?.visits??0, stockfish:{nodes, policy_after:ep, puct_after:eu, cp_policy_for_side:cpPolicy, cp_puct_for_side:cpPuct, cp_delta_puct_minus_policy:cpDelta}, material:{before:mb, after_policy:mp, after_puct:mu, delta_puct_minus_policy:matPuct-matPolicy}, queen_loss_after_move:{policy:qPolicy,puct:qPuct} });
  }
  if ((i+1)%10===0) console.error(`[override-delta] ${i+1}/${fens.length} overrides=${overrides}`);
}
sf.close();
const summary={positions:fens.length, visits, overrides, puct_override_rate:overrides/Math.max(1,fens.length), annotated:rows.length, puct_better:puctBetter, puct_equal:puctEqual, puct_worse:puctWorse, puct_override_stockfish_win_rate:puctBetter/Math.max(1,cpN), mean_cp_delta_puct_minus_policy:cpSum/Math.max(1,cpN), queen_risk_delta:queenRiskDelta, mean_material_delta_puct_minus_policy:materialDeltaSum/Math.max(1,rows.length)};
mkdirSync(dirname(out),{recursive:true}); writeFileSync(out,JSON.stringify({summary,rows},null,2));
for (const [k,v] of Object.entries(summary)) console.log(`METRIC ${k}=${typeof v==='number'?v.toFixed?.(6)??v:v}`);
console.log(`wrote ${out}`);
