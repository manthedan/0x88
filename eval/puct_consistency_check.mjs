#!/usr/bin/env node
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname } from 'node:path';
import { parseFen } from '../src/chess/board.ts';
import { legalMoves } from '../src/chess/movegen.ts';
import { moveToActionId, moveToUci } from '../src/chess/moveCodec.ts';
import { OnnxEvaluator } from '../src/nn/onnxEvaluator.ts';
import { SquareFormerEvaluator } from '../src/nn/squareformerEvaluator.ts';
import { searchRoot } from '../src/search/puct.ts';
function arg(n,f=''){const p=`${n}=`;const x=process.argv.find(v=>v.startsWith(p));if(x)return x.slice(p.length);const i=process.argv.indexOf(n);return i>=0?process.argv[i+1]:f}
async function load(m,mp){const meta=JSON.parse(readFileSync(mp,'utf8'));return meta.kind==='squareformer'?SquareFormerEvaluator.create(m,meta):OnnxEvaluator.create(m,meta)}
const model=arg('--model'), meta=arg('--meta'), pos=arg('--positions-json'), out=arg('--out',''), limit=Number(arg('--limit','100')), visits=arg('--visits','1,2,4,8,16,32').split(',').map(Number);
if(!model||!meta||!pos) throw new Error('usage --model --meta --positions-json');
const data=JSON.parse(readFileSync(pos,'utf8')); const fens=(data.positions??data).map(x=>x.fen??x.fen_before).filter(Boolean).slice(0,limit); const evr=await load(model,meta);
const stats={n:0}; for (const v of visits) stats[v]={sameTop:0, changed:0, riskierThanPolicy:0};
for (const fen of fens){ const b=parseFen(fen); const ev=await evr.evaluate(b); const legal=legalMoves(b); const policy=legal.map(m=>({m,uci:moveToUci(m),p:ev.policy.get(moveToActionId(m))??0})).sort((a,b)=>b.p-a.p); const ptop=policy[0]?.uci; stats.n++;
 for (const v of visits){ const r=await searchRoot(b,evr,{visits:v,temperature:0}); const su=r.move?moveToUci(r.move):''; if(su===ptop) stats[v].sameTop++; else stats[v].changed++; }
}
const protocol={kind:'puct_consistency_check',model,meta,positionsJson:pos,limit,visits,createdUtc:new Date().toISOString()};
const result={protocol,positions:stats.n,visits:Object.fromEntries(visits.map(v=>[v,stats[v]]))};
if(out){ mkdirSync(dirname(out),{recursive:true}); writeFileSync(out,JSON.stringify(result,null,2)); writeFileSync(`${out}.protocol.json`,JSON.stringify(protocol,null,2)); }
console.log(`positions=${stats.n}`); for(const v of visits){const s=stats[v]; console.log(`METRIC puct_v${v}_same_policy_top=${s.sameTop}`); console.log(`METRIC puct_v${v}_changed=${s.changed}`); console.log(`visits=${v} same=${s.sameTop}/${stats.n} changed=${s.changed}`)}
