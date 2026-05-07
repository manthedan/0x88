#!/usr/bin/env node
import { readFileSync, writeFileSync, mkdirSync, readdirSync, statSync } from 'node:fs';
import { dirname, join } from 'node:path';

function arg(name, fallback = '') { const p=`${name}=`; const x=process.argv.find(v=>v.startsWith(p)); if(x)return x.slice(p.length); const i=process.argv.indexOf(name); return i>=0?process.argv[i+1]:fallback; }
function elo(scoreRate) { const s=Math.min(0.999,Math.max(0.001,scoreRate)); return 400*Math.log10(s/(1-s)); }
function normalApproxCi(scoreRate,games){ const s=Math.min(0.999,Math.max(0.001,scoreRate)); const se=Math.sqrt(Math.max(1e-9,s*(1-s)/Math.max(1,games))); const deriv=400/Math.log(10)*(1/s+1/(1-s)); return 1.96*se*deriv; }
function ptnmlFromPairScores(pairScores){ const c={'0':0,'0.5':0,'1':0,'1.5':0,'2':0}; for(const x of pairScores)c[String(x)]++; return c; }
function expand(pattern) {
  if (!pattern.includes('*')) return pattern.split(',').filter(Boolean);
  const slash=pattern.lastIndexOf('/'); const dir=slash>=0?pattern.slice(0,slash):'.'; const base=slash>=0?pattern.slice(slash+1):pattern;
  const re=new RegExp('^'+base.split('*').map(s=>s.replace(/[.+?^${}()|[\]\\]/g,'\\$&')).join('.*')+'$');
  return readdirSync(dir).map(f=>join(dir,f)).filter(p=>re.test(p.slice(dir.length+1)) && statSync(p).isFile() && !p.endsWith('.protocol.json'));
}
const inputs=expand(arg('--inputs')); const out=arg('--out','artifacts/anchor_arena/merged.json');
if(!inputs.length) throw new Error('usage --inputs "artifacts/jobs/*.json" --out merged.json');
const shards=[]; const games=[]; const protocolKeys=new Map();
for(const path of inputs){ const j=JSON.parse(readFileSync(path,'utf8')); if(!j.games||!j.protocol) continue; shards.push({path, protocol:j.protocol, summaries:j.summaries??[], games:j.games.length}); games.push(...j.games.map(g=>({...g, shard:path}))); const p=j.protocol; const key=JSON.stringify({candidate:p.candidate, visits:p.visits, cpuct:p.cpuct, maxPlies:p.maxPlies, openingsFile:p.openingsFile}); protocolKeys.set(key,(protocolKeys.get(key)??0)+1); }
if(protocolKeys.size>1 && arg('--allow-mixed','false')!=='true') throw new Error(`Refusing to merge incompatible protocols (${protocolKeys.size}); pass --allow-mixed=true for report-only merge`);
const anchors=[...new Set(games.map(g=>g.anchor))].sort(); const summaries=[];
for(const anchor of anchors){ const ag=games.filter(g=>g.anchor===anchor); const score=ag.reduce((s,g)=>s+g.tinyScore,0); const n=ag.length; const wins=ag.filter(g=>g.tinyScore===1).length, draws=ag.filter(g=>g.tinyScore===0.5).length, losses=ag.filter(g=>g.tinyScore===0).length; const pairMap=new Map(); for(const g of ag){ const key=`${g.shard}|${g.opening}`; const cur=pairMap.get(key)??0; pairMap.set(key,cur+g.tinyScore); } const pairScores=[...pairMap.values()].filter(x=>[0,0.5,1,1.5,2].includes(x)); const scoreRate=score/Math.max(1,n); summaries.push({anchor,games:n,pairs:pairScores.length,wins,draws,losses,scoreRate,eloDiff:elo(scoreRate),eloCi95:normalApproxCi(scoreRate,n),ptnml:ptnmlFromPairScores(pairScores),illegal:ag.filter(g=>g.illegal).length}); }
const protocol={kind:'merged_uci_anchor_arena', mergedUtc:new Date().toISOString(), shardCount:shards.length, shards:shards.map(s=>({path:s.path, games:s.games, protocol:s.protocol}))};
mkdirSync(dirname(out),{recursive:true}); writeFileSync(out,JSON.stringify({protocol,summaries,games},null,2)); writeFileSync(`${out}.protocol.json`,JSON.stringify(protocol,null,2));
for(const s of summaries){ console.log(`METRIC merged_${s.anchor}_score_rate=${s.scoreRate.toFixed(6)}`); console.log(`METRIC merged_${s.anchor}_elo_diff=${s.eloDiff.toFixed(3)}`); console.log(`METRIC merged_${s.anchor}_elo_ci95=${s.eloCi95.toFixed(3)}`); console.log(`METRIC merged_${s.anchor}_wdl=${s.wins}_${s.draws}_${s.losses}`); console.log(`METRIC merged_${s.anchor}_illegal=${s.illegal}`); }
console.log(`METRIC merged_games=${games.length}`);
console.log(`wrote ${out}`);
