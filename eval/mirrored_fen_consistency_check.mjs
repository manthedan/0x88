#!/usr/bin/env node
import { parseFen, boardToFen } from '../src/chess/board.ts';
import { legalMoves } from '../src/chess/movegen.ts';
import { moveFromUci, moveToActionId, moveToUci } from '../src/chess/moveCodec.ts';

const FENS = [
  'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
  'r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1',
  'r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1',
  '8/P6p/8/8/8/8/p6P/8 w - - 0 1',
  '8/P6p/8/8/8/8/p6P/8 b - - 0 1',
  'rnbqkbnr/pp3ppp/2p1p3/3pP3/3P4/2N2N2/PPP2PPP/R1BQKB1R b KQkq - 0 4',
];
function mirSq(i){ const f=i%8, r=Math.floor(i/8); return f+(7-r)*8; }
function mirCastle(c){ if(!c||c==='-') return '-'; let out=''; for(const x of c){ if(x==='K')out+='k'; else if(x==='Q')out+='q'; else if(x==='k')out+='K'; else if(x==='q')out+='Q'; } return out || '-'; }
function mirrorBoard(b){ const squares=Array(64).fill(null); for(let i=0;i<64;i++){ const p=b.squares[i]; if(p) squares[mirSq(i)] = `${p[0]==='w'?'b':'w'}${p[1]}`; } return { squares, turn:b.turn==='w'?'b':'w', castling:mirCastle(b.castling), epSquare:b.epSquare==null?null:mirSq(b.epSquare), halfmove:b.halfmove, fullmove:b.fullmove }; }
function mirrorUci(uci){ const m=moveFromUci(uci); return moveToUci({from:mirSq(m.from), to:mirSq(m.to), promotion:m.promotion}); }
function symmetricPrior(uci){ const mu=mirrorUci(uci); const key=uci < mu ? uci : mu; let h=0; for(const c of key) h=(h*33+c.charCodeAt(0))>>>0; return 1 + (h % 997); }
let legalSets=0, roundtrips=0, priorParity=0;
for (const fen of FENS) {
  const b=parseFen(fen), mb=mirrorBoard(b), rb=mirrorBoard(mb);
  if (boardToFen(rb) !== boardToFen(b)) throw new Error(`mirror roundtrip failed: ${fen} -> ${boardToFen(rb)}`);
  roundtrips++;
  const a=new Set(legalMoves(b).map(m=>mirrorUci(moveToUci(m))));
  const z=new Set(legalMoves(mb).map(moveToUci));
  if (a.size !== z.size || [...a].some(x=>!z.has(x))) throw new Error(`mirrored legal move set mismatch: ${fen}`);
  legalSets++;
  const moves=legalMoves(b).map(moveToUci);
  const total=moves.reduce((s,u)=>s+symmetricPrior(u),0);
  const mmoves=legalMoves(mb).map(moveToUci);
  const mtotal=mmoves.reduce((s,u)=>s+symmetricPrior(u),0);
  for (const u of moves) {
    const p=symmetricPrior(u)/total, mp=symmetricPrior(mirrorUci(u))/mtotal;
    if (Math.abs(p-mp)>1e-12) throw new Error(`mirrored prior mismatch: ${fen} ${u}`);
  }
  for (const m of legalMoves(b)) {
    const mid=moveToActionId(moveFromUci(mirrorUci(moveToUci(m))));
    if (mid < 0 || mid >= 64*64*5) throw new Error(`mirrored action id range failed: ${fen}`);
  }
  priorParity++;
}
console.log(`METRIC mirrored_fen_roundtrips=${roundtrips}`);
console.log(`METRIC mirrored_fen_legal_sets=${legalSets}`);
console.log(`METRIC mirrored_fen_prior_parity=${priorParity}`);
