#!/usr/bin/env node
import { parseFen } from '../src/chess/board.ts';
import { legalMoves } from '../src/chess/movegen.ts';
import { moveFromUci, moveToActionId, moveToUci } from '../src/chess/moveCodec.ts';

const FENS = [
  'r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1',
  'r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1',
  '8/P6p/8/8/8/8/p6P/8 w - - 0 1',
  '8/P6p/8/8/8/8/p6P/8 b - - 0 1',
  'rnbqkbnr/pp3ppp/2p1p3/3pP3/3P4/2N2N2/PPP2PPP/R1BQKB1R b KQkq - 0 4',
];
let moves = 0;
for (const fen of FENS) {
  const seen = new Set();
  for (const move of legalMoves(parseFen(fen))) {
    const uci = moveToUci(move);
    const rt = moveFromUci(uci);
    if (moveToUci(rt) !== uci) throw new Error(`uci roundtrip ${fen} ${uci}`);
    const id = moveToActionId(move);
    if (id < 0 || id >= 64 * 64 * 5) throw new Error(`action id range ${fen} ${uci}`);
    if (seen.has(id)) throw new Error(`duplicate action id ${fen} ${uci}`);
    seen.add(id); moves++;
  }
}
console.log(`METRIC move_codec_roundtrip_positions=${FENS.length}`);
console.log(`METRIC move_codec_roundtrip_moves=${moves}`);
