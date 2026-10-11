import { readFileSync, writeFileSync } from 'node:fs';
import { runCell } from '../stockfish_upgrade_bench.mjs';

const [specPath, resultPath] = process.argv.slice(2);
const { cell, options } = JSON.parse(readFileSync(specPath, 'utf8'));
const result = await runCell(cell, options);
writeFileSync(resultPath, JSON.stringify({ nodes: Number(result.benchSignature), rows: result.benchRows, parityRows: result.rows }));
