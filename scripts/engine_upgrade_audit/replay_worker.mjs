import { readFileSync, writeFileSync } from 'node:fs';
import { runCell } from '../stockfish_upgrade_bench.mjs';

const [specPath, resultPath] = process.argv.slice(2);
const { cell, options } = JSON.parse(readFileSync(specPath, 'utf8'));
// Stay inside runProcess's owning group so its outer timeout also kills UCI
// grandchildren. The standalone harness instead owns each cell's group.
const result = await runCell(cell, { ...options, ownProcessGroup: false });
writeFileSync(resultPath, JSON.stringify({ nodes: Number(result.benchSignature), rows: result.benchRows, parityRows: result.rows }));
