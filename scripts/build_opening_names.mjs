// Run: node --experimental-strip-types scripts/build_opening_names.mjs
// Optional first argument: directory containing a.tsv … e.tsv for offline regeneration.
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { parsePgnGame } from '../src/chess/pgn.ts';
import { openingPositionKey } from '../src/lc0/openingNames.ts';

const revision = '3ba6b20d31b2e6125cb0e6a2e792f400717d9d39';
const source = `https://raw.githubusercontent.com/lichess-org/chess-openings/${revision}`;
const index = Object.create(null);
let entries = 0;
for (const volume of ['a', 'b', 'c', 'd', 'e']) {
  const name = `${volume}.tsv`;
  const response = process.argv[2] ? null : await fetch(`${source}/${name}`);
  if (response && !response.ok) throw new Error(`${name}: HTTP ${response.status}`);
  const tsv = response ? await response.text() : await readFile(`${process.argv[2]}/${name}`, 'utf8');
  const [header, ...rows] = tsv.trim().split(/\r?\n/);
  if (header !== 'eco\tname\tpgn') throw new Error(`Unexpected TSV header: ${header}`);
  for (const row of rows) {
    const [eco, name, pgn] = row.split('\t');
    if (!/^[A-E]\d\d$/.test(eco) || !name || !pgn) throw new Error(`Invalid opening: ${row}`);
    const { tree } = parsePgnGame(pgn);
    tree.toEnd();
    const expected = pgn
      .replace(/\d+\.(?:\.\.)?/g, '')
      .trim()
      .split(/\s+/).length;
    if (tree.current.ply !== expected) throw new Error(`Incomplete PGN: ${eco} ${name}`);
    index[openingPositionKey(tree.current.fen)] = [eco, name];
    entries++;
  }
}
await mkdir('public/openings', { recursive: true });
await writeFile('public/openings/eco.json', `${JSON.stringify(index)}\n`);
console.log(`${entries} source entries → ${Object.keys(index).length} positions (${revision})`);
