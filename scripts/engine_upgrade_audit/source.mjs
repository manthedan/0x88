import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fetchUpstream, sha256 } from './provenance.mjs';
import { runProcess } from './process.mjs';

export class Unavailable extends Error {}

// Prefer the shipped, checked-in corresponding source. Verify its path is for
// this exact ref; never reuse another version's bench corpus implicitly.
export async function benchSource(context, file) {
  if (context.options.benchSource) return readFileSync(context.options.benchSource, 'utf8');
  const { manifest, upstream } = context.provenance;
  const archive = manifest?.sourceArchive?.path && join(context.artifacts, manifest.sourceArchive.path.split('/').at(-1));
  if (archive && existsSync(archive)) {
    if (!manifest.sourceArchive.sha256) throw new Error('Corresponding-source archive has no recorded sha256; supply reviewed --bench-source instead');
    if (sha256(archive) !== manifest.sourceArchive.sha256) throw new Error('Corresponding-source archive sha256 mismatch');
    const listing = await runProcess('tar', ['-tzf', archive], { timeout: context.options.timeout });
    const candidates = listing.stdout.split('\n').filter((path) => path.includes(upstream.ref) && path.endsWith(`/${file}`));
    if (candidates.length === 1) return (await runProcess('tar', ['-xOf', archive, candidates[0]], { timeout: context.options.timeout })).stdout;
  }
  if (context.options.upstream) {
    const dir = await fetchUpstream(context);
    return (await runProcess('git', ['-C', dir, 'show', `FETCH_HEAD:${file}`], { timeout: context.options.timeout })).stdout;
  }
  throw new Unavailable(`locked ${file} unavailable locally; supply --bench-source or --upstream`);
}
