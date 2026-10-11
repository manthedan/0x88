import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, readdirSync } from 'node:fs';
import { basename, dirname, join, resolve } from 'node:path';
import { parseCommitBench, parseTags } from './parsers.mjs';
import { runProcess } from './process.mjs';

const json = (path) => JSON.parse(readFileSync(path, 'utf8'));
export const sha256 = (path) => createHash('sha256').update(readFileSync(path)).digest('hex');

export function loadProvenance(engine, artifacts, options) {
  const manifests = readdirSync(artifacts).filter((name) => name.endsWith('.manifest.json'));
  if (!options.manifest && manifests.length > 1) throw new Error('Ambiguous manifests; select --manifest <path>');
  const manifestPath = options.manifest ?? (manifests.length ? join(artifacts, manifests[0]) : null);
  const manifest = manifestPath ? json(manifestPath) : null;
  // Candidate directories must never silently inherit this checkout's old pin.
  const candidateRoot = basename(dirname(artifacts)) === 'public' ? dirname(dirname(artifacts)) : null;
  const lockPath = options.lock ?? (candidateRoot ? join(candidateRoot, 'engines', engine, 'upstream.lock.json') : null);
  const hasLock = Boolean(options.lock || (lockPath && existsSync(lockPath)));
  const lock = hasLock ? json(lockPath) : null;
  const pin = hasLock ? lock?.upstream : manifest?.upstream;
  if (!pin?.repo || !(pin.ref ?? pin.commit) || !pin.license) throw new Error('Missing locked repo/ref/license; supply --lock or --manifest');
  const upstream = { ...pin, ref: pin.ref ?? pin.commit };
  return { lock, lockPath: lock ? lockPath : null, manifest, manifestPath, upstream,
    network: lock?.network ?? manifest?.assets ?? [] };
}

export async function fetchUpstream(context) {
  if (context.upstreamDir) return context.upstreamDir;
  const dir = join(context.tmp, 'upstream');
  mkdirSync(dir);
  const { repo, ref } = context.provenance.upstream;
  await runProcess('git', ['init', '--quiet', dir], { timeout: 10000 });
  await runProcess('git', ['-C', dir, 'fetch', '--quiet', '--depth', '1', repo, ref], { timeout: context.options.timeout });
  const resolved = (await runProcess('git', ['-C', dir, 'rev-parse', 'FETCH_HEAD'])).stdout.trim();
  if (/^[a-f0-9]{40}$/.test(ref) && resolved !== ref) throw new Error(`Fetched ref mismatch: ${resolved} != ${ref}`);
  context.upstreamDir = dir;
  return dir;
}

export async function upstreamGate(context) {
  if (!context.options.upstream) return { name: 'provenance', status: 'SKIP', detail: 'upstream tag/commit checks disabled (use --upstream)', data: { check: 'upstream' } };
  const { repo, ref } = context.provenance.upstream;
  const output = await runProcess('git', ['ls-remote', '--tags', repo], { timeout: context.options.timeout });
  const tags = parseTags(output.stdout, context.provenance.upstream);
  const dir = await fetchUpstream(context);
  const message = (await runProcess('git', ['-C', dir, 'show', '-s', '--format=%B', 'FETCH_HEAD'])).stdout;
  context.commitBench = parseCommitBench(message);
  return { name: 'provenance', status: tags.tagMatchesRef ? 'PASS' : 'WARN',
    detail: `${tags.version}: ${tags.tagMatchesRef ? 'tag points at locked commit' : tags.tagCommit ? `tag points at ${tags.tagCommit}, NOT ${ref}` : 'not a literal engine tag'}; Bench: ${context.commitBench ?? 'absent'}; newest tags: ${tags.newestTags.map((tag) => tag.tag).join(', ')}`,
    data: { check: 'upstream', ...tags, commitMessage: message, bench: context.commitBench } };
}

export function provenanceGates(context) {
  context.verifiedNetworkFiles = [];
  const { upstream, lock, manifest, lockPath, manifestPath, network } = context.provenance;
  const problems = [];
  if (manifest?.engine && manifest.engine !== context.engine) problems.push(`manifest engine=${manifest.engine}`);
  const manifestRef = manifest?.upstream?.ref ?? manifest?.upstream?.commit;
  if (lock && manifestRef && manifestRef !== upstream.ref) problems.push(`lock/manifest ref mismatch (${manifestRef})`);
  if (lock && manifest?.upstream?.license && manifest.upstream.license !== upstream.license) problems.push('lock/manifest license mismatch');
  const artifacts = [];
  const entries = [...(manifest?.artifacts ?? [])];
  if (manifest?.sourceArchive?.path && !entries.some((entry) => entry.path === manifest.sourceArchive.path)) entries.push(manifest.sourceArchive);
  for (const entry of entries) {
    const path = join(context.artifacts, basename(entry.path));
    if (!existsSync(path)) continue; // inventory/required tier checks handle absence
    const actual = sha256(path);
    artifacts.push({ path, expected: entry.sha256, actual });
    if (entry.sha256 && actual !== entry.sha256) problems.push(`${basename(path)} sha256 differs from manifest`);
  }
  const gates = [{ name: 'provenance', status: problems.length ? 'FAIL' : 'PASS',
    detail: `${upstream.version ?? upstream.tag ?? '(no version)'} ${upstream.ref}; ${upstream.license}${problems.length ? `; ${problems.join('; ')}` : ''}`,
    data: { check: 'pin', upstream, network, lockPath, manifestPath, artifacts, problems } }];
  // Emscripten .data in these facades is exactly one preloaded net, not a tar.
  const dataFiles = readdirSync(context.artifacts).filter((file) => file.endsWith('.data'));
  const assets = Array.isArray(network) ? network : [network];
  for (const asset of assets) {
    const name = asset.name ?? asset.embeddedAs;
    const files = context.options?.nnue ? [resolve(context.options.nnue)] : dataFiles.length ? dataFiles.map((file) => join(context.artifacts, file)) :
      name ? [join(context.artifacts, name), join(context.artifacts, `${context.engine}-${name}`)].filter(existsSync) : [];
    const prefix = asset.sha256Prefix12AsName ? name?.match(/-([a-f0-9]{12})\./)?.[1] : null;
    const expected = asset.processedSha256 ?? asset.rawSha256 ?? asset.sha256 ?? prefix;
    if (!files.length) {
      gates.push({ name: 'provenance', status: 'SKIP', detail: `network ${name ?? asset.url ?? 'unspecified'} is embedded; separate-file sha256 not available`, data: { check: 'network', asset } });
      continue;
    }
    for (const path of files) {
      const actual = sha256(path);
      const matches = expected && (prefix ? actual.startsWith(expected) : actual === expected);
      if (matches) context.verifiedNetworkFiles.push(resolve(path));
      gates.push({ name: 'provenance', status: !expected ? 'SKIP' : matches ? 'PASS' : 'FAIL',
        detail: `${basename(path)} sha256=${actual}${prefix ? ' (12-digit lock prefix)' : ''}${!expected ? '; no locked digest' : !matches ? ` != ${expected}` : ''}`,
        data: { check: 'network', path: resolve(path), asset, expected, actual, prefixOnly: Boolean(prefix) } });
    }
  }
  return gates;
}
