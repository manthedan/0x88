#!/usr/bin/env node
import { readFileSync } from 'node:fs';

const doc = readFileSync('docs/artifact_retention_rollback_policy.md', 'utf8');
const publish = readFileSync('scripts/publish_hashed_artifacts_to_r2.mjs', 'utf8');

const failures = [];
function requireMatch(name, pattern, text = doc) {
  if (!pattern.test(text)) failures.push(name);
}

requireMatch('policy documents write-once content-addressed keys', /write-once/i);
requireMatch('policy documents rollback only repoints channels', /Rollback only repoints channels/i);
requireMatch('policy documents 30-90 day retention', /30[–-]90 days/);
requireMatch('policy documents no routine purge of immutable blobs', /Do not purge or overwrite/i);
requireMatch('publisher verifies key hash against artifact sha256', /keySha256 !== artifact\.sha256\.toLowerCase\(\)/, publish);

if (/allow-overwrite/.test(publish)) failures.push('publisher must not expose allow-overwrite');

if (failures.length) {
  console.error(JSON.stringify({ ok: false, failures }, null, 2));
  process.exit(1);
}

console.log(JSON.stringify({
  ok: true,
  checked: [
    'docs/artifact_retention_rollback_policy.md',
    'scripts/publish_hashed_artifacts_to_r2.mjs',
  ],
  invariants: [
    'write-once sha256 keys documented',
    'rollback only repoints channel manifests',
    '30-90 day retention documented',
    'publisher validates content-addressed key hash',
    'publisher has no overwrite escape hatch',
  ],
}, null, 2));
