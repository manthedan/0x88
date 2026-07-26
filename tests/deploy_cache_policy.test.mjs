import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdtemp, mkdir, writeFile, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { test } from 'node:test';
import { renderHeadersFile, generateHeadersFile, checkHeadersFileUpToDate } from '../scripts/generate_netlify_headers_file.mjs';
import { parseHeaderBlocks } from '../scripts/netlify_headers.mjs';

const repoRoot = process.cwd();
const checker = resolve(repoRoot, 'scripts/check_deploy_cache_policy.mjs');

function runChecker(cwd) {
  return new Promise((done) => {
    const child = spawn(process.execPath, [checker], { cwd, stdio: ['ignore', 'pipe', 'pipe'] });
    let stdout = '';
    let stderr = '';
    child.stdout.setEncoding('utf8');
    child.stderr.setEncoding('utf8');
    child.stdout.on('data', (chunk) => { stdout += chunk; });
    child.stderr.on('data', (chunk) => { stderr += chunk; });
    child.on('close', (status) => done({ status, stdout, stderr }));
  });
}

// A fixture mutation that silently fails to apply leaves a test asserting
// against the unmodified shipped config, which still passes — vacuously — for
// the green-path cases. Every substitution must therefore prove it matched.
function replaceOnce(text, find, replacement) {
  const first = text.indexOf(find);
  assert.notEqual(first, -1, `fixture substitution never matched, so the scenario under test was not applied: ${JSON.stringify(find)}`);
  assert.equal(text.indexOf(find, first + find.length), -1, `fixture substitution matched more than once, so the mutation is ambiguous: ${JSON.stringify(find)}`);
  return `${text.slice(0, first)}${replacement}${text.slice(first + find.length)}`;
}

// Fixtures are built from the real netlify.toml so cases stay anchored to the
// policy the repository actually ships. `_headers` is regenerated to mirror the
// documented workflow — edit the TOML, run the generator — except where a test
// is specifically about drift between the two.
async function fixture(mutate = (toml) => toml, { regenerate = true } = {}) {
  const original = await readFile(resolve(repoRoot, 'netlify.toml'), 'utf8');
  const netlify = mutate(original);
  const dir = await mkdtemp(join(tmpdir(), 'cache-policy-'));
  await mkdir(join(dir, 'public'), { recursive: true });
  await writeFile(join(dir, 'netlify.toml'), netlify);
  let headers;
  try {
    headers = regenerate ? renderHeadersFile(netlify) : renderHeadersFile(original);
  } catch {
    // A fixture whose TOML the generator cannot read still needs a file on disk
    // so the checker reaches its own parse failure rather than an ENOENT.
    headers = renderHeadersFile(original);
  }
  await writeFile(join(dir, 'public/_headers'), headers);
  return dir;
}

function failures(result) {
  return JSON.parse(result.stderr).failures;
}

test('the shipped deploy cache policy passes', async () => {
  const result = await runChecker(repoRoot);
  assert.equal(result.status, 0, result.stderr);
  assert.equal(JSON.parse(result.stdout).ok, true);
});

test('public/_headers is generated from netlify.toml and matches it', async () => {
  const [netlify, headers] = await Promise.all([
    readFile(resolve(repoRoot, 'netlify.toml'), 'utf8'),
    readFile(resolve(repoRoot, 'public/_headers'), 'utf8'),
  ]);
  assert.equal(headers, renderHeadersFile(netlify));
});

// Calling the pure renderer twice proves nothing. Generation has to run
// against a real tree, write, and be read back, or the committed artifact is
// never shown to be reproducible.
test('generating twice into a real tree produces identical bytes', async () => {
  const dir = await fixture();
  const first = generateHeadersFile(dir);
  const afterFirst = await readFile(join(dir, 'public/_headers'), 'utf8');
  const second = generateHeadersFile(dir);
  const afterSecond = await readFile(join(dir, 'public/_headers'), 'utf8');
  assert.equal(first, second);
  assert.equal(afterFirst, afterSecond);
  assert.equal(afterSecond, await readFile(resolve(repoRoot, 'public/_headers'), 'utf8'));
  assert.deepEqual(checkHeadersFileUpToDate(dir), { ok: true, path: 'public/_headers' });
});

// The generated file is committed, so drift is possible and must be caught.
test('a stale public/_headers is rejected', async () => {
  const dir = await fixture((toml) => replaceOnce(toml,
    '  for = "/engines/*"\n  [headers.values]\n    Cache-Control = "public, max-age=0, must-revalidate"',
    '  for = "/engines/*"\n  [headers.values]\n    Cache-Control = "public, max-age=60"',
  ), { regenerate: false });
  const result = await runChecker(dir);
  assert.equal(result.status, 1);
  assert.ok(
    failures(result).some((entry) => entry.includes('public/_headers is out of date with netlify.toml')),
    result.stderr,
  );
});

test('a path declared twice is rejected', async () => {
  const dir = await fixture((toml) => `${toml}\n[[headers]]\n  for = "/channels/*"\n  [headers.values]\n    Cache-Control = "public, max-age=0, no-cache"\n`);
  const result = await runChecker(dir);
  assert.equal(result.status, 1);
  assert.ok(
    failures(result).some((entry) => entry.includes('netlify.toml declares /channels/* more than once')),
    result.stderr,
  );
});

test('one-year immutable on a mutable logical alias is rejected', async () => {
  const dir = await fixture((toml) => replaceOnce(toml,
    '  for = "/engines/*"\n  [headers.values]\n    Cache-Control = "public, max-age=0, must-revalidate"',
    '  for = "/engines/*"\n  [headers.values]\n    Cache-Control = "public, max-age=31536000, immutable"',
  ));
  const result = await runChecker(dir);
  assert.equal(result.status, 1);
  assert.ok(
    failures(result).some((entry) => entry.includes('immutable Cache-Control outside content-addressed artifacts: /engines/*')),
    result.stderr,
  );
});

// RFC 9111: directive names are case insensitive and arguments may be quoted.
test('an uppercase immutable directive still trips the prohibition', async () => {
  const dir = await fixture((toml) => replaceOnce(toml,
    '  for = "/engines/*"\n  [headers.values]\n    Cache-Control = "public, max-age=0, must-revalidate"',
    '  for = "/engines/*"\n  [headers.values]\n    Cache-Control = "public, MAX-AGE=31536000, IMMUTABLE"',
  ));
  const result = await runChecker(dir);
  assert.equal(result.status, 1);
  assert.ok(
    failures(result).some((entry) => entry.includes('immutable Cache-Control outside content-addressed artifacts: /engines/*')),
    result.stderr,
  );
});

test('a quoted max-age argument still trips the prohibition', async () => {
  const dir = await fixture((toml) => replaceOnce(toml,
    '  for = "/engines/*"\n  [headers.values]\n    Cache-Control = "public, max-age=0, must-revalidate"',
    '  for = "/engines/*"\n  [headers.values]\n    Cache-Control = \'public, max-age="31536000", immutable\'',
  ));
  const result = await runChecker(dir);
  assert.equal(result.status, 1);
  assert.ok(
    failures(result).some((entry) => entry.includes('immutable Cache-Control outside content-addressed artifacts: /engines/*')),
    result.stderr,
  );
});

test('a required value differing only in separator whitespace and order is accepted', async () => {
  const dir = await fixture((toml) => replaceOnce(toml,
    '  for = "/*.html"\n  [headers.values]\n    Cache-Control = "public, max-age=0, must-revalidate"',
    '  for = "/*.html"\n  [headers.values]\n    Cache-Control = "must-revalidate,public,max-age=0"',
  ));
  const result = await runChecker(dir);
  assert.equal(result.status, 0, result.stderr);
});

test('dropping cross-origin CORP from content-addressed artifacts is rejected', async () => {
  const dir = await fixture((toml) => replaceOnce(toml, '    Cross-Origin-Resource-Policy = "cross-origin"\n', ''));
  const result = await runChecker(dir);
  assert.equal(result.status, 1);
  assert.ok(
    failures(result).some((entry) => entry.includes('missing cross-origin CORP')),
    result.stderr,
  );
});

test('write-once release manifests keep their immutable policy', async () => {
  const dir = await fixture((toml) => replaceOnce(toml,
    '  for = "/releases/*"\n  [headers.values]\n    Cache-Control = "public, max-age=31536000, immutable"',
    '  for = "/releases/*"\n  [headers.values]\n    Cache-Control = "public, max-age=60"',
  ));
  const result = await runChecker(dir);
  assert.equal(result.status, 1);
  assert.ok(
    failures(result).some((entry) => entry.includes('missing write-once release manifest Cache-Control')),
    result.stderr,
  );
});

// Netlify: "Any line beginning with # will be ignored as a comment."
test('a commented-out directive does not satisfy a requirement', async () => {
  const dir = await fixture((toml) => replaceOnce(toml,
    '    Cross-Origin-Resource-Policy = "cross-origin"',
    '    # Cross-Origin-Resource-Policy = "cross-origin"',
  ));
  const result = await runChecker(dir);
  assert.equal(result.status, 1);
  assert.ok(
    failures(result).some((entry) => entry.includes('missing cross-origin CORP')),
    result.stderr,
  );
});

test('a commented-out directive does not trip a prohibition', async () => {
  const dir = await fixture((toml) => replaceOnce(toml,
    '  for = "/engines/*"\n  [headers.values]',
    '  for = "/engines/*"\n  [headers.values]\n    # Cache-Control = "public, max-age=31536000, immutable"',
  ));
  const result = await runChecker(dir);
  assert.equal(result.status, 0, result.stderr);
});

test('a trailing comment does not hide a prohibited header', async () => {
  const dir = await fixture((toml) => replaceOnce(toml,
    '  for = "/engines/*"\n  [headers.values]',
    '  for = "/engines/*"\n  [headers.values]\n    Content-Encoding = "br" # publish-time sidecar',
  ));
  const result = await runChecker(dir);
  assert.equal(result.status, 1);
  assert.ok(
    failures(result).some((entry) => entry.includes('must not declare Content-Encoding on /engines/*')),
    result.stderr,
  );
});

// netlify.toml that Netlify itself could not read must not pass a gate that
// claims to verify it.
test('malformed TOML is reported, not skipped', async () => {
  const dir = await fixture((toml) => replaceOnce(toml,
    '  for = "/engines/*"\n  [headers.values]',
    '  for = "/engines/*"\n  [headers.values]\n    Cache-Control = public, max-age=0',
  ));
  const result = await runChecker(dir);
  assert.equal(result.status, 1);
  assert.ok(
    failures(result).some((entry) => entry.includes('Invalid TOML document')),
    result.stderr,
  );
});

// Netlify documents triple-quoted TOML strings for multi-value headers.
test('a documented multiline TOML header value is parsed, not rejected', async () => {
  const dir = await fixture((toml) => replaceOnce(toml,
    '  for = "/*.html"\n  [headers.values]\n    Cache-Control = "public, max-age=0, must-revalidate"',
    "  for = \"/*.html\"\n  [headers.values]\n    Cache-Control = '''\n    public,\n    max-age=0,\n    must-revalidate'''",
  ));
  const result = await runChecker(dir);
  assert.equal(result.status, 0, result.stderr);
});

// A multiline value may legally contain a line that looks like a TOML table.
test('a multiline value containing a bracketed line does not truncate the block', async () => {
  const dir = await fixture((toml) => replaceOnce(toml,
    '  for = "/lab/webgpu-lc0-diag/*"\n  [headers.values]\n    X-Robots-Tag = "noindex, nofollow, noarchive"',
    "  for = \"/lab/webgpu-lc0-diag/*\"\n  [headers.values]\n    X-Robots-Tag = '''\n    noindex,\n    nofollow,\n    noarchive'''\n    Cache-Control = \"public, max-age=0, must-revalidate\"",
  ));
  const result = await runChecker(dir);
  assert.equal(result.status, 0, result.stderr);
});

test('an ordinary TOML table after a headers block is not absorbed into it', async () => {
  const dir = await fixture((toml) => `${toml}\n[context.production.environment]\n  NODE_VERSION = "24"\n`);
  const result = await runChecker(dir);
  assert.equal(result.status, 0, result.stderr);
});

// RFC 9111: a directive stated twice with different arguments has no single
// meaning. Keeping only the last let `max-age=31536000, immutable, max-age=0`
// read as mutable while the emitted header still carried the year-long policy.
test('a duplicate max-age cannot hide a one-year immutable policy', async () => {
  const dir = await fixture((toml) => replaceOnce(toml,
    '  for = "/engines/*"\n  [headers.values]\n    Cache-Control = "public, max-age=0, must-revalidate"',
    '  for = "/engines/*"\n  [headers.values]\n    Cache-Control = "public, max-age=31536000, immutable, max-age=0"',
  ));
  const result = await runChecker(dir);
  assert.equal(result.status, 1);
  assert.ok(
    failures(result).some((entry) => entry.includes('immutable Cache-Control outside content-addressed artifacts: /engines/*')),
    result.stderr,
  );
  assert.ok(
    failures(result).some((entry) => entry.includes('states max-age more than once with different arguments')),
    result.stderr,
  );
});

// A table header may carry a trailing comment. Missing that dropped the whole
// block from the generated file, and the shared parser blessed its own output.
test('a commented table header does not drop the block from the generated file', async () => {
  const netlify = await readFile(resolve(repoRoot, 'netlify.toml'), 'utf8');
  const commented = replaceOnce(netlify,
    '[[headers]]\n  for = "/sw.js"',
    '[[headers]] # service worker\n  for = "/sw.js"',
  );
  assert.ok(renderHeadersFile(commented).includes('/sw.js'), 'the /sw.js block must survive a commented table header');
  const dir = await fixture(() => commented);
  const result = await runChecker(dir);
  assert.equal(result.status, 0, result.stderr);
});

// A TOML number, boolean, or array under [headers.values] is not a header this
// policy knows how to serve; coercing one would publish something nobody wrote.
test('a non-string header value is rejected rather than coerced', async () => {
  const dir = await fixture((toml) => replaceOnce(toml,
    '  for = "/engines/*"\n  [headers.values]',
    '  for = "/engines/*"\n  [headers.values]\n    X-Count = 42',
  ));
  const result = await runChecker(dir);
  assert.equal(result.status, 1);
  assert.ok(
    failures(result).some((entry) => entry.includes('X-Count must be a string, got number')),
    result.stderr,
  );
});

// A bare `[[headers]]` line inside a multiline value defeated the hand-rolled
// parser's line-based table detection. A real TOML parser reads it as data.
test('a multiline value containing a table-like line is read as data', async () => {
  const netlify = await readFile(resolve(repoRoot, 'netlify.toml'), 'utf8');
  const rendered = renderHeadersFile(replaceOnce(netlify,
    '  for = "/sw.js"\n  [headers.values]',
    "  for = \"/sw.js\"\n  [headers.values]\n    X-Note = '''\n[[headers]]\n'''",
  ));
  assert.ok(rendered.includes('X-Note: [[headers]]'), rendered);
  assert.ok(rendered.includes('/reckless/*.wasm'), 'later blocks must survive');
});

// Internal runs of whitespace are part of a value; only line breaks collapse.
test('internal whitespace in a header value survives generation', async () => {
  const netlify = await readFile(resolve(repoRoot, 'netlify.toml'), 'utf8');
  const rendered = renderHeadersFile(replaceOnce(netlify,
    '  for = "/sw.js"\n  [headers.values]',
    '  for = "/sw.js"\n  [headers.values]\n    X-Note = \'attachment; filename="a  b.txt"\'',
  ));
  assert.ok(rendered.includes('filename="a  b.txt"'), rendered);
});

// TOML basic strings carry escapes; emitting the raw backslashes would ship a
// different value than netlify.toml declares.
test('escaped quotes in a basic string are accepted and decoded', async () => {
  const netlify = await readFile(resolve(repoRoot, 'netlify.toml'), 'utf8');
  const rendered = renderHeadersFile(replaceOnce(netlify,
    '  for = "/sw.js"\n  [headers.values]',
    '  for = "/sw.js"\n  [headers.values]\n    X-Note = "a \\"b\\""',
  ));
  assert.ok(rendered.includes('X-Note: a "b"'), rendered);
});

// A real TOML parser guarantees the document is valid TOML, not that it says
// what this policy expects. `values = true` yields no entries, so the block
// would vanish from the generated file with the staleness check blessing it.
test('a non-table [headers.values] is rejected rather than read as empty', async () => {
  const dir = await fixture((toml) => replaceOnce(toml,
    '  for = "/sw.js"\n  [headers.values]\n    Cache-Control = "public, max-age=0, must-revalidate"\n    Service-Worker-Allowed = "/"',
    '  for = "/sw.js"\n  values = true',
  ));
  const result = await runChecker(dir);
  assert.equal(result.status, 1);
  assert.ok(
    failures(result).some((entry) => entry.includes('[headers.values] must be a table, got boolean')),
    result.stderr,
  );
});

test('an array [headers.values] does not become indexed pseudo-headers', () => {
  const invalid = [];
  const blocks = parseHeaderBlocks('[[headers]]\nfor="/x"\nvalues=["a"]\n', (message) => invalid.push(message));
  assert.deepEqual(blocks[0].entries, []);
  assert.ok(invalid.some((entry) => entry.includes('must be a table, got array')), invalid.join('; '));
});

test('a `headers` key that is not an array of tables is reported, not crashed on', () => {
  const invalid = [];
  const blocks = parseHeaderBlocks('[headers]\nfoo="bar"\n', (message) => invalid.push(message));
  assert.deepEqual(blocks, []);
  assert.ok(invalid.some((entry) => entry.includes('must be an array of [[headers]] tables')), invalid.join('; '));
});

// `max-age=031536000` is the same year; comparing the argument as text let the
// leading zero carry a mutable path past the prohibition.
test('a leading-zero max-age still trips the immutable prohibition', async () => {
  const dir = await fixture((toml) => replaceOnce(toml,
    '  for = "/engines/*"\n  [headers.values]\n    Cache-Control = "public, max-age=0, must-revalidate"',
    '  for = "/engines/*"\n  [headers.values]\n    Cache-Control = "public, max-age=031536000, immutable"',
  ));
  const result = await runChecker(dir);
  assert.equal(result.status, 1);
  assert.ok(
    failures(result).some((entry) => entry.includes('immutable Cache-Control outside content-addressed artifacts: /engines/*')),
    result.stderr,
  );
});

// _headers is line-oriented, so an interpolated path must not be able to forge
// its structure and mean something the TOML never said.
test('a path containing a line break cannot forge a second path line', () => {
  assert.throws(
    () => renderHeadersFile('[[headers]]\nfor = "/safe\\n/other"\n[headers.values]\nX-Note = "y"\n'),
    /contains a line break/,
  );
});

test('a path with leading whitespace is rejected', () => {
  assert.throws(
    () => renderHeadersFile('[[headers]]\nfor = " /x"\n[headers.values]\nX-Note = "y"\n'),
    /leading or trailing whitespace/,
  );
});

test('a header name that is not a token is rejected', () => {
  assert.throws(
    () => renderHeadersFile('[[headers]]\nfor = "/x"\n[headers.values]\n"X: Y" = "z"\n'),
    /is not a token/,
  );
});

// Losing cross-origin isolation does not break the build or any test — it
// breaks SharedArrayBuffer, and with it every threaded engine, in the browser.
test('removing cross-origin isolation from /* is rejected', async () => {
  const dir = await fixture((toml) => replaceOnce(toml,
    '  for = "/*"\n  [headers.values]\n    Cross-Origin-Opener-Policy = "same-origin"\n    Cross-Origin-Embedder-Policy = "require-corp"',
    '  for = "/*"\n  [headers.values]\n    X-Frame-Options = "DENY"\n    Cross-Origin-Embedder-Policy = "require-corp"',
  ));
  const result = await runChecker(dir);
  assert.equal(result.status, 1);
  assert.ok(
    failures(result).some((entry) => entry.includes('missing cross-origin isolation COOP on /*')),
    result.stderr,
  );
});

test('weakening COEP from require-corp is rejected', async () => {
  const dir = await fixture((toml) => replaceOnce(toml,
    '    Cross-Origin-Embedder-Policy = "require-corp"',
    '    Cross-Origin-Embedder-Policy = "unsafe-none"',
  ));
  const result = await runChecker(dir);
  assert.equal(result.status, 1);
  assert.ok(
    failures(result).some((entry) => entry.includes('missing cross-origin isolation COEP on /*')),
    result.stderr,
  );
});

// `max-age=31536001` is not less dangerous than exactly one year.
test('an immutable ttl beyond one year is rejected', async () => {
  const dir = await fixture((toml) => replaceOnce(toml,
    '  for = "/engines/*"\n  [headers.values]\n    Cache-Control = "public, max-age=0, must-revalidate"',
    '  for = "/engines/*"\n  [headers.values]\n    Cache-Control = "public, max-age=31536001, immutable"',
  ));
  const result = await runChecker(dir);
  assert.equal(result.status, 1);
  assert.ok(
    failures(result).some((entry) => entry.includes('immutable Cache-Control outside content-addressed artifacts: /engines/*')),
    result.stderr,
  );
});

test('a year-long ttl without immutable is still rejected on a mutable alias', async () => {
  const dir = await fixture((toml) => replaceOnce(toml,
    '  for = "/models/*"\n  [headers.values]\n    Cache-Control = "public, max-age=0, must-revalidate"',
    '  for = "/models/*"\n  [headers.values]\n    Cache-Control = "public, max-age=31536000"',
  ));
  const result = await runChecker(dir);
  assert.equal(result.status, 1);
  assert.ok(
    failures(result).some((entry) => entry.includes('at or beyond the one-year ceiling')),
    result.stderr,
  );
});

// Without a REQUIRED entry the gate only rejected the immutable combination, so
// an ordinary positive ttl let a stale logical URL stay fresh.
test('a mutable alias that stops revalidating is rejected', async () => {
  const dir = await fixture((toml) => replaceOnce(toml,
    '  for = "/engines/*"\n  [headers.values]\n    Cache-Control = "public, max-age=0, must-revalidate"',
    '  for = "/engines/*"\n  [headers.values]\n    Cache-Control = "public, max-age=86400"',
  ));
  const result = await runChecker(dir);
  assert.equal(result.status, 1);
  assert.ok(
    failures(result).some((entry) => entry.includes('missing engine alias revalidation Cache-Control')),
    result.stderr,
  );
});

// TOML permits keys in any order, so the prohibition cannot depend on the `.br`
// target appearing before `force = true`.
test('a forced .br redirect is rejected whatever order its keys are in', async () => {
  const dir = await fixture((toml) => `${toml}\n[[redirects]]\n  force = true\n  status = 200\n  from = "/lc0/net.onnx"\n  to = "/lc0/net.onnx.br"\n`);
  const result = await runChecker(dir);
  assert.equal(result.status, 1);
  assert.ok(
    failures(result).some((entry) => entry.includes('must not force a stable URL to a .br sidecar')),
    result.stderr,
  );
});

// smol-toml returns TomlDate instances, which are object-shaped but not tables.
test('a TOML date used as a header table is rejected, not read as empty', () => {
  const invalid = [];
  const blocks = parseHeaderBlocks('[[headers]]\nfor="/x"\nvalues=2026-01-01\n', (message) => invalid.push(message));
  assert.deepEqual(blocks[0].entries, []);
  assert.ok(invalid.some((entry) => entry.includes('[headers.values] must be a table')), invalid.join('; '));
});

// The gate only protects deploys whose build command runs it. Switching
// [build].command to an ungated builder is how R4's bypass would return.
test('a build command that does not run the gate is rejected', async () => {
  const dir = await fixture((toml) => replaceOnce(toml, 'npm run build:netlify:r2', 'npm run build:client'));
  const result = await runChecker(dir);
  assert.equal(result.status, 1);
  assert.ok(
    failures(result).some((entry) => entry.includes('does not run a gated build script')),
    result.stderr,
  );
});

// Netlify applies overlapping rules cumulatively (verified against production),
// so a more specific route redeclaring COEP loses isolation for that route while
// the /* block still reads correctly.
test('a route that weakens cross-origin isolation is rejected', async () => {
  const dir = await fixture((toml) => replaceOnce(toml,
    '  for = "/*.html"\n  [headers.values]',
    '  for = "/*.html"\n  [headers.values]\n    Cross-Origin-Embedder-Policy = "unsafe-none"',
  ));
  const result = await runChecker(dir);
  assert.equal(result.status, 1);
  assert.ok(
    failures(result).some((entry) => entry.includes('drops cross-origin isolation for that route')),
    result.stderr,
  );
});

// Netlify reads a _redirects file from the publish directory too, which the
// TOML redirect tables cannot express.
test('a forced .br rewrite in public/_redirects is rejected', async () => {
  const dir = await fixture();
  await writeFile(join(dir, 'public/_redirects'), '# comment\n/lc0/net.onnx  /lc0/net.onnx.br  200!\n');
  const result = await runChecker(dir);
  assert.equal(result.status, 1);
  assert.ok(
    failures(result).some((entry) => entry.includes('public/_redirects must not force a stable URL to a .br sidecar')),
    result.stderr,
  );
});

test('an unforced redirect in public/_redirects is allowed', async () => {
  const dir = await fixture();
  await writeFile(join(dir, 'public/_redirects'), '/old  /new  301\n');
  const result = await runChecker(dir);
  assert.equal(result.status, 0, result.stderr);
});

// Requirements are a subset test, so a non-weakening directive is not a failure.
test('a safe additive Cache-Control directive is accepted on a required path', async () => {
  const dir = await fixture((toml) => replaceOnce(toml,
    '  for = "/models/*"\n  [headers.values]\n    Cache-Control = "public, max-age=0, must-revalidate"',
    '  for = "/models/*"\n  [headers.values]\n    Cache-Control = "public, max-age=0, must-revalidate, no-transform"',
  ));
  const result = await runChecker(dir);
  assert.equal(result.status, 0, result.stderr);
});

// ...but an unlisted directive is a policy change and must be stated.
test('an unrecognised extra Cache-Control directive is rejected', async () => {
  const dir = await fixture((toml) => replaceOnce(toml,
    '  for = "/models/*"\n  [headers.values]\n    Cache-Control = "public, max-age=0, must-revalidate"',
    '  for = "/models/*"\n  [headers.values]\n    Cache-Control = "public, max-age=0, must-revalidate, s-maxage=86400"',
  ));
  const result = await runChecker(dir);
  assert.equal(result.status, 1);
  assert.ok(
    failures(result).some((entry) => entry.includes('adds s-maxage to the required Cache-Control on /models/*')),
    result.stderr,
  );
});
