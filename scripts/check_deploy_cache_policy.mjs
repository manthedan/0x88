#!/usr/bin/env node
import { readFileSync } from 'node:fs';

const files = ['netlify.toml', 'public/_headers'];
const failures = [];

function fail(message) {
  failures.push(message);
}

function read(path) {
  return readFileSync(path, 'utf8');
}

function parseNetlifyHeaderBlocks(text) {
  const blocks = [];
  let current = null;
  for (const line of text.split(/\r?\n/)) {
    if (line.trim() === '[[headers]]') {
      current = { path: undefined, lines: [] };
      blocks.push(current);
      continue;
    }
    if (line.trim().startsWith('[[')) {
      current = null;
      continue;
    }
    if (!current) continue;
    current.lines.push(line);
    const match = line.match(/^\s*for\s*=\s*"([^"]+)"/);
    if (match) current.path = match[1];
  }
  return blocks;
}

function parsePublicHeaderBlocks(text) {
  const blocks = [];
  let current = null;
  for (const line of text.split(/\r?\n/)) {
    if (!line.trim()) continue;
    if (!line.startsWith(' ') && !line.startsWith('\t')) {
      current = { path: line.trim(), lines: [] };
      blocks.push(current);
      continue;
    }
    current?.lines.push(line);
  }
  return blocks;
}

const netlify = read('netlify.toml');
const publicHeaders = read('public/_headers');

if (/\[\[redirects\]\][\s\S]*?\.br[\s\S]*?force\s*=\s*true/.test(netlify)) {
  fail('netlify.toml must not force stable URLs to .br sidecars');
}

for (const file of files) {
  const text = read(file);
  if (/Content-Encoding\s*[:=]/.test(text)) {
    fail(`${file} must not declare Content-Encoding; serve compressed variants via CDN/object metadata or negotiated edge code`);
  }
}

const immutableAllowed = (path) => path === '/artifacts/sha256/*';
for (const block of parseNetlifyHeaderBlocks(netlify)) {
  const body = block.lines.join('\n');
  if (/Cache-Control\s*=\s*"[^"]*max-age=31536000[^"]*immutable/.test(body) && !immutableAllowed(block.path)) {
    fail(`netlify.toml has one-year immutable Cache-Control outside content-addressed artifacts: ${block.path ?? '(unknown)'}`);
  }
}
for (const block of parsePublicHeaderBlocks(publicHeaders)) {
  const body = block.lines.join('\n');
  if (/Cache-Control\s*:\s*.*max-age=31536000.*immutable/.test(body) && !immutableAllowed(block.path)) {
    fail(`public/_headers has one-year immutable Cache-Control outside content-addressed artifacts: ${block.path}`);
  }
}

function requireNetlifyHeader(path, pattern, description) {
  const block = parseNetlifyHeaderBlocks(netlify).find((candidate) => candidate.path === path);
  if (!block || !pattern.test(block.lines.join('\n'))) fail(`netlify.toml missing ${description} on ${path}`);
}

function requirePublicHeader(path, pattern, description) {
  const block = parsePublicHeaderBlocks(publicHeaders).find((candidate) => candidate.path === path);
  if (!block || !pattern.test(block.lines.join('\n'))) fail(`public/_headers missing ${description} on ${path}`);
}

requireNetlifyHeader('/*.html', /Cache-Control\s*=\s*"public, max-age=0, must-revalidate"/, 'HTML revalidation Cache-Control');
requireNetlifyHeader('/channels/*', /Cache-Control\s*=\s*"public, max-age=0, no-cache"/, 'channel revalidation Cache-Control');
requireNetlifyHeader('/artifacts/sha256/*', /Cache-Control\s*=\s*"public, max-age=31536000, immutable"/, 'content-addressed immutable Cache-Control');
requirePublicHeader('/*.html', /Cache-Control\s*:\s*public, max-age=0, must-revalidate/, 'HTML revalidation Cache-Control');
requirePublicHeader('/channels/*', /Cache-Control\s*:\s*public, max-age=0, no-cache/, 'channel revalidation Cache-Control');
requirePublicHeader('/artifacts/sha256/*', /Cache-Control\s*:\s*public, max-age=31536000, immutable/, 'content-addressed immutable Cache-Control');

if (failures.length) {
  console.error(JSON.stringify({ ok: false, failures }, null, 2));
  process.exit(1);
}

console.log(JSON.stringify({
  ok: true,
  checked: files,
  invariants: [
    'no forced .br redirects',
    'no custom Content-Encoding headers',
    'one-year immutable only on /artifacts/sha256/*',
    'HTML and channel pointers revalidate',
  ],
}, null, 2));
