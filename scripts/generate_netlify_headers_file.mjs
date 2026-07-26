#!/usr/bin/env node
// Renders public/_headers from the header tables in netlify.toml.
//
// Netlify treats netlify.toml header tables and a _headers file in the publish
// directory as alternative ways to say the same thing, and does not document
// which wins when both declare a path. Stating the policy once and generating
// the other file removes that ambiguity by construction, while keeping the
// _headers artifact in the published tree for any host that reads it.
import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { parseHeaderBlocks } from './netlify_headers.mjs';

const BANNER = [
  '# Generated from netlify.toml by scripts/generate_netlify_headers_file.mjs.',
  '# Do not edit by hand: edit the [[headers]] tables in netlify.toml and run',
  '#   npm run deploy:generate-headers',
  '# The deploy cache policy check fails if this file is out of date.',
].join('\n');

// `_headers` is line-oriented: a path sits at column 0 and its headers are
// indented. Anything interpolated into it must not be able to forge that
// structure. A TOML path of "/safe\n/other" would otherwise emit a second path
// line, and a leading space would turn a path into header content — in both
// cases the generated artifact means something different from the TOML it came
// from, while the byte-for-byte staleness check still passes.
function rejectUnserializable(block, invalid) {
  const { path, entries } = block;
  if (/[\r\n]/.test(path)) invalid.push(`path ${JSON.stringify(path)} contains a line break`);
  if (path !== path.trim()) invalid.push(`path ${JSON.stringify(path)} has leading or trailing whitespace`);
  if (!path.startsWith('/')) invalid.push(`path ${JSON.stringify(path)} must start with "/"`);
  for (const { name, value } of entries) {
    if (!/^[A-Za-z0-9-]+$/.test(name)) invalid.push(`${path}: header name ${JSON.stringify(name)} is not a token`);
    if (/[\r\n]/.test(value)) invalid.push(`${path}: value of ${name} contains a line break`);
  }
}

export function renderHeadersFile(tomlText) {
  const invalid = [];
  const sections = [];
  for (const block of parseHeaderBlocks(tomlText, (message) => invalid.push(message))) {
    if (block.path === undefined) {
      invalid.push('a [[headers]] table has no `for` path');
      continue;
    }
    rejectUnserializable(block, invalid);
    if (!block.entries.length) continue;
    sections.push([block.path, ...block.entries.map(({ name, value }) => `  ${name}: ${value}`)].join('\n'));
  }
  if (invalid.length) {
    throw new Error(`netlify.toml declares headers this generator cannot serve, so public/_headers cannot be derived:\n  ${invalid.join('\n  ')}`);
  }
  return `${BANNER}\n\n${sections.join('\n\n')}\n`;
}

export function headersFilePath(rootPath = process.cwd()) {
  return resolve(rootPath, 'public/_headers');
}

export function generateHeadersFile(rootPath = process.cwd()) {
  const rendered = renderHeadersFile(readFileSync(resolve(rootPath, 'netlify.toml'), 'utf8'));
  writeFileSync(headersFilePath(rootPath), rendered);
  return rendered;
}

// A generated file that is committed can drift from its source. The check is
// what makes the generation trustworthy rather than merely conventional.
export function checkHeadersFileUpToDate(rootPath = process.cwd()) {
  const rendered = renderHeadersFile(readFileSync(resolve(rootPath, 'netlify.toml'), 'utf8'));
  const onDisk = readFileSync(headersFilePath(rootPath), 'utf8');
  if (onDisk !== rendered) {
    throw new Error('public/_headers is out of date with netlify.toml; run `npm run deploy:generate-headers`');
  }
  return { ok: true, path: 'public/_headers' };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  try {
    generateHeadersFile();
    console.log(JSON.stringify({ ok: true, wrote: 'public/_headers' }, null, 2));
  } catch (error) {
    console.error(JSON.stringify({ ok: false, error: error.message }, null, 2));
    process.exit(1);
  }
}
