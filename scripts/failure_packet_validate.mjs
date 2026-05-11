#!/usr/bin/env node
import { readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';

function flag(name) { return process.argv.includes(name); }
function argValues() { return process.argv.slice(2).filter((value) => !value.startsWith('--')); }
function isObject(x) { return x !== null && typeof x === 'object' && !Array.isArray(x); }
function fail(path, errors) {
  for (const error of errors) console.error(`ERROR ${path}: ${error}`);
}

const KINDS = new Set(['tactical_blunder', 'backend_drift', 'cache_drift', 'illegal_move', 'schema_violation', 'performance_regression', 'other']);
const SEVERITIES = new Set(['low', 'medium', 'high', 'critical']);

function validatePacket(packet) {
  const errors = [];
  if (packet.schema !== 'failure_packet_v1') errors.push('schema must be failure_packet_v1');
  for (const key of ['id', 'created_utc', 'kind', 'position', 'model', 'backend', 'repro']) {
    if (!(key in packet)) errors.push(`missing ${key}`);
  }
  if (typeof packet.id !== 'string' || !packet.id) errors.push('id must be non-empty string');
  if (typeof packet.created_utc !== 'string' || !packet.created_utc) errors.push('created_utc must be non-empty string');
  if (!KINDS.has(packet.kind)) errors.push(`invalid kind ${packet.kind}`);
  if ('severity' in packet && !SEVERITIES.has(packet.severity)) errors.push(`invalid severity ${packet.severity}`);
  if (!isObject(packet.position)) errors.push('position must be object');
  else {
    if (typeof packet.position.fen !== 'string' || !packet.position.fen) errors.push('position.fen must be non-empty string');
    if ('history_fens' in packet.position && !Array.isArray(packet.position.history_fens)) errors.push('position.history_fens must be array');
    if ('legal_uci' in packet.position && (!Array.isArray(packet.position.legal_uci) || !packet.position.legal_uci.every((m) => typeof m === 'string'))) errors.push('position.legal_uci must be string array');
  }
  if (!isObject(packet.model)) errors.push('model must be object');
  else if (typeof packet.model.id !== 'string' || !packet.model.id) errors.push('model.id must be non-empty string');
  if (!isObject(packet.backend)) errors.push('backend must be object');
  else if (typeof packet.backend.runtime !== 'string' || !packet.backend.runtime) errors.push('backend.runtime must be non-empty string');
  if (!isObject(packet.repro)) errors.push('repro must be object');
  else if (typeof packet.repro.command !== 'string' || !packet.repro.command) errors.push('repro.command must be non-empty string');
  if ('artifacts' in packet && (!Array.isArray(packet.artifacts) || !packet.artifacts.every((x) => typeof x === 'string'))) errors.push('artifacts must be string array');
  return errors;
}

const paths = argValues();
if (!paths.length) {
  console.error('usage: node --experimental-strip-types scripts/failure_packet_validate.mjs PACKET.json [PACKET.json...] [--print-repro] [--replay]');
  process.exit(2);
}

let ok = true;
for (const path of paths) {
  let packet;
  try { packet = JSON.parse(readFileSync(path, 'utf8')); }
  catch (error) { fail(path, [`JSON parse failed: ${error.message}`]); ok = false; continue; }
  const errors = validatePacket(packet);
  if (errors.length) { fail(path, errors); ok = false; continue; }
  console.log(`packet=${path} id=${packet.id} kind=${packet.kind} ok=1`);
  if (flag('--print-repro')) console.log(packet.repro.command);
  if (flag('--replay')) {
    const proc = spawnSync(packet.repro.command, { shell: true, stdio: 'inherit', env: { ...process.env, ...(packet.repro.env ?? {}) } });
    if (proc.status !== 0) ok = false;
  }
}
process.exit(ok ? 0 : 2);
