#!/usr/bin/env node
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname, resolve, relative } from 'node:path';

const SECTION_NAMES = new Map([
  [0, 'custom'], [1, 'type'], [2, 'import'], [3, 'function'], [4, 'table'], [5, 'memory'], [6, 'global'],
  [7, 'export'], [8, 'start'], [9, 'element'], [10, 'code'], [11, 'data'], [12, 'dataCount'],
]);

function readU32(bytes, cursor) {
  let result = 0;
  let shift = 0;
  let pos = cursor;
  while (pos < bytes.length) {
    const byte = bytes[pos++];
    result |= (byte & 0x7f) << shift;
    if ((byte & 0x80) === 0) return { value: result >>> 0, next: pos };
    shift += 7;
    if (shift > 35) throw new Error('invalid u32 LEB');
  }
  throw new Error('truncated u32 LEB');
}

function skipConstExpr(bytes, cursor, end) {
  let pos = cursor;
  while (pos < end) {
    const opcode = bytes[pos++];
    if (opcode === 0x0b) return pos;
    // i32.const/i64.const/global.get and most const-expression immediates are LEB encoded.
    if (opcode === 0x41 || opcode === 0x42 || opcode === 0x23) pos = readU32(bytes, pos).next;
    else if (opcode === 0x43) pos += 4;
    else if (opcode === 0x44) pos += 8;
  }
  throw new Error('unterminated const expr');
}

function parseDataSection(bytes, start, end) {
  let pos = start;
  const count = readU32(bytes, pos); pos = count.next;
  let payloadBytes = 0;
  const segmentPayloads = [];
  for (let i = 0; i < count.value; i += 1) {
    const flags = readU32(bytes, pos); pos = flags.next;
    if (flags.value === 0) {
      pos = skipConstExpr(bytes, pos, end);
    } else if (flags.value === 2) {
      const memoryIndex = readU32(bytes, pos); pos = memoryIndex.next;
      pos = skipConstExpr(bytes, pos, end);
    } else if (flags.value !== 1) {
      throw new Error(`unsupported data segment flags ${flags.value}`);
    }
    const len = readU32(bytes, pos); pos = len.next;
    payloadBytes += len.value;
    segmentPayloads.push(len.value);
    pos += len.value;
    if (pos > end) throw new Error('truncated data segment payload');
  }
  return { segmentCount: count.value, payloadBytes, largestSegmentBytes: Math.max(0, ...segmentPayloads) };
}

function inspect(file) {
  const bytes = readFileSync(file);
  if (bytes.length < 8 || bytes[0] !== 0 || bytes[1] !== 0x61 || bytes[2] !== 0x73 || bytes[3] !== 0x6d) {
    throw new Error(`${file} is not a wasm module`);
  }
  let pos = 8;
  const sections = [];
  while (pos < bytes.length) {
    const id = bytes[pos++];
    const size = readU32(bytes, pos); pos = size.next;
    const start = pos;
    const end = pos + size.value;
    if (end > bytes.length) throw new Error(`${file} section overruns file`);
    const section = {
      id,
      name: SECTION_NAMES.get(id) ?? `section-${id}`,
      bytes: size.value,
    };
    if (id === 11) Object.assign(section, parseDataSection(bytes, start, end));
    sections.push(section);
    pos = end;
  }
  const byName = Object.fromEntries(sections.map((section) => [section.name, section]));
  return {
    file,
    bytes: bytes.length,
    sections,
    codeBytes: byName.code?.bytes ?? 0,
    dataSectionBytes: byName.data?.bytes ?? 0,
    dataPayloadBytes: byName.data?.payloadBytes ?? 0,
    dataSegmentCount: byName.data?.segmentCount ?? 0,
    largestDataSegmentBytes: byName.data?.largestSegmentBytes ?? 0,
    dataPayloadRatio: (byName.data?.payloadBytes ?? 0) / bytes.length,
  };
}

const inputs = process.argv.slice(2).filter((arg) => arg !== '--out');
const outIndex = process.argv.indexOf('--out');
const out = outIndex >= 0 ? process.argv[outIndex + 1] : '';
if (outIndex >= 0) inputs.splice(inputs.indexOf(out), 1);
if (!inputs.length) {
  console.error('usage: node scripts/inspect_wasm_sections.mjs <module.wasm>... [--out report.json]');
  process.exit(2);
}
const cwd = process.cwd();
const report = {
  generatedAt: new Date().toISOString(),
  modules: inputs.map((input) => inspect(resolve(input))).map((entry) => ({ ...entry, file: relative(cwd, entry.file) })),
};
const text = `${JSON.stringify(report, null, 2)}\n`;
if (out) {
  mkdirSync(dirname(resolve(out)), { recursive: true });
  writeFileSync(resolve(out), text);
}
console.log(text);
