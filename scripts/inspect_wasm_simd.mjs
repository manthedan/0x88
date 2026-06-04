#!/usr/bin/env node
import { existsSync, readFileSync } from 'node:fs';

function readVarUint(bytes, cursor) {
  let value = 0;
  let shift = 0;
  let pos = cursor;
  while (true) {
    if (pos >= bytes.length) throw new Error('unexpected EOF in LEB128');
    const byte = bytes[pos++];
    value |= (byte & 0x7f) << shift;
    if ((byte & 0x80) === 0) return [value >>> 0, pos];
    shift += 7;
    if (shift > 35) throw new Error('LEB128 value is too large for this inspector');
  }
}

function readVarInt(bytes, cursor) {
  let shift = 0;
  let pos = cursor;
  while (true) {
    if (pos >= bytes.length) throw new Error('unexpected EOF in signed LEB128');
    const byte = bytes[pos++];
    if ((byte & 0x80) === 0) return pos;
    shift += 7;
    if (shift > 70) throw new Error('signed LEB128 value is too large for this inspector');
  }
}

function skipBlockType(bytes, pos) {
  const first = bytes[pos];
  // empty, value-type, and reference-type block signatures are a single byte.
  if (first === 0x40 || first === 0x7f || first === 0x7e || first === 0x7d || first === 0x7c || first === 0x7b || first === 0x70 || first === 0x6f) return pos + 1;
  // Otherwise it is a signed type index.
  return readVarInt(bytes, pos);
}

function findCodeSection(bytes) {
  if (bytes.length < 8 || bytes[0] !== 0x00 || bytes[1] !== 0x61 || bytes[2] !== 0x73 || bytes[3] !== 0x6d) {
    throw new Error('not a WebAssembly binary');
  }
  let pos = 8;
  while (pos < bytes.length) {
    const id = bytes[pos++];
    const [size, bodyStart] = readVarUint(bytes, pos);
    const bodyEnd = bodyStart + size;
    if (bodyEnd > bytes.length) throw new Error(`section ${id} overruns file`);
    if (id === 10) return bytes.subarray(bodyStart, bodyEnd);
    pos = bodyEnd;
  }
  throw new Error('code section not found');
}

const SIMD_NAMES = new Map([
  [0, 'v128.load'], [1, 'v128.load8x8_s'], [2, 'v128.load8x8_u'], [3, 'v128.load16x4_s'], [4, 'v128.load16x4_u'],
  [5, 'v128.load32x2_s'], [6, 'v128.load32x2_u'], [7, 'v128.load8_splat'], [8, 'v128.load16_splat'], [9, 'v128.load32_splat'], [10, 'v128.load64_splat'],
  [11, 'v128.store'], [12, 'v128.const'], [13, 'i8x16.shuffle'], [14, 'i8x16.swizzle'], [15, 'i8x16.splat'], [16, 'i16x8.splat'], [17, 'i32x4.splat'], [18, 'i64x2.splat'], [19, 'f32x4.splat'], [20, 'f64x2.splat'],
  [21, 'i8x16.extract_lane_s'], [22, 'i8x16.extract_lane_u'], [23, 'i8x16.replace_lane'], [24, 'i16x8.extract_lane_s'], [25, 'i16x8.extract_lane_u'], [26, 'i16x8.replace_lane'], [27, 'i32x4.extract_lane'], [28, 'i32x4.replace_lane'], [29, 'i64x2.extract_lane'], [30, 'i64x2.replace_lane'], [31, 'f32x4.extract_lane'], [32, 'f32x4.replace_lane'], [33, 'f64x2.extract_lane'], [34, 'f64x2.replace_lane'],
  [35, 'i8x16.eq'], [36, 'i8x16.ne'], [37, 'i8x16.lt_s'], [38, 'i8x16.lt_u'], [39, 'i8x16.gt_s'], [40, 'i8x16.gt_u'], [41, 'i8x16.le_s'], [42, 'i8x16.le_u'], [43, 'i8x16.ge_s'], [44, 'i8x16.ge_u'],
  [45, 'i16x8.eq'], [46, 'i16x8.ne'], [47, 'i16x8.lt_s'], [48, 'i16x8.lt_u'], [49, 'i16x8.gt_s'], [50, 'i16x8.gt_u'], [51, 'i16x8.le_s'], [52, 'i16x8.le_u'], [53, 'i16x8.ge_s'], [54, 'i16x8.ge_u'],
  [55, 'i32x4.eq'], [56, 'i32x4.ne'], [57, 'i32x4.lt_s'], [58, 'i32x4.lt_u'], [59, 'i32x4.gt_s'], [60, 'i32x4.gt_u'], [61, 'i32x4.le_s'], [62, 'i32x4.le_u'], [63, 'i32x4.ge_s'], [64, 'i32x4.ge_u'],
  [77, 'v128.not'], [78, 'v128.and'], [79, 'v128.andnot'], [80, 'v128.or'], [81, 'v128.xor'], [82, 'v128.bitselect'], [83, 'v128.any_true'],
  [84, 'v128.load8_lane'], [85, 'v128.load16_lane'], [86, 'v128.load32_lane'], [87, 'v128.load64_lane'], [88, 'v128.store8_lane'], [89, 'v128.store16_lane'], [90, 'v128.store32_lane'], [91, 'v128.store64_lane'], [92, 'v128.load32_zero'], [93, 'v128.load64_zero'],
  [94, 'f32x4.demote_f64x2_zero'], [95, 'f64x2.promote_low_f32x4'],
  [110, 'i8x16.add'], [113, 'i8x16.sub'], [142, 'i16x8.add'], [145, 'i16x8.sub'], [174, 'i32x4.add'], [177, 'i32x4.sub'], [228, 'f32x4.add'], [231, 'f32x4.sub'],
]);

function simdFamily(op) {
  const name = SIMD_NAMES.get(op);
  if (name) return name.split('.')[0];
  if (op <= 13) return 'v128';
  if (op <= 23) return 'i8x16';
  if (op <= 34) return 'lane';
  if (op <= 44) return 'i8x16';
  if (op <= 54) return 'i16x8';
  if (op <= 64) return 'i32x4';
  if (op <= 70) return 'float compare';
  if (op <= 93) return 'v128';
  if (op <= 123) return 'i8x16';
  if (op <= 169) return 'i16x8/integer';
  if (op <= 223) return 'i32x4/i64x2/integer';
  if (op <= 239) return 'f32x4';
  if (op <= 255) return 'f64x2';
  return 'simd extended';
}

function skipMemarg(bytes, pos) {
  [, pos] = readVarUint(bytes, pos); // align
  [, pos] = readVarUint(bytes, pos); // offset
  return pos;
}

function skipSimdImmediate(bytes, pos, subop) {
  if (subop <= 11 || (subop >= 92 && subop <= 93)) return skipMemarg(bytes, pos);
  if (subop === 12 || subop === 13) return pos + 16;
  if (subop >= 84 && subop <= 91) return skipMemarg(bytes, pos) + 1;
  if (subop >= 21 && subop <= 34) return pos + 1;
  return pos;
}

function scanInstructionStream(bytes, start, end, counts) {
  let pos = start;
  while (pos < end) {
    const opcode = bytes[pos++];
    switch (opcode) {
      case 0x02: case 0x03: case 0x04:
        pos = skipBlockType(bytes, pos); break;
      case 0x0c: case 0x0d: case 0x10: case 0x12: case 0x20: case 0x21: case 0x22: case 0x23: case 0x24: case 0x25: case 0x26: case 0xd2:
        [, pos] = readVarUint(bytes, pos); break;
      case 0x0e: {
        let count; [count, pos] = readVarUint(bytes, pos);
        for (let i = 0; i <= count; i += 1) [, pos] = readVarUint(bytes, pos);
        break;
      }
      case 0x11:
        [, pos] = readVarUint(bytes, pos); [, pos] = readVarUint(bytes, pos); break;
      case 0x1c: {
        let count; [count, pos] = readVarUint(bytes, pos);
        pos += count;
        break;
      }
      case 0x28: case 0x29: case 0x2a: case 0x2b: case 0x2c: case 0x2d: case 0x2e: case 0x2f:
      case 0x30: case 0x31: case 0x32: case 0x33: case 0x34: case 0x35: case 0x36: case 0x37:
      case 0x38: case 0x39: case 0x3a: case 0x3b: case 0x3c: case 0x3d: case 0x3e:
        pos = skipMemarg(bytes, pos); break;
      case 0x3f: case 0x40:
        pos += 1; break;
      case 0x41: case 0x42:
        pos = readVarInt(bytes, pos); break;
      case 0x43:
        pos += 4; break;
      case 0x44:
        pos += 8; break;
      case 0xd0:
        pos += 1; break;
      case 0xfc: {
        let subop; [subop, pos] = readVarUint(bytes, pos);
        if (subop === 8) { [, pos] = readVarUint(bytes, pos); [, pos] = readVarUint(bytes, pos); }
        else if (subop === 9 || subop === 11 || subop === 13 || subop === 15 || subop === 16 || subop === 17) [, pos] = readVarUint(bytes, pos);
        else if (subop === 10 || subop === 12 || subop === 14) { [, pos] = readVarUint(bytes, pos); [, pos] = readVarUint(bytes, pos); }
        break;
      }
      case 0xfd: {
        let subop; [subop, pos] = readVarUint(bytes, pos);
        counts.set(subop, (counts.get(subop) ?? 0) + 1);
        pos = skipSimdImmediate(bytes, pos, subop);
        break;
      }
      case 0xfe: {
        let subop; [subop, pos] = readVarUint(bytes, pos);
        if (subop === 0x03) pos += 1; // atomic.fence reserved byte
        else pos = skipMemarg(bytes, pos);
        break;
      }
      default:
        // Most MVP/reference/numeric opcodes have no immediates.
        break;
    }
  }
  if (pos !== end) throw new Error(`function body parse ended at ${pos}, expected ${end}`);
}

function scanSimd(code) {
  let pos = 0;
  let functionCount; [functionCount, pos] = readVarUint(code, pos);
  const counts = new Map();
  for (let fn = 0; fn < functionCount; fn += 1) {
    let bodySize; [bodySize, pos] = readVarUint(code, pos);
    const bodyEnd = pos + bodySize;
    if (bodyEnd > code.length) throw new Error(`function ${fn} overruns code section`);
    let localDecls; [localDecls, pos] = readVarUint(code, pos);
    for (let i = 0; i < localDecls; i += 1) {
      [, pos] = readVarUint(code, pos);
      pos += 1;
    }
    scanInstructionStream(code, pos, bodyEnd, counts);
    pos = bodyEnd;
  }
  return counts;
}

if (process.argv.length < 3) {
  console.error('Usage: node scripts/inspect_wasm_simd.mjs <file.wasm> [file2.wasm ...]');
  process.exit(2);
}

for (const file of process.argv.slice(2)) {
  if (!existsSync(file)) {
    console.log(`${file}`);
    console.log('  missing; build local artifacts first with npm run reckless:build-wasi and npm run reckless:build-simd-wasi');
    continue;
  }
  const bytes = readFileSync(file);
  const code = findCodeSection(bytes);
  const counts = scanSimd(code);
  const total = [...counts.values()].reduce((a, b) => a + b, 0);
  const familyCounts = new Map();
  for (const [op, count] of counts) familyCounts.set(simdFamily(op), (familyCounts.get(simdFamily(op)) ?? 0) + count);
  const families = [...familyCounts.entries()].sort((a, b) => b[1] - a[1]);
  const subops = [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 24);
  console.log(`${file}`);
  console.log(`  bytes=${bytes.length} codeBytes=${code.length} simdOpcodeCount=${total}`);
  console.log(`  families=${families.map(([name, count]) => `${name}:${count}`).join(', ') || 'none'}`);
  console.log(`  topOps=${subops.map(([op, count]) => `${SIMD_NAMES.get(op) ?? `simd.${op}`}:${count}`).join(', ') || 'none'}`);
}
