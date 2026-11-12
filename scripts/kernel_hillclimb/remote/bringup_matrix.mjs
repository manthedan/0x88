#!/usr/bin/env node
// Bring-up matrix: run the device probe (bench=0, adapter/features only) for every
// Chrome binary x flag set and append one summary line per config to --out (JSONL).
// On Yukon this must run INSIDE one gpu-queue job (it touches the GPU).
//   node bringup_matrix.mjs --root <dir-with-device-probe.html> --out matrix.jsonl --matrix <json>
// matrix json: { "binaries": {"name": "/path"}, "flagsets": {"name": ["--flag"]}, "page": "device-probe.html?bench=0" }
import { spawnSync } from 'node:child_process';
import { appendFileSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const a = Object.fromEntries(process.argv.slice(2).flatMap((x, i, arr) => (x.startsWith('--') ? [[x.slice(2), arr[i + 1]]] : [])));
const m = JSON.parse(readFileSync(a.matrix, 'utf8'));
const agent = join(dirname(fileURLToPath(import.meta.url)), 'agent.mjs');
for (const [bn, bin] of Object.entries(m.binaries)) {
  for (const [fn, flags] of Object.entries(m.flagsets)) {
    if (m.only && !m.only.some(([b, f]) => b === bn && f === fn)) continue;
    const r = spawnSync(
      process.execPath,
      [agent, '--root', a.root, '--page', m.page, '--chrome', bin, '--flags-json', JSON.stringify(flags), '--timeout', String(m.timeout ?? 40), '--gpu-info'],
      { encoding: 'utf8', timeout: 180_000 },
    );
    let j = null;
    try {
      j = JSON.parse(r.stdout.trim().split('\n').pop());
    } catch {}
    const w = j?.result?.webgpu;
    const gi = j?.meta?.gpuInfo;
    const line = {
      binary: bn,
      flagset: fn,
      exit: r.status,
      ok: j?.ok ?? false,
      error: j?.error ?? (r.stderr || '').slice(-500),
      product: j?.meta?.browser?.product,
      available: w?.available ?? null,
      reason: w?.reason,
      info: w?.info,
      want: w?.want,
      features: w?.features,
      fallback: w?.isFallbackAdapter,
      shaderChecks: w?.shaderChecks,
      deviceError: w?.deviceError,
      gpuDevices: gi?.gpu?.devices?.map((d) => ({
        vendorId: d.vendorId,
        deviceId: d.deviceId,
        vendorString: d.vendorString,
        deviceString: d.deviceString,
        driverVendor: d.driverVendor,
        driverVersion: d.driverVersion,
        active: d.active,
      })),
      featureStatus: gi?.gpu?.featureStatus,
      aux: gi?.gpu?.auxAttributes
        ? Object.fromEntries(
            Object.entries(gi.gpu.auxAttributes).filter(([k]) => /angle|gl|vulkan|skia|direct|sandbox|optimus|amd|vendor|renderer|version/i.test(k)),
          )
        : undefined,
      console: j?.meta?.console?.slice(-10),
      full: j,
    };
    appendFileSync(a.out, JSON.stringify(line) + '\n');
    console.log(
      `${bn} ${fn}: exit=${r.status} ok=${line.ok} ${w?.info ? `${w.info.vendor}/${w.info.architecture}/${w.info.description} backend=${w.info.backend ?? '?'} type=${w.info.type ?? '?'}` : (line.reason ?? line.error)}`,
    );
  }
}
