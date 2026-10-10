#!/usr/bin/env node
// Node cannot execute WebGPU. Print an expression for agent-browser eval --stdin
// on a secure origin (or a browser console). It uses the actual production WGSL
// and throws if equal-prior candidates differ from the stable JS control.
import { readFile } from 'node:fs/promises';

const source = await readFile(new URL('../src/lc0/wgslMatmulAddProbe.ts', import.meta.url), 'utf8');
const maxMoves = Number(source.match(/const WGSL_GPU_LEGAL_MAX_MOVES = (\d+);/)[1]);
const shader = source.match(/const WGSL_LEGAL_PRIORS_PROBE = `([\s\S]*?)`;/)[1].replaceAll('${WGSL_GPU_LEGAL_MAX_MOVES}', String(maxMoves));

console.log(`(async () => {
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('WebGPU adapter unavailable');
  const device = await adapter.requestDevice();
  const buffers = [];
  let readback;
  try {
    const upload = (values, usage) => {
      const buffer = device.createBuffer({ size: values.byteLength, usage, mappedAtCreation: true });
      buffers.push(buffer);
      new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(values.buffer));
      buffer.unmap();
      return buffer;
    };
    const logits = upload(new Float32Array([0, 0, 1]), GPUBufferUsage.STORAGE);
    const indices = upload(new Uint32Array([0, 1, 2]), GPUBufferUsage.STORAGE);
    const args = upload(new Float32Array([3, 1, 0, 0]), GPUBufferUsage.UNIFORM);
    const bytes = ${maxMoves} * 3 * 4;
    const output = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    readback = device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    buffers.push(output, readback);
    const module = device.createShaderModule({ code: ${JSON.stringify(shader)} });
    const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
    const bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [logits, indices, output, args].map((buffer, binding) => ({ binding, resource: { buffer } })) });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(1);
    pass.end();
    encoder.copyBufferToBuffer(output, 0, readback, 0, bytes);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(readback.getMappedRange());
    const actual = [result[0], result[3], result[6]];
    const expected = [0, 1, 2].sort((a, b) => [0, 0, 1][b] - [0, 0, 1][a]);
    if (JSON.stringify(actual) !== JSON.stringify(expected)) throw new Error('GPU tie order ' + JSON.stringify(actual) + ' differs from JS ' + JSON.stringify(expected));
    return { passed: true, actual, expected };
  } finally {
    readback?.unmap();
    for (const buffer of buffers) buffer.destroy();
    device.destroy();
  }
})()`);
